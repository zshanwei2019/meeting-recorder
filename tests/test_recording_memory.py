"""录音内存修复的回归测试（直接 import app.py）。

覆盖 probe 实测确认的三处问题：
  1. _frames 攒整场录音 + stop() 全量 concatenate → 内存与会议时长成正比，
     8 小时混合录音在"按下停止"那一刻 OOM，整场录音全丢
  2. _audio_queue 无界，唯一消费者是 _realtime_transcribe_task，
     只录音不开实时转写时只进不出（16kHz 单声道 int16 = 32 KB/s）
  3. np.interp 把 float32 静默提升成 float64（内存翻倍，且沿混音链传播）

不加载 ASR 模型、不打开音频设备、不发网络请求。
运行：.venv\\Scripts\\python.exe tests\\test_recording_memory.py
"""
import io
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app  # noqa: E402

PASS = []
FAIL = []
_CLEANUP = []


def check(name, got, expect):
    if got == expect:
        PASS.append(name)
        print(f"  OK   {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL {name}\n         got={got!r}\n         expect={expect!r}")


def check_true(name, cond, hint=""):
    check(name + (f" ({hint})" if hint else ""), bool(cond), True)


def make_recorder(actual_rate=48000, actual_ch=1):
    """造一个可用的 recorder，但不打开任何音频设备。

    只注入 numpy（_import_deps 平时才做这件事），绕开 sounddevice。
    """
    r = app.AudioRecorder()
    r.np = np
    r._actual_sample_rate = actual_rate
    r._actual_channels = actual_ch
    return r


def block(n, rate=48000, ch=1, value=0.5):
    """造一块 float32 音频，形状与真实回调一致：(frames, channels)。"""
    return np.full((n, ch), value, dtype=np.float32)


print("=== 1. 旧 bug 复现：内存与时长成正比（证明修复必要）===")
# 旧实现：每块都 append 进 _frames 且永不释放，stop() 再全量 concatenate。
# 这里按真实块大小（0.3 秒 @48kHz）模拟，用 nbytes 实测增长，不真分配 GB。
CHUNK = int(48000 * 0.3)
one = block(CHUNK)
per_block_bytes = one.nbytes
old_frames = []
for _ in range(200):
    old_frames.append(block(CHUNK))
old_total = sum(f.nbytes for f in old_frames)
check_true("旧写法 _frames 随块数线性增长",
           old_total == per_block_bytes * 200,
           f"200 块 = {old_total/1024/1024:.1f} MB")
# stop() 的连锁副本：concatenate 出一份完整副本，之后 *32767 / astype / tobytes
old_concat = np.concatenate(old_frames, axis=0)
check_true("旧 stop() 的 concatenate 又复制一整份",
           old_concat.nbytes == old_total,
           f"再 +{old_concat.nbytes/1024/1024:.1f} MB")
# 折算真实量级（48kHz float32 单声道）
h8 = 48000 * 4 * 3600 * 8
print(f"       → 折算 8 小时: _frames 约 {h8/1024**3:.2f} GB，"
      f"叠加 stop() 副本链后峰值数倍于此")
del old_frames, old_concat

print()
print("=== 2. _frames 现在有界（落盘线程卡死也不炸）===")
r = make_recorder()
# 故意不启动落盘线程，模拟"落盘线程已死"的最坏情况
for _ in range(_limit := app._WRITER_MAX_PENDING_BLOCKS * 5):
    r._append_frame(block(1000))
check_true("积压不超过上限",
           len(r._frames) <= app._WRITER_MAX_PENDING_BLOCKS,
           f"投 {_limit} 块，实际留 {len(r._frames)} 块")
check_true("超量部分被计入丢弃计数（不静默）", r._dropped_chunks > 0,
           f"丢弃 {r._dropped_chunks} 块")
check_true("上限是有限值", app._WRITER_MAX_PENDING_BLOCKS < 10000)

print()
print("=== 3. _audio_queue 现在有界，且满时丢最旧不阻塞 ===")
r2 = make_recorder()
check("队列上限已设", r2._audio_queue.maxsize, app._AUDIO_QUEUE_MAX_CHUNKS)
check_true("上限为正", app._AUDIO_QUEUE_MAX_CHUNKS > 0)
# 灌到 2 倍上限：不能抛异常、不能阻塞
for i in range(app._AUDIO_QUEUE_MAX_CHUNKS * 2):
    r2._enqueue_pcm(bytes([i % 256]) * 10)
check("队列不超上限", r2._audio_queue.qsize(), app._AUDIO_QUEUE_MAX_CHUNKS)
check_true("溢出被计数", r2._dropped_chunks > 0, f"丢弃 {r2._dropped_chunks} 块")
# 丢的必须是最旧的：最后进的应还在
last = r2._audio_queue.queue[-1]
check("保留的是最新数据", last, bytes([(app._AUDIO_QUEUE_MAX_CHUNKS * 2 - 1) % 256]) * 10)

print()
print("=== 4. np.interp 的 float64 静默提升已修 ===")
r3 = make_recorder()
mono = np.full(4800, 0.25, dtype=np.float32)
# 旧写法（复现）：np.interp 总返回 float64，即便输入是 float32
old_out = np.interp(np.linspace(0, len(mono) - 1, 1600), np.arange(len(mono)), mono)
check("旧写法确实提升为 float64（证明修复必要）", old_out.dtype, np.dtype("float64"))
new_out = r3._resample_linear_f32(mono, 1600)
check("新写法保持 float32", new_out.dtype, np.dtype("float32"))
check("每采样字节数减半", new_out.itemsize, 4)
check_true("数值无损", np.abs(new_out.astype(np.float64) - old_out).max() == 0.0)
check("空输入返回空 float32", r3._resample_linear_f32(mono, 0).dtype, np.dtype("float32"))
check("零长输入不抛异常", len(r3._resample_linear_f32(np.array([], dtype=np.float32), 10)), 0)
# 整条重采样链都不得产出 float64
check("_resample_to_16k_mono_float 保持 float32",
      r3._resample_to_16k_mono_float(block(4800)).dtype, np.dtype("float32"))

print()
print("=== 5. 边录边落盘：WAV 内容与采样数正确 ===")
r4 = make_recorder(actual_rate=16000, actual_ch=1)   # 免重采样，便于精确计数
r4._recording = True
path = r4._open_wav_writer()
_CLEANUP.append(path)
check_true("start 阶段就已创建文件", Path(path).exists(), Path(path).name)
# 分 10 次写入，每次 1600 采样（模拟 0.1 秒 @16kHz）
for _ in range(10):
    r4._append_frame(block(1600, rate=16000))
written = r4._write_blocks_to_wav(r4._take_pending_frames())
check("落盘采样数正确", written, 16000)
check("交接缓冲已被取空", len(r4._frames), 0)
out_path = r4.stop()
check_true("stop 返回文件路径", out_path is not None)
with wave.open(str(out_path), "rb") as wf:
    check("单声道", wf.getnchannels(), 1)
    check("16bit", wf.getsampwidth(), 2)
    check("16kHz", wf.getframerate(), 16000)
    check("WAV 总采样数与写入量一致", wf.getnframes(), 16000)
    data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
check_true("音频内容非静音", int(np.abs(data).max()) > 0, f"峰值 {int(np.abs(data).max())}")
check_true("未发生 int16 回绕（无反相尖峰）", int(data.min()) > 0)

print()
print("=== 6. stop() 的尾巴不丢：未落盘的残留会被写完 ===")
r5 = make_recorder(actual_rate=16000, actual_ch=1)
r5._recording = True
p5 = r5._open_wav_writer()
_CLEANUP.append(p5)
r5._write_blocks_to_wav(r5._take_pending_frames())     # 先落一部分（此时为空）
for _ in range(3):
    r5._append_frame(block(1600, rate=16000))          # 这 3 块还没落盘
out5 = r5.stop()                                        # stop 必须把它们写完
with wave.open(str(out5), "rb") as wf:
    check("stop 收尾写完残留块", wf.getnframes(), 4800)

print()
print("=== 7. 一帧未录到：返回 None 且不留空壳文件 ===")
r6 = make_recorder()
r6._recording = True
p6 = r6._open_wav_writer()
check_true("空壳文件先被创建", Path(p6).exists())
out6 = r6.stop()
check("无数据时返回 None（与旧行为一致）", out6, None)
check_true("空壳文件已被清理", not Path(p6).exists(), str(p6))

print()
print("=== 8. 启动失败时不留垃圾文件 ===")
r7 = make_recorder()
p7 = r7._open_wav_writer()
check_true("文件已创建", Path(p7).exists())
r7._discard_wav_writer()
check_true("_discard_wav_writer 删除文件", not Path(p7).exists())
check("writer 已置空", r7._wav_writer, None)

print()
print("=== 9. 静态断言：防回归 ===")
src = io.open(ROOT / "app.py", encoding="utf-8").read()
check("stop() 不再全量 concatenate _frames",
      src.count("self.np.concatenate(self._frames"), 0)
check_true("有增量写入器", "_open_wav_writer" in src)
check_true("有落盘线程", "_writer_loop" in src)
check_true("回调改走有界入口 _append_frame", "self._append_frame(" in src)
check("回调不再直接 append _frames",
      src.count("self._frames.append(audio_data)")
      + src.count("self._frames.append(mixed.reshape(-1, 1))"), 0)
check("队列不再无界 put", src.count("self._audio_queue.put(pcm)"), 0)
check_true("队列改走 _enqueue_pcm", "self._enqueue_pcm(pcm)" in src)
check_true("stop 会 join 混音线程（否则写已关闭文件）",
           "self._mix_thread" in src and "join(timeout=" in src)
# 所有重采样必须走 float32 包装器；裸 np.interp 只允许出现在包装器内部
bare = [l for l in src.splitlines()
        if "np.interp(" in l and "_resample_linear_f32" not in l]
check("裸 np.interp 仅剩包装器内部一处", len(bare), 1)

print()
for p in _CLEANUP:
    try:
        Path(p).unlink()
    except Exception:
        pass

print(f"通过 {len(PASS)} 条，失败 {len(FAIL)} 条")
if FAIL:
    for f in FAIL:
        print(f"  - {f}")
sys.exit(1 if FAIL else 0)
