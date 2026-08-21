"""音频混合与 PCM 转换的回归测试。

直接从 app.py 导入被测方法，避免测试与实现漂移（不复制任何实现逻辑）。
不打开任何音频设备、不需要 sounddevice：AudioRecorder 的 self.np 是惰性
注入的，测试里直接塞真 numpy 即可绕开 _import_deps()。

运行：.venv\\Scripts\\python.exe tests\\test_audio_mixing.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402

FAILED = []


def check(name, got, expect):
    if got == expect:
        print("OK   {}".format(name))
    else:
        print("FAIL {}\n       got={!r}\n     expect={!r}".format(name, got, expect))
        FAILED.append(name)


def check_true(name, cond):
    check(name, bool(cond), True)


def make_recorder():
    """构造一个不碰音频设备的 AudioRecorder"""
    r = app.AudioRecorder()
    r.np = np          # 绕过 _import_deps()，不需要 sounddevice
    return r


rec = make_recorder()
f32 = lambda *v: np.array(v, dtype=np.float32)


print("=== BUG1 复现：单流场景已消费数据必须移出缓冲 ===")
# 旧实现：mix_len = min(len(sys), len(mic)) == 0 时整段输出了 mic，
# 却按 mix_len==0 回写缓冲 → 一个采样都没消费 → 缓冲无限增长、音频复读。
sys_buf = f32()
mic_buf = f32()
total_in = 0
total_out = 0
for tick in range(1, 6):
    mic_buf = np.concatenate([mic_buf, np.full(100, tick, dtype=np.float32)])
    total_in += 100
    mixed, sys_buf, mic_buf = rec._drain_mix_buffers(sys_buf, mic_buf)
    total_out += len(mixed)

check("只有麦克风时：输出总量 == 输入总量（不复读）", total_out, total_in)
check("只有麦克风时：缓冲被清空（不无限增长）", len(mic_buf), 0)

# 反向：只有系统音频
sys_buf, mic_buf = f32(), f32()
total_in = total_out = 0
for tick in range(1, 6):
    sys_buf = np.concatenate([sys_buf, np.full(100, tick, dtype=np.float32)])
    total_in += 100
    mixed, sys_buf, mic_buf = rec._drain_mix_buffers(sys_buf, mic_buf)
    total_out += len(mixed)
check("只有系统音频时：输出总量 == 输入总量", total_out, total_in)
check("只有系统音频时：缓冲被清空", len(sys_buf), 0)


print()
print("=== 单流数据内容正确（顺序不乱、不重复） ===")
sys_buf, mic_buf = f32(), f32()
collected = []
for tick in range(1, 4):
    mic_buf = np.concatenate([mic_buf, f32(tick, tick)])
    mixed, sys_buf, mic_buf = rec._drain_mix_buffers(sys_buf, mic_buf)
    collected.extend(mixed.tolist())
check("单流透传保持原始顺序且无重复", collected, [1, 1, 2, 2, 3, 3])


print()
print("=== 双流混合 ===")
mixed, ns, nm = rec._drain_mix_buffers(f32(0.5, 0.5), f32(0.5, 0.5))
check("两路等长：各 *0.7 相加", [round(x, 4) for x in mixed.tolist()], [0.7, 0.7])
check("两路等长：sys 全消费", len(ns), 0)
check("两路等长：mic 全消费", len(nm), 0)

# 长度不等：只混对齐部分，长出来的必须留在缓冲里等下一轮
mixed, ns, nm = rec._drain_mix_buffers(f32(1, 1, 1, 1, 1), f32(1, 1))
check("长度不等：只输出对齐长度", len(mixed), 2)
check("长度不等：sys 剩余 3 个等下一轮", len(ns), 3)
check("长度不等：mic 已耗尽", len(nm), 0)

# 不等长时的守恒：sys 总量必须守恒（输出的 + 剩余的 == 输入的）
sys_in = f32(*([1.0] * 7))
mic_in = f32(*([1.0] * 3))
mixed, ns, nm = rec._drain_mix_buffers(sys_in, mic_in)
check("不等长守恒：sys 已消费 + 剩余 == 输入", len(mixed) + len(ns), len(sys_in))
check("不等长守恒：mic 已消费 + 剩余 == 输入", len(mixed) + len(nm), len(mic_in))


print()
print("=== 边界情况 ===")
mixed, ns, nm = rec._drain_mix_buffers(f32(), f32())
check("两路皆空：输出为空", len(mixed), 0)
check("两路皆空：不报错，缓冲仍为空", (len(ns), len(nm)), (0, 0))
check_true("返回的剩余缓冲是 float32（可继续 concatenate）",
           ns.dtype == np.float32 and nm.dtype == np.float32)


print()
print("=== BUG2 复现：int16 转换必须 clip，不能整数回绕 ===")
# 注释声称「各降0.7避免削波」，但 0.7+0.7=1.4 > 1.0 并未真正避免。
loud, _, _ = rec._drain_mix_buffers(f32(0.9, 0.9), f32(0.9, 0.9))
check_true("两路 0.9 混合后确实越界 >1.0", float(loud[0]) > 1.0)

pcm = np.frombuffer(rec._float_to_pcm16(loud), dtype=np.int16)
check_true("越界样本饱和为正的 32767，而非回绕成负数", int(pcm[0]) == 32767)
check_true("绝不出现反相尖峰（全部同号为正）", bool(np.all(pcm > 0)))

# 旧写法的回绕行为，证明这个测试确有意义
old = (loud * 32767).astype(np.int16)
check_true("旧写法确实回绕成负数（证明修复必要）", int(old[0]) < 0)

# 负向越界同样要饱和
neg = np.full(3, -1.4, dtype=np.float32)
pcm_neg = np.frombuffer(rec._float_to_pcm16(neg), dtype=np.int16)
check("负向越界饱和为 -32767", int(pcm_neg[0]), -32767)

# 正常范围内不应被改变
normal = f32(0.0, 0.5, -0.5)
pcm_n = np.frombuffer(rec._float_to_pcm16(normal), dtype=np.int16)
check("范围内数值保持线性映射", pcm_n.tolist(), [0, 16383, -16383])
check("输出字节数 = 采样数 * 2（16bit）", len(rec._float_to_pcm16(normal)), 3 * 2)
check("空输入返回空 bytes", rec._float_to_pcm16(f32()), b"")


print()
print("=== 所有 int16 转换都走了带 clip 的公共方法 ===")
import inspect  # noqa: E402
src = inspect.getsource(app.AudioRecorder)
raw = src.count("* 32767).astype")
check("全类只剩 1 处裸转换（即 _float_to_pcm16 内部）", raw, 1)
check_true("那一处紧跟在 clip 之后",
           "clipped = self.np.clip(samples, -1.0, 1.0)" in src
           and "(clipped * 32767).astype" in src)
# 三个原调用点：_resample_to_16k_mono / mix_worker / stop
check("_float_to_pcm16 被 3 处调用 + 1 处定义", src.count("_float_to_pcm16"), 4)

mix_src = src[src.index("def mix_worker"):]
mix_src = mix_src[:mix_src.index("\n            try:")]
check_true("mix_worker 不再自行回写缓冲切片",
           "self._mix_buffer['sys'] = sys_buf[mix_len:]" not in mix_src)
check_true("mix_worker 改为调用 _drain_mix_buffers",
           "_drain_mix_buffers" in mix_src)
check_true("mix_worker 对空输出提前跳过（避免 append 空帧）",
           "if len(mixed) == 0:" in mix_src)


print()
print("=== 长跑稳定性：缓冲不得无界增长 ===")
sys_buf, mic_buf = f32(), f32()
rng = np.random.default_rng(0)
fed = out = 0
peak = 0
for _ in range(300):
    # 模拟两路速率不同步（很常见：不同采样率 + 不同 blocksize）
    a = rng.integers(0, 160)
    b = rng.integers(0, 160)
    sys_buf = np.concatenate([sys_buf, np.ones(a, dtype=np.float32)])
    mic_buf = np.concatenate([mic_buf, np.ones(b, dtype=np.float32)])
    fed += a + b
    mixed, sys_buf, mic_buf = rec._drain_mix_buffers(sys_buf, mic_buf)
    out += len(mixed)
    peak = max(peak, len(sys_buf) + len(mic_buf))

print("     300 轮后残留 {} 采样，峰值残留 {}".format(len(sys_buf) + len(mic_buf), peak))
check_true("残留有界（不随轮次线性增长）", peak < 20000)
check_true("输出量不超过输入量（绝不复读）", out <= fed)


print()
if FAILED:
    print("失败 {} 项: {}".format(len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("全部通过")
