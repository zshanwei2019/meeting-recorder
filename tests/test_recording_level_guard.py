# -*- coding: utf-8 -*-
"""AudioRecorder 实时电平表 / 开场静默守护状态机的回归测试。

只测纯逻辑：手动构造 16kHz 单声道 int16 PCM 块喂给 _update_level，
不打开任何音频设备、不加载模型、不依赖 soundfile/torch。
运行：.venv\\Scripts\\python.exe tests\\test_recording_level_guard.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import app  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name); print("  OK  ", name)
    else:
        FAIL.append(name); print("  FAIL", name, detail)


def pcm(frac):
    """造一块 0.3s 的 16k 单声道 int16 PCM，幅度 = frac（-1..1）。"""
    n = int(16000 * 0.3)
    x = np.full(n, float(frac), dtype=np.float32)
    return (np.clip(x, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def fresh_recorder():
    rec = app.AudioRecorder()     # __init__ 不导入 sounddevice
    rec.np = np                   # _update_level 只需要 numpy
    # 模拟 start() 里的电平状态复位（不真正开设备）
    rec._recording = True
    rec._meter = 0.0
    rec._last_chunk_peak = 0.0
    rec._last_level_ts = None
    rec._cap_start = time.time()
    rec._cap_max_peak = 0.0
    rec._cap_voiced = False
    rec._silence_warned = False
    return rec


# 1) 有声输入：热身期内不预警；热身后也不预警
r = fresh_recorder()
r._update_level(pcm(0.2))
snap = r.silence_snapshot()
check("有声：快照返回 dict", isinstance(snap, dict))
check("有声：热身期内 state 为空（不打扰）", snap["state"] == "", snap)
r._cap_start = time.time() - 6.0   # 模拟已过 6 秒
snap = r.silence_snapshot()
check("有声：超过 5 秒仍不预警", snap["state"] == "", snap)

# 2) 纯静音：热身后预警；一旦出现声音转 clear
r = fresh_recorder()
r._update_level(pcm(0.0))
r._cap_start = time.time() - 6.0
snap = r.silence_snapshot()
check("静音：超 5 秒触发 warn", snap["state"] == "warn", snap)
r.mark_silence_warned()
# 仍在静音 → 维持 ''（前端保持红条常亮，不重复告警）
check("静音：已预警后继续静音维持空状态", r.silence_snapshot()["state"] == "")
# 来了一段正常说话电平
r._update_level(pcm(0.25))
snap = r.silence_snapshot()
check("恢复声音：状态转 clear", snap["state"] == "clear", snap)

# 3) 电平表：喂大声块后读数明显抬升
r = fresh_recorder()
before = r.get_level()
r._update_level(pcm(0.3))
after = r.get_level()
check("电平表：无声读数为 0", before == 0.0, before)
check("电平表：有声后读数抬升(>0.1)", after > 0.1, after)
check("电平表：读数不超过 1", r.get_level() <= 1.0)

# 4) 数字静音（实测 -90dB 量级的底噪）不应被当成有声
r = fresh_recorder()
r._update_level(pcm(0.0001))   # 约 -80dB，远低于 _LEVEL_GUARD_PEAK=0.01
r._cap_start = time.time() - 6.0
snap = r.silence_snapshot()
check("数字底噪(-80dB)：仍判为静默并预警", snap["state"] == "warn", snap)

# 5) 阈值常量健全
check("静默峰值阈值在合理区间", 0.0 < app._LEVEL_GUARD_PEAK < 0.05, app._LEVEL_GUARD_PEAK)
check("热身时间为正且在 2~15 秒", 2.0 <= app._LEVEL_WARMUP_S <= 15.0, app._LEVEL_WARMUP_S)

# 6) 不在录音时 silence_snapshot/get_level 的行为
r = fresh_recorder()
r._recording = False
check("未录音：silence_snapshot 返回 None", r.silence_snapshot() is None)
check("未录音：电平读数为 0", r.get_level() == 0.0)

# 7) 坏数据不得抛异常（音频回调安全）
r = fresh_recorder()
threw = False
try:
    r._update_level(b"")          # 空缓冲
    r._update_level(b"\x00\x01")  # 奇数字节/垃圾
except Exception as e:
    threw = True
check("坏 PCM 不抛异常", not threw)

print(f"\n通过 {len(PASS)} 条，失败 {len(FAIL)} 条")
sys.exit(1 if FAIL else 0)
