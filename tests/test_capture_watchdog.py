# -*- coding: utf-8 -*-
"""采集看门狗（僵尸流检测 / 自动重开）回归测试。

只测纯逻辑与状态机：不开任何音频设备、不加载模型。
运行：.venv\\Scripts\\python.exe -X utf8 tests\\test_capture_watchdog.py
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


C = app.AudioRecorder._classify_capture
T0 = 1000.0

# ── 1. 纯函数分类器 ──

# 1.1 暂停 / 未开流：永远不判故障
check("暂停时不判故障", C(T0, T0, None, None, True) is None)
check("open_ts=None 不判故障", C(T0, None, None, None, False) is None)

# 1.2 正常有数据：块在到、且有声
check("正常有声不判故障",
      C(T0, T0 - 20, T0 - 0.1, T0 - 0.1, False) is None)

# 1.3 no-data：开流超过阈值仍没收到任何块
r = C(T0 + 4, T0, None, None, False,
      warmup=10, no_data=3, digital=8)
check("开流 4s 无任何块 => no-data", r == "no-data", r)
r = C(T0 + 2, T0, None, None, False,
      warmup=10, no_data=3, digital=8)
check("开流 2s 无块（阈值内）=> None", r is None, r)

# 1.4 no-data：块曾经到过但中途停了（僵尸流典型形态）。
# now=T0+10，最后一个块在 now-4s 到，之后再无块
r = C(T0 + 10, T0 - 30, T0 + 6, T0 + 6, False,
      warmup=10, no_data=3, digital=8)
check("块中断 4s => no-data", r == "no-data", r)

# 1.5 digital-silence：块在到（0.1s 前），但过了 warmup 后
# 最后一个“有声块”距今超过 digital 阈值
r = C(T0 + 30, T0 - 30, T0 + 29.9, T0 + 20, False,
      warmup=10, no_data=3, digital=8)
check("持续数字静音 => digital-silence", r == "digital-silence", r)

# 1.6 warmup 内不判 digital-silence（给开场/设备初始化留时间）
r = C(T0 + 4, T0 - 5, T0 + 3.9, T0 - 5, False,
      warmup=10, no_data=3, digital=8)
check("warmup 内不判 digital-silence", r is None, r)

# 1.7 从来没响过：last_loud=None，以开流时间为参照。过 warmup 即报警
r = C(T0 + 11, T0, T0 + 10.9, None, False,
      warmup=10, no_data=3, digital=8)
check("从未有声且超过 warmup => digital-silence",
      r == "digital-silence", r)
r = C(T0 + 9, T0, T0 + 8.9, None, False,
      warmup=10, no_data=3, digital=8)
check("从未有声但仍在 warmup 内 => None", r is None, r)

# 1.8 安静房间但流活着（真实底噪峰值 >= floor）：last_loud 持续刷新
check("真实底噪持续刷新 last_loud => 不判故障",
      C(T0 + 600, T0 - 600, T0 + 599.9, T0 + 599.9, False) is None)


# ── 2. 状态机：重开次数上限、冷却、dead ──

def fresh_recorder():
    rec = app.AudioRecorder()
    rec.np = np
    rec._recording = True
    rec._paused = False
    rec._source = "both"
    rec._capture_fault = ""
    rec._restart_count = 0
    rec._last_restart_ts = 0.0
    rec._capture_open_ts = T0 - 30
    rec._last_chunk_ts = None
    rec._last_loud_ts = None
    rec._open_calls = []

    def fake_open(source):
        rec._open_calls.append(source)
        rec._capture_open_ts = time.time()
        rec._last_chunk_ts = None
        rec._last_loud_ts = None
        return True

    rec._open_capture = fake_open
    rec._close_streams = lambda: None
    return rec


def feed_loud(rec):
    n = int(16000 * 0.3)
    pcm = (np.full(n, 0.2, dtype=np.float32) * 32767).astype(np.int16).tobytes()
    rec._update_level(pcm)


# 2.1 第一次故障：重开一次，进入 restarting
rec = fresh_recorder()
rec._restart_capture("no-data", now=T0)
check("首次故障触发重开", rec._open_calls == ["both"], rec._open_calls)
check("首次后 restart_count=1", rec._restart_count == 1, rec._restart_count)
check("状态=restarting", rec._capture_fault == "restarting", rec._capture_fault)

# 2.2 冷却期内不重复重开
rec._restart_capture("no-data", now=T0 + 2)
check("冷却期内不重复重开", rec._open_calls == ["both"], rec._open_calls)

# 2.3 重开后收到真实音频 => recovered（走 _update_level 路径）
feed_loud(rec)
check("重开后有声 => recovered", rec._capture_fault == "recovered",
      rec._capture_fault)

# 2.4 用满次数后再故障 => dead，不再调用 open
rec2 = fresh_recorder()
rec2._restart_capture("no-data", now=T0)
rec2._restart_capture("no-data", now=T0 + 10)
feed_loud(rec2)  # 第二次重开后有声 => recovered
check("两次重开计数=2", rec2._restart_count == 2, rec2._restart_count)
# 再次故障（超过上限）
rec2._capture_fault = ""  # 模拟 recovered 后又断
rec2._restart_capture("digital-silence", now=T0 + 60)
check("超过重开上限 => dead", rec2._capture_fault == "dead", rec2._capture_fault)
check("dead 后不再 open（仍 2 次）", len(rec2._open_calls) == 2, rec2._open_calls)

# 2.5 未在录音时不重开
rec3 = fresh_recorder()
rec3._recording = False
rec3._restart_capture("no-data", now=T0)
check("未录音不重开", rec3._open_calls == [], rec3._open_calls)


# ── 3. _watchdog_tick 串联：僵尸流场景自动重开 ──

rec4 = fresh_recorder()
rec4._capture_open_ts = T0 - 30
rec4._last_chunk_ts = T0     # 块在 now-10s 后再没到
rec4._watchdog_tick(now=T0 + 10)
check("watchdog tick 检出僵尸流并重开", rec4._open_calls == ["both"],
      rec4._open_calls)

# 重开验证窗口内不再立即重判
rec4._last_chunk_ts = None
rec4._watchdog_tick(now=T0 + 12)
check("重开验证窗口内不重判", len(rec4._open_calls) == 1, rec4._open_calls)

# 暂停时不触发
rec5 = fresh_recorder()
rec5._paused = True
rec5._last_chunk_ts = T0 - 30
rec5._watchdog_tick(now=T0)
check("暂停时不重开", rec5._open_calls == [], rec5._open_calls)


# ── 4. 常量合理性（防止误调导致误杀正常录音）──
check("no-data 阈值 1.5~10s",
      1.5 <= app._CAPTURE_STALE_NO_DATA_S <= 10, app._CAPTURE_STALE_NO_DATA_S)
check("digital-silence 阈值 5~30s",
      5 <= app._CAPTURE_STALE_DIGITAL_S <= 30, app._CAPTURE_STALE_DIGITAL_S)
check("数字底噪阈值远低于真实房间底噪（实测 0.17）",
      0 < app._CAPTURE_DIGITAL_FLOOR < 0.01, app._CAPTURE_DIGITAL_FLOOR)
check("自动重开上限 1~5 次",
      1 <= app._CAPTURE_MAX_RESTARTS <= 5, app._CAPTURE_MAX_RESTARTS)

print(f"\n通过 {len(PASS)} 条，失败 {len(FAIL)} 条")
sys.exit(1 if FAIL else 0)
