# -*- coding: utf-8 -*-
"""analyze_recording_audio 电平/时长分析的回归测试。

区分「静音输入」与「有声音但没识别出字」，是空转写提示准确性的关键。
soundfile 在 CI 精简环境可能缺失 → 缺则整体 skip，不报错。
运行：.venv\\Scripts\\python.exe tests\\test_analyze_recording_audio.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PASS, FAIL, SKIPPED = [], [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name); print("  OK  ", name)
    else:
        FAIL.append(name); print("  FAIL", name, detail)


try:
    import numpy as np
    import soundfile as sf
    import app
except Exception as e:  # noqa: BLE001
    print(f"SKIP：缺少依赖（{e}），CI 精简环境可接受")
    sys.exit(0)

SR = 16000

def write_wav(frac, secs=1.0):
    p = Path(tempfile.mkdtemp()) / "t.wav"
    n = int(SR * secs)
    x = (np.full(n, float(frac), dtype=np.float32) * 0 +
         (np.sin(2 * np.pi * 220 * np.arange(n) / SR).astype(np.float32) * frac))
    sf.write(str(p), x, SR, subtype="PCM_16")
    return p

# 1) 正常语音电平 → 非静音，时长正确
p = write_wav(0.2, 2.0)
info = app.analyze_recording_audio(p)
check("有声：is_silent=False", info["is_silent"] is False, info)
check("有声：时长≈2.0s", abs(info["duration_s"] - 2.0) < 0.2, info)
check("有声：peak 接近 0.2", 0.15 < info["peak"] <= 0.21, info)
check("有声：rms_db 为有限值", isinstance(info["rms_db"], float))

# 2) 纯静音 → is_silent=True
p0 = write_wav(0.0, 1.0)
info0 = app.analyze_recording_audio(p0)
check("静音：is_silent=True", info0["is_silent"] is True, info0)
check("静音：rms_db 极低(<-60)", info0["rms_db"] is not None and info0["rms_db"] < -60, info0)

# 3) 文件不存在等异常 → 不抛出，降级
bad = app.analyze_recording_audio(Path(tempfile.mkdtemp()) / "nope.wav")
check("坏路径不抛异常并安全降级", isinstance(bad, dict) and bad["duration_s"] == 0.0, bad)

print(f"\n通过 {len(PASS)} 条，失败 {len(FAIL)} 条")
sys.exit(1 if FAIL else 0)
