# -*- coding: utf-8 -*-
"""音源设备匹配纯函数测试：_find_loopback_device / _find_mic_device。

覆盖 Windows 中文声卡名、macOS 英文内建名、BlackHole/Soundflower 虚拟声卡、
系统默认输入兜底（且 BlackHole 不得被误判为麦克风）。纯函数，CI 无声卡可跑。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app   # noqa: E402

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        print(f"  OK   {name}")
        passed += 1
    else:
        print(f"  FAIL {name} {detail}")
        failed += 1


def dev(i, name, ch=2, sr=48000):
    return {"id": i, "name": name, "channels": ch, "sample_rate": sr}


WIN_DEVICES = [
    dev(0, "麦克风阵列 (Realtek High Definition Audio)", 2, 48000),
    dev(1, "立体声混音 (Realtek High Definition Audio)", 2, 48000),
    dev(2, "线路输入 (Realtek)", 2, 48000),
]

MAC_DEVICES = [
    dev(0, "MacBook Air Microphone", 1, 48000),
    dev(1, "MacBook Air Speakers", 2, 48000),
]

MAC_BLACKHOLE = MAC_DEVICES + [dev(2, "BlackHole 2ch", 2, 48000)]

print("=== 1. 回录设备（loopback）===")
check("Windows 立体声混音命中",
      app._find_loopback_device(WIN_DEVICES)["id"] == 1)
check("macOS 无虚拟声卡 -> None",
      app._find_loopback_device(MAC_DEVICES) is None)
check("macOS BlackHole 命中",
      app._find_loopback_device(MAC_BLACKHOLE)["id"] == 2)
check("Soundflower 命中",
      app._find_loopback_device([dev(0, "Soundflower (2ch)", 2, 48000)]) is not None)
check("大小写不敏感 BLACKHOLE",
      app._find_loopback_device([dev(0, "BLACKHOLE 16ch", 16, 48000)]) is not None)
check("空列表 -> None", app._find_loopback_device([]) is None)

print("\n=== 2. 麦克风 ===")
check("Windows 中文麦克风阵列优先",
      app._find_mic_device(WIN_DEVICES)["id"] == 0)
check("macOS 英文内建麦克风命中（旧逻辑找不到的坑）",
      app._find_mic_device(MAC_DEVICES)["id"] == 0)
check("装了 BlackHole 仍选对真麦克风",
      app._find_mic_device(MAC_BLACKHOLE)["id"] == 0)
check("系统默认输入兜底（无可识别名时）",
      app._find_mic_device([dev(3, "未知USB音频设备", 2, 44100)],
                           default_input_id=3)["id"] == 3)
check("兜底排除回录设备：默认输入是 BlackHole 时 -> None",
      app._find_mic_device([dev(3, "BlackHole 2ch", 2, 48000)],
                           default_input_id=3) is None)
check("default_input_id=-1（无声卡）不炸 -> None",
      app._find_mic_device([dev(3, "未知USB音频设备", 2, 44100)],
                           default_input_id=-1) is None)
check("空列表 -> None", app._find_mic_device([], default_input_id=0) is None)
check("channels>4 的内建设备跳过，走默认兜底",
      app._find_mic_device([dev(0, "MacBook Air Microphone", 8, 48000),
                            dev(1, "未知设备", 2, 48000)],
                           default_input_id=1)["id"] == 1)

print(f"\n通过 {passed} 条，失败 {failed} 条")
sys.exit(1 if failed else 0)
