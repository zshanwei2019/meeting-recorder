#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""FunASR 字级时间戳转写的字间空格规整（纯逻辑回归测试）。

背景：开启 sentence_timestamp/pred_timestamp 的分离转写里，FunASR 把 token
用空格拼成句子，中文表现为每个字之间一个空格（“我 们 去 年”）。规整只压掉
CJK/全角字符两侧的空白，拉丁短语内部空格保留。

不加载 ASR 模型、不打开音频设备、不发网络请求。
运行：venv\\Scripts\\python.exe tests\\test_funasr_char_spacing.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app  # noqa: E402

PASS = []
FAIL = []


def check(name, cond, hint=""):
    if cond:
        PASS.append(name)
        print(f"  OK   {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL {name} {hint}")


print("=== 1. 纯中文逐字空格被压掉 ===")
check("基本拼接",
      app._normalize_funasr_char_spacing("我 们 去 年") == "我们去年")
check("含中文标点",
      app._normalize_funasr_char_spacing("你 好 ， 世 界 。") == "你好，世界。")
check("全角标点两侧",
      app._normalize_funasr_char_spacing("第 一 、 第 二") == "第一、第二")

print("=== 2. 中英混排：只压 CJK 侧空格 ===")
got = app._normalize_funasr_char_spacing("通 过 AI 的 形 式")
check("CJK 与拉丁相邻", got == "通过AI的形式", got)
got2 = app._normalize_funasr_char_spacing("这 是 RFID 盘 点 APP")
check("连续拉丁 token", got2 == "这是RFID盘点APP", got2)
got3 = app._normalize_funasr_char_spacing("三 d 打 印")
check("单字母夹中文", got3 == "三d打印", got3)

print("=== 3. 不该改动的情况 ===")
check("纯拉丁带空格保留",
      app._normalize_funasr_char_spacing("hello world") == "hello world")
check("空串安全", app._normalize_funasr_char_spacing("") == "")
check("无空格中文原样", app._normalize_funasr_char_spacing("正常文本") == "正常文本")
check("首尾空白被 strip",
      app._normalize_funasr_char_spacing("  你 好  ") == "你好")

print("=== 4. sentence_info 批量规整 ===")
sents = [
    {"text": "我 们 去 年", "spk": 0, "start": 0},
    {"text": "通 过 AI", "spk": 1, "start": 1000},
    {"sentence": "再 见", "spk": 0},
    {"text": "", "spk": 0},
]
out = app._normalize_sentence_info_spacing(sents)
check("text 字段规整", out[0]["text"] == "我们去年", out[0]["text"])
check("CJK-拉丁相邻", out[1]["text"] == "通过AI", out[1]["text"])
check("sentence 别名也规整", out[2]["sentence"] == "再见", out[2]["sentence"])
check("空文本不报错且保留", out[3]["text"] == "")
check("非中文内容不动其余字段", out[0]["spk"] == 0 and out[0]["start"] == 0)
check("None 输入安全", app._normalize_sentence_info_spacing(None) is None)

print()
if FAIL:
    print(f"FAILED: {len(FAIL)} / {len(PASS) + len(FAIL)}")
    for n in FAIL:
        print("  -", n)
    sys.exit(1)
print(f"ALL PASS: {len(PASS)}")
