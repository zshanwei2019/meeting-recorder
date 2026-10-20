# -*- coding: utf-8 -*-
"""语种自动路由 + 英文词级说话人对齐单测（不装重依赖也能跑）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import app

PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(("PASS " if cond else "FAIL ") + name + (f"  [{extra}]" if extra and not cond else ""))


# ── classify_script ──
check("空文本 -> empty", app.classify_script("") == "empty")
check("纯标点/数字 -> empty", app.classify_script("123, 9100 !!!") == "empty")
check("纯中文 -> zh", app.classify_script("我们今天讨论刀具修磨和质量体系") == "zh")
check("中文夹英文术语仍 -> zh（术语不主导）",
      app.classify_script("我们导入了 AS9100D 和 RFID 流程盘点库存") == "zh")
check("纯英文 -> latin",
      app.classify_script("Next conversation Navneet is with us here in our studios") == "latin")
check("中英大段混合 -> mixed",
      app.classify_script("这个 market 我们觉得 valuation 偏高但 growth 还在继续") == "mixed")
check("英文+数字型号 -> latin",
      app.classify_script("we use RFID and AS9100D standard every day here") == "latin")

# ── route_engine_for_script ──
check("zh -> FunASR", app.route_engine_for_script("zh") == "FunASR")
check("latin -> Qwen3", app.route_engine_for_script("latin") == "Qwen3")
check("mixed -> FunASR（中文为主，吃热词）", app.route_engine_for_script("mixed") == "FunASR")
check("empty -> FunASR（安全回退，不乱送）", app.route_engine_for_script("empty") == "FunASR")

# ── 英文词级说话人对齐 ──
# tracks 秒；words 秒
tracks = [
    (0.0, 2.0, "SPEAKER_00"),
    (2.0, 4.0, "SPEAKER_01"),
    (4.0, 6.0, "SPEAKER_00"),
]
words = [
    (0.1, 0.5, "hello"), (0.6, 1.0, "world"),     # spk0
    (2.1, 2.5, "good"), (2.6, 3.0, "morning"),    # spk1
    (4.1, 4.5, "nice"), (4.6, 5.0, "day"),        # spk0
]
sents, label_map = app._assign_speakers_to_words(words, tracks)
check("英文对齐：合并成 3 段", len(sents) == 3, f"got {len(sents)}")
check("label_map 两个说话人", len(label_map) == 2, str(label_map))
spks = [s["spk"] for s in sents]
check("说话人序列 0,1,0", spks == [0, 1, 0], str(spks))
check("第一段文本", sents[0]["text"] == "hello world", sents[0]["text"])
check("第二段文本", sents[1]["text"] == "good morning", sents[1]["text"])
check("第三段文本", sents[2]["text"] == "nice day", sents[2]["text"])
check("时间戳为毫秒且段内数量=词数",
      sents[0]["start"] == 100 and sents[0]["end"] == 1000
      and len(sents[0]["timestamp"]) == 2,
      str(sents[0]))
check("同一说话人连续词合并为一段",
      app._assign_speakers_to_words(
          [(0.1, 0.4, "a"), (0.5, 0.8, "b"), (0.9, 1.2, "c")],
          [(0.0, 2.0, "S0")])[0][0]["text"] == "a b c")

# ── 空输入稳健性 ──
e_sents, _e_map = app._assign_speakers_to_words([], tracks)
check("空词 -> 空句子", e_sents == [])

print(f"\n通过 {len(PASS)} 条，失败 {len(FAIL)} 条")
if FAIL:
    print("失败项:", ", ".join(FAIL))
    sys.exit(1)
