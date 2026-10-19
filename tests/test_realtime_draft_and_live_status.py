#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""实时转写草稿落盘 + 录音库实时状态标记的回归测试。

覆盖两件事：
1. _save_realtime_draft 在录音进行中写 WAV 同名 .meta.json（draft=true，
   has_transcript 随文本），原子写；最终 save_recording_meta 覆盖后 draft 消失。
2. list_recordings 对 state.live_wav_name 指向的录音标 live_stage，
   使前端列表显示“实时转写中/转写完成·分离中”而不是“未转写”。

不加载 ASR 模型、不打开音频设备、不发网络请求。
运行：venv\\Scripts\\python.exe tests\\test_realtime_draft_and_live_status.py
"""
import json
import sys
import wave
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


def make_wav(p: Path, seconds=0.1, rate=16000):
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * int(rate * seconds))


import tempfile

tmp = Path(tempfile.mkdtemp(prefix="mrdraft_"))
app.RECORDINGS_DIR = tmp
app.DATA_DIR = tmp
app.TRANSCRIPTS_DIR = tmp / "transcripts"
app.TRANSCRIPTS_DIR.mkdir(exist_ok=True)

wav = tmp / "recording_20261019_135904.wav"
make_wav(wav)

print("=== 1. 录音器空闲时 wav_path 为 None ===")
rec = app.AudioRecorder()
check("空闲 wav_path=None", rec.wav_path is None, rec.wav_path)

print("=== 2. 草稿写入：draft=true，文本可读 ===")
app._save_realtime_draft(str(wav), "我 们".replace(" ", ""))
mp = app._meta_path_for_wav(wav)
check("草稿 meta 已生成", mp.exists())
m = json.loads(mp.read_text(encoding="utf-8"))
check("draft 标记为真", m.get("draft") is True, m.get("draft"))
check("has_transcript 随非空文本为真", m.get("has_transcript") is True)
check("文本内容正确", m.get("text") == "我们", m.get("text"))
check("草稿引擎记录", m.get("engine") == "FunASR")
check("草稿 wav_name 正确", m.get("wav_name") == wav.name)

entry = app._recording_entry(wav)
check("列表项带 draft=true", entry.get("draft") is True)
check("列表项 has_transcript=true", entry.get("has_transcript") is True)

print("=== 3. 空文本草稿不算已转写 ===")
app._save_realtime_draft(str(wav), "   ")
m2 = json.loads((tmp / (wav.stem + ".meta.json")).read_text(encoding="utf-8"))
check("空白文本 has_transcript=false", m2.get("has_transcript") is False)
check("仍是 draft", m2.get("draft") is True)

print("=== 4. list_recordings 实时状态标记 ===")
app.state.live_wav_name = wav.name
app.state.live_stage = "recording"
items = app.list_recordings()
hit = [i for i in items if i["wav_name"] == wav.name]
check("列表含该录音", len(hit) == 1, len(hit))
check("录制中标 live_stage=recording", hit and hit[0].get("live_stage") == "recording",
      hit[0].get("live_stage") if hit else None)

app.state.live_stage = "diarizing"
items = app.list_recordings()
hit = [i for i in items if i["wav_name"] == wav.name]
check("分离中标 live_stage=diarizing", hit and hit[0].get("live_stage") == "diarizing")

app.state.live_wav_name = None
app.state.live_stage = ""
items = app.list_recordings()
hit = [i for i in items if i["wav_name"] == wav.name]
check("会话结束后无 live_stage", hit and "live_stage" not in hit[0])

print("=== 5. 最终 meta 覆盖草稿，draft 消失 ===")
app.state.live_wav_name = wav.name
app.state.live_stage = "diarizing"
app.save_recording_meta(str(wav), text="最终全文", engine="FunASR",
                        sentence_info=[], speaker_count=0)
app.state.live_wav_name = None
app.state.live_stage = ""
m3 = json.loads(mp.read_text(encoding="utf-8"))
check("最终 meta 无 draft 键", "draft" not in m3, m3.get("draft"))
check("最终文本覆盖草稿", m3.get("text") == "最终全文")
entry3 = app._recording_entry(wav)
check("最终列表项 draft=false", entry3.get("draft") is False)
check("最终列表项 has_transcript=true", entry3.get("has_transcript") is True)

print("=== 6. 不存在的 wav 写草稿不抛异常 ===")
try:
    app._save_realtime_draft(str(tmp / "nope.wav"), "x")
    check("缺失 wav 容错", True)
except Exception as e:
    check("缺失 wav 容错", False, repr(e))

print()
if FAIL:
    print(f"FAILED: {len(FAIL)} / {len(PASS) + len(FAIL)}")
    for n in FAIL:
        print("  -", n)
    sys.exit(1)
print(f"ALL PASS: {len(PASS)}")
