#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""会后处理层（postmeeting.py）纯逻辑回归测试。

不加载 ASR 模型、不打开音频设备、不发网络请求；声纹部分用随机向量
代替真实嵌入（只验证余弦匹配/存取逻辑，不验证模型本身）。

运行：.venv\\Scripts\\python.exe tests\\test_postmeeting.py
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import postmeeting as pm  # noqa: E402

PASS = []
FAIL = []


def check(name, cond, hint=""):
    if cond:
        PASS.append(name)
        print(f"  OK   {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL {name} {hint}")


def check_eq(name, got, expect):
    check(name, got == expect, f"got={got!r} expect={expect!r}")


# ── 用临时目录冒充录音库，避免碰真实数据 ──
_tmp = Path(tempfile.mkdtemp(prefix="mr_test_"))
_rec = _tmp / "recordings"
_rec.mkdir(parents=True)
pm.RECORDINGS_DIR = _rec
pm.DATA_DIR = _tmp
pm.VOICEPRINT_JSON = _tmp / "voiceprints.json"
pm.VOICEPRINT_NPZ = _tmp / "voiceprints.npz"


def _make_recording(rid, text, sentence_info=None, labels=None):
    wav = _rec / f"{rid}.wav"
    wav.write_bytes(b"RIFFfake")  # 内容不参与纯逻辑测试
    meta = {
        "id": rid, "wav_name": wav.name, "created": 1700000000,
        "duration": 12.3, "engine": "FunASR", "speaker_count": 2,
        "text": text, "has_transcript": bool(text),
    }
    if sentence_info is not None:
        meta["sentence_info"] = sentence_info
    if labels:
        meta["speaker_labels"] = labels
    pm.save_meta(wav, meta)
    return wav


print("=== 1. 文件名安全 / 路径穿越 ===")
p = pm.safe_recording_path("../../../etc/passwd", _rec)
check("穿越路径被压成纯文件名", p is not None and p.parent == _rec.resolve())
p2 = pm.safe_recording_path("recording_20260101_120000.wav", _rec)
check("正常文件名解析成功", p2 is not None and p2.name.endswith(".wav"))
check("空名返回 None", pm.safe_recording_path("", _rec) is None)
check("wav_path_from_id 防穿越",
      pm.wav_path_from_id("../evil", _rec) is None or
      pm.wav_path_from_id("../evil", _rec).parent == _rec.resolve())

print("=== 2. 转写渲染 + 说话人改名 ===")
si = [
    {"spk": 0, "start": 0, "end": 1500, "text": "你好今天开会"},
    {"spk": 0, "start": 1600, "end": 3000, "text": "讨论预算"},
    {"spk": 1, "start": 3500, "end": 5000, "text": "我同意"},
]
rendered = pm.render_transcript_text(si, {"0": "张总", "1": "李工"})
check("渲染含改名后说话人", "张总" in rendered and "李工" in rendered)
check("渲染含时间戳", "00:00" in rendered)
check("同说话人短句合并成段", rendered.count("【") == 2)
rendered_default = pm.render_transcript_text(si)
check("无映射时回退说话人N", "说话人1" in rendered_default and "说话人2" in rendered_default)

ok, meta = pm.update_recording_edit(
    "rec_a", speaker_labels={"0": "张总", "1": "李工"}, recordings_dir=_rec) \
    if False else (False, None)
wav_a = _make_recording("rec_a", "原文", sentence_info=si)
ok, meta = pm.update_recording_edit(
    "rec_a", speaker_labels={"0": "张总", "1": "  "}, recordings_dir=_rec)
check("改名保存成功", ok)
check("空名字被删除（1号无名）", meta["speaker_labels"] == {"0": "张总"})
check("sentence_info 同步写入 speaker_name",
      any(s.get("speaker_name") == "张总" for s in meta["sentence_info"]))

ok, meta = pm.update_recording_edit("rec_a", text="编辑后的全文", recordings_dir=_rec)
check("文本编辑保存成功", ok and meta["text"] == "编辑后的全文" and meta["edited"] is True)

ok, err = pm.update_recording_edit("rec_not_exist", text="x", recordings_dir=_rec)
check("不存在录音报错", not ok)

print("=== 3. 结构化纪要 JSON 解析 ===")
good = '''```json
{"topic": "预算评审",
 "summary": "讨论了Q3预算",
 "todos": [
   {"content": "整理预算表", "owner": "张总", "due": "周五", "done": false},
   {"content": "订会议室", "owner": "", "due": ""}
 ],
 "decisions": ["预算上调10%", "周五复审"],
 "keywords": ["预算", "Q3"]}
```'''
m = pm.parse_structured_minutes(good)
check("围栏剥离+JSON解析", m["parse_ok"] is True)
check_eq("topic", m["topic"], "预算评审")
check_eq("待办条数", len(m["todos"]), 2)
check_eq("待办责任人", m["todos"][0]["owner"], "张总")
check("空责任人留空串", m["todos"][1]["owner"] == "")
check_eq("决策条数", len(m["decisions"]), 2)
check_eq("关键词", m["keywords"], ["预算", "Q3"])

# 中文字段名容错
m2 = pm.parse_structured_minutes(json.dumps({
    "会议主题": "周会",
    "待办": [{"事项": "发周报", "负责人": "小李"}],
    "关键决策": ["下周发布"],
    "关键词": ["发布"],
}, ensure_ascii=False))
check("中文字段名容错", m2["parse_ok"] and m2["todos"][0]["content"] == "发周报"
      and m2["decisions"][0] == "下周发布")

# 纯文本 / 垃圾输入降级
m3 = pm.parse_structured_minutes("今天天气不错，没有JSON结构")
check("非JSON输入降级 parse_ok=False", m3["parse_ok"] is False and m3["raw_text"])

# 纯文本导出
txt = pm.structured_minutes_to_text(m)
check("导出含待办勾选框", "[ ]" in txt and "整理预算表" in txt)
check("导出含责任人", "张总" in txt)
check("导出含决策段", "关键决策" in txt and "预算上调10%" in txt)
m["todos"][0]["done"] = True
txt2 = pm.structured_minutes_to_text(m)
check("勾选状态导出 [x]", "[x]" in txt2)

print("=== 4. 全文检索 ===")
_make_recording("rec_b", "张总在会上说预算要上调百分之十，大家都同意了")
_make_recording("rec_c", "下周安排供应商评审，地点在三号会议室")
_make_recording("rec_d", "")  # 无转写

res = pm.search_transcripts("预算", recordings_dir=_rec)
check("检索命中含关键词的录音", any(r["id"] == "rec_b" for r in res))
check("无转写录音不出现", all(r["id"] != "rec_d" for r in res))
hit_b = next(r for r in res if r["id"] == "rec_b")
check("命中片段 snippet", "预算" in hit_b["matches"][0]["snippet"])

res2 = pm.search_transcripts(" 预算 ", recordings_dir=_rec)  # 空白归一化
check("query 空白归一化", any(r["id"] == "rec_b" for r in res2))

res3 = pm.search_transcripts("供应商", recordings_dir=_rec)
check("另一关键词命中", any(r["id"] == "rec_c" for r in res3))

res4 = pm.search_transcripts("完全不存在的词xyz", recordings_dir=_rec)
check("无命中返回空", res4 == [])

# 缓存路径一致
res5 = pm.search_transcripts_cached("预算", recordings_dir=_rec)
check("缓存检索结果一致", any(r["id"] == "rec_b" for r in res5))

# 大小写/空白归一化（英文）
_make_recording("rec_e", "The API endpoint is /api/recordings")
res6 = pm.search_transcripts("api", recordings_dir=_rec)
check("英文小写命中", any(r["id"] == "rec_e" for r in res6))

print("=== 5. 声纹注册/匹配（随机向量验证逻辑） ===")
import numpy as np
rng = np.random.default_rng(42)


def _unit(v):
    v = np.asarray(v, dtype="float32")
    return (v / np.linalg.norm(v)).tolist()


base_zhang = rng.standard_normal(256)
base_li = rng.standard_normal(256)
ok, msg = pm.register_voiceprint("张总", _unit(base_zhang), sample_recording="rec_a")
check("注册张总", ok)
ok, msg = pm.register_voiceprint("李工", _unit(base_li))
check("注册李工", ok)

vps = pm.list_voiceprints()
check_eq("声纹库 2 人", len(vps), 2)

# 簇嵌入：簇0≈张总（加小噪声），簇1≈李工
cluster0 = _unit(base_zhang + rng.standard_normal(256) * 0.1)
cluster1 = _unit(base_li + rng.standard_normal(256) * 0.1)
cluster_unknown = _unit(rng.standard_normal(256))  # 完全陌生人
matches = pm.match_speakers({0: cluster0, 1: cluster1, 2: cluster_unknown},
                            threshold=0.6)
check("簇0 匹配张总", matches[0]["matched"] and matches[0]["name"] == "张总",
      f"score={matches[0]['score']}")
check("簇1 匹配李工", matches[1]["matched"] and matches[1]["name"] == "李工",
      f"score={matches[1]['score']}")
check("陌生人低于阈值回退 None",
      (not matches[2]["matched"]) and matches[2]["name"] is None,
      f"score={matches[2]['score']}")

# 同名更新
ok, _ = pm.register_voiceprint("张总", _unit(base_zhang + rng.standard_normal(256) * 0.05))
vps2 = pm.list_voiceprints()
check_eq("同名注册为更新而非新增", len(vps2), 2)

# 删除
ok, _ = pm.delete_voiceprint("李工")
check_eq("删除后 1 人", len(pm.list_voiceprints()), 1)
ok, _ = pm.delete_voiceprint("不存在的人")
check("删除不存在者报错", not ok)

# 空/零向量拒绝
ok, err = pm.register_voiceprint("", _unit(base_zhang))
check("空名拒绝", not ok)
ok, err = pm.register_voiceprint("零向量", np.zeros(256))
check("零向量拒绝", not ok)

# 原型平均
proto = pm.average_embeddings([_unit(base_zhang), _unit(base_zhang * 0.9)])
check("多段嵌入平均出原型", proto is not None and len(proto) == 256)
check("平均原型 L2 归一化", abs(float(np.linalg.norm(proto)) - 1.0) < 1e-4)

print("=== 6. 时间戳格式化 ===")
check_eq("0秒", pm._format_ms(0), "00:00")
check_eq("65秒", pm._format_ms(65000), "01:05")
check_eq("1小时", pm._format_ms(3725000), "01:02:05")

print()
print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
if FAIL:
    print("失败项：", FAIL)
    sys.exit(1)
print("全部通过 PASS")
