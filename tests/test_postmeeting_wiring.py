#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""会后处理"接线层"回归测试（app.py <-> postmeeting.py）。

不加载 ASR 模型、不碰音频设备、不发网络请求；声纹部分用随机向量模拟
（只验证聚类透出 / label 映射 / 自动匹配 / WS 核心函数的接线逻辑，
不验证模型本身）。

运行：venv\\Scripts\\python.exe tests\\test_postmeeting_wiring.py
"""
import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

import app  # noqa: E402
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


def cos(a, b):
    a, b = np.asarray(a, dtype="float64"), np.asarray(b, dtype="float64")
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def unit(v):
    v = np.asarray(v, dtype="float32")
    return (v / np.linalg.norm(v)).tolist()


# ── 临时目录冒充录音库，避免碰真实数据 ──
_tmp = Path(tempfile.mkdtemp(prefix="mr_wiring_test_"))
_rec = _tmp / "recordings"
_rec.mkdir(parents=True)
pm.RECORDINGS_DIR = _rec
pm.DATA_DIR = _tmp
pm.TRANSCRIPTS_DIR = _tmp / "transcripts"
pm.VOICEPRINT_JSON = _tmp / "voiceprints.json"
pm.VOICEPRINT_NPZ = _tmp / "voiceprints.npz"
app.RECORDINGS_DIR = _rec
app.DATA_DIR = _tmp


def make_recording(rid, text, sentence_info=None, labels=None):
    wav = _rec / f"{rid}.wav"
    wav.write_bytes(b"RIFFxxxxWAVEfmt ")
    meta = {"id": rid, "wav_name": f"{rid}.wav", "created": 1000,
            "duration": 10, "engine": "pyannote+FunASR", "speaker_count": 2,
            "text": text, "has_transcript": bool(text)}
    if sentence_info is not None:
        meta["sentence_info"] = sentence_info
    if labels:
        meta["speaker_labels"] = labels
    pm.save_meta(wav, meta)
    return wav


SENT = [
    {"spk": 0, "start": 0, "end": 2000, "text": "张总先讲预算"},
    {"spk": 1, "start": 2500, "end": 4000, "text": "李工补充技术"},
]


print("=== 1. _pm_map_cluster_embeddings_to_spk 标签映射 ===")
v_a, v_b = unit(np.random.randn(256)), unit(np.random.randn(256))
mapped = app._pm_map_cluster_embeddings_to_spk(
    {"SPEAKER_00": v_a, "SPEAKER_01": v_b, "SPEAKER_99": None},
    {"SPEAKER_00": 0, "SPEAKER_01": 1})
check("映射后 key 为 spk 序号字符串",
      set(mapped.keys()) == {"0", "1"}, str(mapped.keys()))
check("向量原样传递", cos(mapped["0"], v_a) > 0.999)
check("label_map 缺失 / 空向量的项被丢弃", "SPEAKER_99" not in mapped and "99" not in mapped)
check_eq("空输入返回空 dict", app._pm_map_cluster_embeddings_to_spk(None, None), {})


print("=== 2. 全局聚类透出簇嵌入（_cluster_speakers_global）===")
rng = np.random.default_rng(42)
base0 = rng.standard_normal(256).astype("float32")
base1 = rng.standard_normal(256).astype("float32")


def noisy(base):
    v = base + rng.standard_normal(256).astype("float32") * 0.05
    return unit(v)


# 块内标签 A/C 是同一人（声纹0），B/D 是另一人（声纹1）
tracks = [
    {"start": 0.0, "end": 5.0, "label": "A", "emb": noisy(base0)},
    {"start": 5.0, "end": 10.0, "label": "B", "emb": noisy(base1)},
    {"start": 10.0, "end": 15.0, "label": "C", "emb": noisy(base0)},
    {"start": 15.0, "end": 20.0, "label": "D", "emb": noisy(base1)},
]
try:
    merged, cluster_embs = app.PyannoteDiarizer._cluster_speakers_global(
        tracks, num_speakers=2)
    cluster_ok = True
except Exception as e:
    cluster_ok = False
    merged, cluster_embs = [], {}
    print(f"  (聚类调用异常: {e!r})")
check("聚类调用成功并返回 (tracks, embs)", cluster_ok and bool(merged) and isinstance(cluster_embs, dict))
if cluster_ok:
    name_of = {lab: spk for (_, _, spk) in []}  # placeholder
    seg_name = {}
    for s, e, spk in merged:
        for t in tracks:
            if abs(t["start"] - s) < 1e-6:
                seg_name[t["label"]] = spk
    check("同声音片段（A/C）归并为同一说话人",
          seg_name.get("A") == seg_name.get("C") and seg_name.get("A") is not None,
          str(seg_name))
    check("异声音片段（A/B）分为不同说话人",
          seg_name.get("A") != seg_name.get("B"), str(seg_name))
    check("簇嵌入 key 与输出说话人名一致",
          set(cluster_embs.keys()) == set(seg_name.values()),
          f"{set(cluster_embs.keys())} vs {set(seg_name.values())}")
    # A/C 所在簇的向量应与 base0 高余弦
    name0 = seg_name["A"]
    check("簇向量与本人声纹余弦>0.9",
          cos(cluster_embs[name0], base0) > 0.9,
          f"cos={cos(cluster_embs[name0], base0):.3f}")
    check("簇向量与他人声纹余弦低（<0.5）",
          cos(cluster_embs[name0], base1) < 0.5,
          f"cos={cos(cluster_embs[name0], base1):.3f}")


print("=== 3. 转写后自动声纹匹配（_pm_auto_match_speakers 端到端）===")
ok_r, msg = pm.register_voiceprint("张总", unit(base0), sample_recording="rec_zhang")
check("注册随机声纹成功", ok_r, str(msg))
# 模拟聚类结果：说话人0 = 张总的声音，说话人1 = 陌生人
result = {
    "cluster_embeddings": {"0": unit(base0), "1": unit(base1)},
    "sentence_info": SENT,
}
info, labels, rendered = app._pm_auto_match_speakers(None, result)
check("自动匹配命中且仅命中张总", labels == {"0": "张总"}, str(labels))
check("返回的 info 含命中分数",
      any(i[0] == 0 and i[1] == "张总" and i[3] for i in info), str(info))
check("重渲染文本含姓名", "张总" in rendered)
check("无簇嵌入时安静退化", app._pm_auto_match_speakers(None, {}) == ([], {}, ""))
check("空结果安静退化", app._pm_auto_match_speakers(None, None) == ([], {}, ""))


print("=== 4. WS 核心：改名 / 编辑保存（_pm_core_update_recording）===")
make_recording("rec_x", "原始文本", sentence_info=[dict(s) for s in SENT])
ok, payload = app._pm_core_update_recording({
    "recording_id": "rec_x",
    "speaker_labels": {"0": "张总", "1": "李工"},
})
check("改名核心返回成功", ok, str(payload))
if ok:
    check("payload 含重渲染全文", "张总" in payload["text"] and "李工" in payload["text"])
    check("payload 回传最终名字映射", payload["speaker_labels"].get("0") == "张总")
ok2, payload2 = app._pm_core_update_recording({
    "recording_id": "rec_x", "text": "人工修订后的全文"})
check("文本编辑核心返回成功", ok2 and payload2["text"] == "人工修订后的全文", str(payload2))
ok3, err3 = app._pm_core_update_recording({"recording_id": "rec_missing"})
check("不存在录音返回失败", not ok3 and isinstance(err3, str))
ok4, err4 = app._pm_core_update_recording({})
check("缺少 id 返回失败", not ok4)


print("=== 5. WS 核心：跨录音检索（_pm_core_search）===")
make_recording("rec_budget", "讨论了明年预算和招聘计划",
               sentence_info=[{"spk": 0, "start": 0, "end": 1000, "text": "讨论预算"}])
make_recording("rec_design", "界面改版方案敲定",
               sentence_info=[{"spk": 0, "start": 0, "end": 1000, "text": "界面改版"}])
pm._SEARCH_CACHE = {"mtime": None, "items": None}
res = app._pm_core_search("预算")
check("检索返回列表", isinstance(res, list))
hit_ids = {r.get("id") for r in res} if res else set()
check("命中含预算的录音", "rec_budget" in hit_ids, str(hit_ids))
check("未命中无关录音", "rec_design" not in hit_ids, str(hit_ids))
check("空查询不炸", isinstance(app._pm_core_search(""), list))


print("=== 6. WS 核心：纪要导出（_pm_core_export_minutes）===")
raw = '```json\n{"topic": "预算评审", "summary": "通过预算", "todos": ["招人"], "decisions": [], "keywords": []}\n```'
exp = app._pm_core_export_minutes(raw)
check("纪要导出返回文本", isinstance(exp, dict) and "预算评审" in exp.get("text", ""))
check("导出 topic 透出", exp.get("topic") == "预算评审", str(exp.get("topic")))
check("垃圾输入不炸（parse_ok=False）",
      app._pm_core_export_minutes("毫无结构的文字")["parse_ok"] is False)


print("=== 7. 模块缺失保护（_pm is None）===")
saved_pm = app._pm
try:
    app._pm = None
    check("_pm=None 时改名核心报错",
          app._pm_core_update_recording({"recording_id": "x"})[0] is False)
    check("_pm=None 时检索返回 None", app._pm_core_search("x") is None)
    check("_pm=None 时导出返回 None", app._pm_core_export_minutes("x") is None)
    check("_pm=None 时自动匹配安静退化",
          app._pm_auto_match_speakers(None, result) == ([], {}, ""))
finally:
    app._pm = saved_pm
check("恢复后模块仍可用", app._pm is not None)


print(f"\n通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
if FAIL:
    print("失败项:", FAIL)
    print("存在失败 FAIL")
    sys.exit(1)
print("全部通过 PASS")
