#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""会后处理层：转写编辑持久化、结构化纪要、全文检索、声纹注册与匹配。

本模块只放"会后"能力，不碰录音 / 实时转写 / ASR 引擎主干：
  * 录音 meta（recordings/*.wav 同名 .meta.json）的读写与编辑持久化
  * 说话人名册（speaker_labels 映射）与文本重渲染
  * 结构化纪要 JSON 解析（待办/决策/关键词）与纯文本导出
  * 历史录音/转写全文检索（简单子串 + 拼音可选，轻量缓存）
  * 声纹库（~/.MeetingRecorder/voiceprints.json + .npz）与余弦匹配框架

所有重活（numpy / 声纹模型）都惰性 import，纯逻辑部分不依赖第三方库，
方便 tests/ 直接 import 单测。
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

# ─── 数据目录（由 app.py 注入同一份，单测里可 monkeypatch） ───
DATA_DIR = Path.home() / "MeetingRecorder"
RECORDINGS_DIR = DATA_DIR / "recordings"
TRANSCRIPTS_DIR = DATA_DIR / "transcripts"

# 声纹库：voiceprints.json（元数据）+ voiceprints.npz（嵌入向量，惰性写）
VOICEPRINT_JSON = DATA_DIR / "voiceprints.json"
VOICEPRINT_NPZ = DATA_DIR / "voiceprints.npz"

# 自动命名阈值：最高相似度低于此值时回退「说话人N」
VOICEPRINT_MATCH_THRESHOLD = 0.60


# ═══════════════════════════════════════════════════════════════
# 录音 meta 读写
# ═══════════════════════════════════════════════════════════════
def _meta_path_for_wav(wav_path) -> Path:
    return Path(wav_path).with_suffix(".meta.json")


def safe_recording_path(name, recordings_dir=None):
    """把录音库里的文件名安全解析到 recordings 目录，杜绝路径穿越。

    与 app.py 里同名函数同逻辑（这里独立一份，避免本模块 import app
    触发 ASR/torch 重依赖加载，单测更快）。
    """
    if not name:
        return None
    base_dir = Path(recordings_dir or RECORDINGS_DIR)
    base = Path(str(name)).name  # 只取最后一段
    p = (base_dir / base).resolve()
    try:
        p.relative_to(base_dir.resolve())
    except ValueError:
        return None
    return p


def wav_path_from_id(recording_id, recordings_dir=None):
    """按录音 id（wav stem）找到 wav 文件；只接受安全文件名。"""
    base_dir = Path(recordings_dir or RECORDINGS_DIR)
    if not recording_id:
        return None
    safe = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff._-]", "_", str(recording_id))
    if not safe or safe in (".", ".."):
        return None
    p = base_dir / f"{safe}.wav"
    p = safe_recording_path(p.name, base_dir)
    if p is not None and p.exists():
        return p
    return None


def load_meta(wav_path) -> dict:
    """读取录音 meta，缺失/损坏返回 {}。"""
    mp = _meta_path_for_wav(wav_path)
    if not mp.exists():
        return {}
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_meta(wav_path, meta: dict) -> bool:
    """落盘 meta（整份覆盖写，UTF-8 不转义中文）。"""
    try:
        _meta_path_for_wav(wav_path).write_text(
            json.dumps(meta, ensure_ascii=False), encoding="utf-8"
        )
        return True
    except Exception:
        return False


def apply_speaker_labels(sentence_info, labels: dict):
    """把 speaker_labels 映射应用到 sentence_info（原地改 + 返回浅拷贝列表）。

    labels: {"0": "张总", "1": "李工"} —— key 是 spk 序号（字符串形式，
    与 JSON 存储一致）。同时兼容旧数据里可能出现的 "说话人1" 形式 key。
    """
    out = []
    for s in sentence_info or []:
        s2 = dict(s)
        spk = s2.get("spk", 0)
        name = labels.get(str(spk))
        if name is None:
            name = labels.get(f"说话人{int(spk) + 1}")
        if name:
            s2["speaker_name"] = name
        out.append(s2)
    return out


def render_transcript_text(sentence_info, labels: dict = None) -> str:
    """按 sentence_info 重新渲染转写全文（编辑后保持说话人/时间戳分段格式）。

    与前端 onTranscriptReady 的分段规则保持一致：同一说话人、间隔 ≤3s 的
    连续句子合并成一段，段首【说话人N/姓名】 HH:MM:SS，空行分隔。
    """
    labels = labels or {}
    paragraphs = []
    current = None
    for s in sentence_info or []:
        spk = s.get("spk", 0)
        text = (s.get("sentence") or s.get("text") or "").strip()
        if not text:
            continue
        start = int(s.get("start") or 0)
        end = int(s.get("end") or 0)
        if current is None or spk != current["spk"] or start - current["end"] > 3000:
            if current:
                paragraphs.append(current)
            current = {"spk": spk, "start": start, "end": end, "texts": [text]}
        else:
            current["end"] = end
            current["texts"].append(text)
    if current:
        paragraphs.append(current)

    lines = []
    for p in paragraphs:
        name = labels.get(str(p["spk"])) or labels.get(f"说话人{int(p['spk']) + 1}") \
            or f"说话人{p['spk'] + 1}"
        lines.append(f"【{name}】 {_format_ms(p['start'])}")
        lines.append("".join(p["texts"]))
        lines.append("")
    return "\n".join(lines).strip()


def _format_ms(ms: int) -> str:
    s = int(ms or 0) // 1000
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def update_recording_edit(recording_id, text=None, speaker_labels=None,
                          recordings_dir=None):
    """持久化转写编辑结果 / 说话人改名。

    - text: 编辑后的全文（写回 meta.text；has_transcript 置真）
    - speaker_labels: {"0": "张总", ...} 与已有映射合并（空串/None 表示删除该名）
    返回 (ok, meta_or_error)。
    """
    wav = wav_path_from_id(recording_id, recordings_dir)
    if wav is None:
        return False, "录音不存在"
    meta = load_meta(wav)
    if not meta:
        return False, "元数据不存在（请先完成转写）"

    changed = False
    if speaker_labels is not None:
        existing = dict(meta.get("speaker_labels") or {})
        for k, v in (speaker_labels or {}).items():
            kk = str(k)
            v = (v or "").strip()
            if v:
                existing[kk] = v
            else:
                existing.pop(kk, None)
        meta["speaker_labels"] = existing
        # 同步把名字写进 sentence_info，供不查映射的旧逻辑使用
        if meta.get("sentence_info"):
            meta["sentence_info"] = apply_speaker_labels(
                meta["sentence_info"], existing)
        changed = True

    if text is not None:
        text = str(text)
        meta["text"] = text
        meta["has_transcript"] = bool(text.strip())
        meta["edited"] = True
        changed = True

    if not changed:
        return False, "没有需要保存的修改"

    meta["edited_at"] = time.time()
    if not save_meta(wav, meta):
        return False, "写入失败"
    return True, meta


# ═══════════════════════════════════════════════════════════════
# 结构化纪要
# ═══════════════════════════════════════════════════════════════
def _strip_fences(s: str) -> str:
    """剥掉 LLM 返回里常见的 ```json ... ``` 代码围栏。"""
    if not s:
        return ""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()


def parse_structured_minutes(raw: str) -> dict:
    """把 LLM 返回解析成结构化纪要；解析失败时降级为空结构 + 原文。

    返回：
      {
        "topic": str,
        "todos":   [{"content","owner","due","done":False}],
        "decisions": [str],
        "keywords":  [str],
        "summary":  str,           # 会议摘要/背景（可选）
        "raw_text": str,           # 原始返回，兜底展示
        "parse_ok": bool,
      }
    """
    empty = {
        "topic": "", "todos": [], "decisions": [], "keywords": [],
        "summary": "", "raw_text": raw or "", "parse_ok": False,
    }
    text = _strip_fences(raw or "")
    if not text:
        return empty

    data = None
    # 1) 整体就是 JSON
    try:
        data = json.loads(text)
    except Exception:
        # 2) 从文本里抠第一个 {...} 块
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                data = json.loads(m.group(0))
            except Exception:
                data = None
    if not isinstance(data, dict):
        return empty

    def _as_list(v):
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return [v]

    todos = []
    for item in _as_list(data.get("todos") or data.get("todo")
                         or data.get("action_items") or data.get("待办")):
        if isinstance(item, dict):
            content = str(item.get("content") or item.get("task")
                          or item.get("item") or item.get("事项") or "").strip()
            owner = str(item.get("owner") or item.get("assignee")
                        or item.get("负责人") or "").strip()
            due = str(item.get("due") or item.get("deadline")
                      or item.get("截止") or item.get("时限") or "").strip()
            done = bool(item.get("done") or item.get("completed")
                        or item.get("完成"))
        else:
            content = str(item).strip()
            owner = due = ""
            done = False
        if content:
            todos.append({"content": content, "owner": owner,
                          "due": due, "done": done})

    decisions = []
    for item in _as_list(data.get("decisions") or data.get("decision")
                         or data.get("关键决策") or data.get("决策")):
        if isinstance(item, dict):
            d = str(item.get("content") or item.get("decision")
                    or item.get("内容") or "").strip()
        else:
            d = str(item).strip()
        if d:
            decisions.append(d)

    keywords = []
    for item in _as_list(data.get("keywords") or data.get("topics")
                         or data.get("关键词") or data.get("主题")):
        if isinstance(item, dict):
            k = str(item.get("keyword") or item.get("name")
                    or item.get("词") or "").strip()
        else:
            k = str(item).strip()
        if k and k not in keywords:
            keywords.append(k)

    topic = str(data.get("topic") or data.get("meeting_topic")
                or data.get("会议主题") or "").strip()
    summary = str(data.get("summary") or data.get("background")
                  or data.get("摘要") or "").strip()

    return {
        "topic": topic,
        "todos": todos,
        "decisions": decisions,
        "keywords": keywords,
        "summary": summary,
        "raw_text": raw or "",
        "parse_ok": True,
    }


def structured_minutes_to_text(minutes: dict) -> str:
    """结构化纪要 → 纯文本（用于 txt 导出 / 纪要区文本兜底）。"""
    if not minutes:
        return ""
    lines = []
    if minutes.get("topic"):
        lines.append(f"会议主题：{minutes['topic']}")
        lines.append("")
    if minutes.get("summary"):
        lines.append("会议摘要：")
        lines.append(minutes["summary"])
        lines.append("")
    lines.append("一、待办事项")
    todos = minutes.get("todos") or []
    if todos:
        for i, t in enumerate(todos, 1):
            mark = "[x]" if t.get("done") else "[ ]"
            tail = []
            if t.get("owner"):
                tail.append(f"负责人：{t['owner']}")
            if t.get("due"):
                tail.append(f"截止：{t['due']}")
            lines.append(f"{mark} {i}. {t['content']}"
                         + (f"（{ '，'.join(tail) }）" if tail else ""))
    else:
        lines.append("无")
    lines.append("")
    lines.append("二、关键决策")
    decisions = minutes.get("decisions") or []
    if decisions:
        for i, d in enumerate(decisions, 1):
            lines.append(f"{i}. {d}")
    else:
        lines.append("无")
    lines.append("")
    keywords = minutes.get("keywords") or []
    if keywords:
        lines.append("三、关键词/主题")
        lines.append("、".join(keywords))
    return "\n".join(lines).strip()


STRUCTURED_MINUTES_PROMPT = """你是一位资深会议秘书。请根据以下会议转写内容，输出**严格 JSON**（不要输出 JSON 以外的任何文字、不要 Markdown 代码围栏）。

JSON 字段：
{
  "topic": "会议主题（一句话）",
  "summary": "会议背景与核心讨论摘要，2-4句",
  "todos": [
    {"content": "待办事项内容", "owner": "责任人（转写中未明确则留空字符串，不得编造）", "due": "截止时间/时限（未提及留空）", "done": false}
  ],
  "decisions": ["关键决策1", "关键决策2"],
  "keywords": ["关键词1", "关键词2"]
}

要求：
- 忠实于转写，不得编造；责任人、截止时间不确定一律留空字符串
- 待办要可执行、具体；讨论中的观点不要写成待办
- 关键词 3-8 个，覆盖主要议题
- 全程中文；JSON 必须合法（双引号、无尾逗号）

{speaker_hint}
转写内容：
{content}"""


# ═══════════════════════════════════════════════════════════════
# 全文检索
# ═══════════════════════════════════════════════════════════════
def _normalize(s: str) -> str:
    """检索归一化：小写 + 去掉所有空白（中文文本里换行/空格不应影响命中）。"""
    if not s:
        return ""
    return re.sub(r"\s+", "", str(s).lower())


def _pinyin_norm(s: str) -> str:
    """拼音归一化（可选）：pypinyin 缺失时返回空串，检索自动退化为子串。"""
    try:
        from pypinyin import lazy_pinyin
        return "".join(lazy_pinyin(str(s or ""))).lower()
    except Exception:
        return ""


def _snippet(text: str, idx: int, qlen: int, radius: int = 40) -> str:
    start = max(0, idx - radius)
    end = min(len(text), idx + qlen + radius)
    pre = "…" if start > 0 else ""
    post = "…" if end < len(text) else ""
    return pre + text[start:end].replace("\n", " ").strip() + post


def search_transcripts(query: str, recordings_dir=None, limit_per_recording: int = 3):
    """跨所有历史录音的转写文本检索。

    返回 [{id, wav_name, created, speaker_count, matches: [{snippet, ...}]}]，
    按命中数/时间排序。简单全量扫描（数据量小），调用方做缓存。
    匹配：归一化子串；query 含中文时额外试拼音（pypinyin 可用时）。
    """
    base_dir = Path(recordings_dir or RECORDINGS_DIR)
    q = (query or "").strip()
    results = []
    if not q or not base_dir.exists():
        return results

    q_norm = _normalize(q)
    q_pin = _pinyin_norm(q) if re.search(r"[\u4e00-\u9fff]", q) else ""

    for wav in sorted(base_dir.glob("*.wav")):
        meta = load_meta(wav)
        if not meta:
            continue
        text = meta.get("text") or ""
        if not text.strip():
            continue
        text_norm = _normalize(text)
        hits = []

        idx = text_norm.find(q_norm) if q_norm else -1
        while idx >= 0 and len(hits) < limit_per_recording:
            # 归一化后索引与原文有偏移（空白被删），snippet 直接在原文里
            # 用附近关键词定位：取原文中该位置前后窗口做宽松匹配
            hits.append({"snippet": _snippet(text, _locate_in_raw(text, idx, q), len(q))})
            idx = text_norm.find(q_norm, idx + max(1, len(q_norm)))

        if not hits and q_pin:
            text_pin = _pinyin_norm(text)
            if q_pin and q_pin in text_pin:
                hits.append({"snippet": _snippet(text, 0, 0), "pinyin": True})

        if hits:
            results.append({
                "id": wav.stem,
                "wav_name": wav.name,
                "created": meta.get("created", 0),
                "duration": meta.get("duration", 0),
                "speaker_count": meta.get("speaker_count", 0),
                "engine": meta.get("engine", ""),
                "match_count": len(hits),
                "matches": hits,
            })

    results.sort(key=lambda r: (-r["match_count"], -(r.get("created") or 0)))
    return results


def _locate_in_raw(raw: str, norm_idx: int, q: str) -> int:
    """把归一化字符串里的偏移量近似映射回原文偏移（跳过空白/大小写差异）。

    归一化删了空白，索引会偏小。这里从原文开头同步扫描计数非空白字符，
    找到第 norm_idx 个非空白字符的原文位置。
    """
    count = 0
    for i, ch in enumerate(raw):
        if not ch.isspace():
            if count == norm_idx:
                return i
            count += 1
    return max(0, norm_idx)


# 简易 mtime 缓存：meta 文件没变就不重复扫
_SEARCH_CACHE = {"mtime": None, "items": None}


def search_transcripts_cached(query: str, recordings_dir=None):
    """带缓存的检索：缓存的是"录音列表+文本"，查询仍实时（快）。"""
    base_dir = Path(recordings_dir or RECORDINGS_DIR)
    key = tuple(sorted(
        (p.name, p.stat().st_mtime_ns) for p in base_dir.glob("*.meta.json")
    )) if base_dir.exists() else ()
    if _SEARCH_CACHE["mtime"] != key:
        items = []
        for wav in sorted(base_dir.glob("*.wav")) if base_dir.exists() else []:
            meta = load_meta(wav)
            if meta and (meta.get("text") or "").strip():
                items.append({
                    "id": wav.stem, "wav_name": wav.name,
                    "created": meta.get("created", 0),
                    "duration": meta.get("duration", 0),
                    "speaker_count": meta.get("speaker_count", 0),
                    "engine": meta.get("engine", ""),
                    "text": meta.get("text") or "",
                })
        _SEARCH_CACHE["mtime"] = key
        _SEARCH_CACHE["items"] = items

    q = (query or "").strip()
    if not q:
        return []
    q_norm = _normalize(q)
    q_pin = _pinyin_norm(q) if re.search(r"[\u4e00-\u9fff]", q) else ""
    results = []
    for it in _SEARCH_CACHE["items"]:
        text = it["text"]
        text_norm = _normalize(text)
        hits = []
        idx = text_norm.find(q_norm) if q_norm else -1
        while idx >= 0 and len(hits) < 3:
            hits.append({"snippet": _snippet(text, _locate_in_raw(text, idx, q), len(q))})
            idx = text_norm.find(q_norm, idx + max(1, len(q_norm)))
        if not hits and q_pin:
            if q_pin in _pinyin_norm(text):
                hits.append({"snippet": _snippet(text, 0, 0), "pinyin": True})
        if hits:
            results.append({
                "id": it["id"], "wav_name": it["wav_name"],
                "created": it["created"], "duration": it["duration"],
                "speaker_count": it["speaker_count"], "engine": it["engine"],
                "match_count": len(hits), "matches": hits,
            })
    results.sort(key=lambda r: (-r["match_count"], -(r.get("created") or 0)))
    return results


# ═══════════════════════════════════════════════════════════════
# 声纹注册与匹配
# ═══════════════════════════════════════════════════════════════
def _load_voiceprint_store():
    """读声纹库元数据。结构：
    {"voices": [{"name": "张总", "emb_key": "v0", "sample": "recording_xxx",
                 "created": ts, "updated": ts}]}
    嵌入向量存在同名 .npz（key = emb_key），避免 JSON 塞几百个浮点数。
    """
    if VOICEPRINT_JSON.exists():
        try:
            data = json.loads(VOICEPRINT_JSON.read_text(encoding="utf-8"))
            if isinstance(data, dict) and isinstance(data.get("voices"), list):
                return data
        except Exception:
            pass
    return {"voices": []}


def _save_voiceprint_store(store):
    VOICEPRINT_JSON.parent.mkdir(parents=True, exist_ok=True)
    VOICEPRINT_JSON.write_text(
        json.dumps(store, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_embeddings():
    """读 npz 里的全部嵌入 {emb_key: np.ndarray(L2归一化)}。无文件返回 {}。"""
    import numpy as np
    if not VOICEPRINT_NPZ.exists():
        return {}
    try:
        with np.load(VOICEPRINT_NPZ, allow_pickle=False) as z:
            return {k: np.asarray(z[k], dtype="float32") for k in z.files}
    except Exception:
        return {}


def _save_embeddings(embs: dict):
    import numpy as np
    VOICEPRINT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    if embs:
        np.savez(VOICEPRINT_NPZ, **{k: np.asarray(v, dtype="float32")
                                    for k, v in embs.items()})
    elif VOICEPRINT_NPZ.exists():
        VOICEPRINT_NPZ.unlink()


def list_voiceprints() -> list:
    store = _load_voiceprint_store()
    out = []
    for v in store["voices"]:
        out.append({k: v.get(k) for k in ("name", "sample", "created", "updated")})
    return out


def _cosine(a, b) -> float:
    import numpy as np
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def register_voiceprint(name: str, embedding, sample_recording: str = "",
                        replace: bool = False):
    """注册/更新一个声纹。embedding 为向量（list 或 ndarray）。

    同名默认更新（重新采样注册）；replace=False 且同名存在时也覆盖——
    声纹库以人名为唯一键，重复注册等于更新样本。
    返回 (ok, msg)。
    """
    name = (name or "").strip()
    if not name:
        return False, "名字不能为空"
    if embedding is None:
        return False, "空声纹向量"
    import numpy as np
    vec = np.asarray(embedding, dtype="float32").reshape(-1)
    if vec.size < 8:
        return False, "声纹向量维度异常"
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-8:
        return False, "声纹向量为零"
    vec = vec / nrm

    store = _load_voiceprint_store()
    embs = _load_embeddings()
    now = time.time()

    existing = next((v for v in store["voices"] if v["name"] == name), None)
    if existing:
        embs[existing["emb_key"]] = vec
        existing["sample"] = sample_recording or existing.get("sample", "")
        existing["updated"] = now
    else:
        emb_key = f"v{len(store['voices'])}"
        while emb_key in embs or any(v["emb_key"] == emb_key
                                     for v in store["voices"]):
            emb_key = f"v{int(emb_key[1:]) + 1}"
        store["voices"].append({
            "name": name, "emb_key": emb_key,
            "sample": sample_recording, "created": now, "updated": now,
        })
        embs[emb_key] = vec

    _save_voiceprint_store(store)
    _save_embeddings(embs)
    return True, f"已注册声纹：{name}"


def delete_voiceprint(name: str):
    store = _load_voiceprint_store()
    target = next((v for v in store["voices"] if v["name"] == name), None)
    if not target:
        return False, "没有该声纹"
    store["voices"] = [v for v in store["voices"] if v["name"] != name]
    embs = _load_embeddings()
    embs.pop(target["emb_key"], None)
    _save_voiceprint_store(store)
    _save_embeddings(embs)
    return True, f"已删除：{name}"


def match_speakers(cluster_embeddings: dict, threshold: float = None):
    """把本次会议的说话人簇嵌入与声纹库做余弦匹配。

    cluster_embeddings: {spk_index(int): vector}
    返回 {spk_index: {"name": str|None, "score": float, "matched": bool}}
    低于阈值的簇 name=None（前端回退「说话人N」）。
    """
    if threshold is None:
        threshold = VOICEPRINT_MATCH_THRESHOLD
    store = _load_voiceprint_store()
    embs = _load_embeddings()
    registry = []
    for v in store["voices"]:
        vec = embs.get(v["emb_key"])
        if vec is not None:
            registry.append((v["name"], vec))

    result = {}
    for spk, vec in (cluster_embeddings or {}).items():
        best_name, best_score = None, -1.0
        for name, ref in registry:
            score = _cosine(vec, ref)
            if score > best_score:
                best_name, best_score = name, score
        matched = best_score >= threshold and best_name is not None
        result[int(spk)] = {
            "name": best_name if matched else None,
            "score": round(max(0.0, best_score), 4),
            "matched": matched,
        }
    return result


def average_embeddings(vectors):
    """把同簇多段嵌入平均成一个原型向量（L2 归一化）。"""
    import numpy as np
    vecs = [np.asarray(v, dtype="float32").reshape(-1) for v in vectors
            if v is not None]
    if not vecs:
        return None
    proto = np.mean(np.stack(vecs), axis=0)
    nrm = float(np.linalg.norm(proto))
    if nrm < 1e-8:
        return None
    return (proto / nrm).tolist()


# ─── 声纹提取（惰性重依赖；失败要给出能看懂的错误） ───
def extract_embedding_from_audio(wav_path, start_s: float = None,
                                 end_s: float = None, status_cb=None):
    """从音频片段提取说话人嵌入。

    优先复用 FunASR cam++（CAMPPlus，仓库已在依赖里、本地已有缓存），
    失败再试 pyannote wespeaker。返回 L2 归一化向量 list[float] 或 None。

    start_s/end_s 指定片段（秒）；None 表示整段（注册样本建议 3-15 秒
    单人语音）。
    """
    def log(msg):
        if status_cb:
            try:
                status_cb(msg)
            except Exception:
                pass

    # 1) FunASR cam++
    try:
        log("正在加载 cam++ 声纹模型...")
        vec = _extract_with_funasr_campp(wav_path, start_s, end_s)
        if vec is not None:
            return vec
        log("cam++ 未得到有效向量，尝试 pyannote...")
    except Exception as e:
        log(f"cam++ 声纹提取失败: {e}")

    # 2) pyannote wespeaker
    try:
        return _extract_with_pyannote(wav_path, start_s, end_s, status_cb=log)
    except Exception as e:
        log(f"pyannote 声纹提取失败: {e}")
        return None


def _load_audio_segment(wav_path, start_s, end_s, target_sr=16000):
    """读音频片段为单声道 16k float32 numpy。soundfile 优先。"""
    import numpy as np
    try:
        import soundfile as sf
        info = sf.info(str(wav_path))
        sr = info.samplerate
        start = int((start_s or 0) * sr)
        frames = int((end_s - (start_s or 0)) * sr) if end_s else -1
        wav, _ = sf.read(str(wav_path), dtype="float32",
                         start=start, frames=frames if frames > 0 else None,
                         always_2d=True)
        wav = wav.mean(axis=1)  # 混单声道
        if sr != target_sr:
            wav = _resample(wav, sr, target_sr)
        return wav.astype("float32"), target_sr
    except Exception:
        pass
    # 兜底：librosa
    import librosa
    wav, sr = librosa.load(str(wav_path), sr=target_sr, mono=True,
                           offset=start_s or 0,
                           duration=(end_s - start_s) if (start_s and end_s) else None)
    return wav.astype("float32"), target_sr


def _resample(wav, sr, target_sr):
    try:
        import librosa
        return librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    except Exception:
        # 简单线性抽点（仅兜底，质量差但不会挂）
        import numpy as np
        if target_sr == sr:
            return wav
        idx = np.arange(0, len(wav), sr / target_sr).astype(int)
        return wav[idx[idx < len(wav)]]


def _extract_with_funasr_campp(wav_path, start_s, end_s):
    """用 FunASR 的 CAMPPlus（cam++）提取声纹。

    FunASR 分离 pipeline 里 spk_model='cam++' 用的就是这个模型。
    直接用 funasr.AutoModel(model='cam++') 加载，generate(input=wav)
    返回 [{'spk_embedding': np.array}]。
    """
    import numpy as np
    from funasr import AutoModel
    model = AutoModel(model="cam++", disable_update=True)
    if start_s is not None or end_s is not None:
        # 切片段：写临时 wav 最稳（FunASR 对 ndarray 输入支持因版本而异）
        import tempfile, soundfile as sf
        wav, sr = _load_audio_segment(wav_path, start_s, end_s)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        try:
            sf.write(tmp.name, wav, sr)
            res = model.generate(input=tmp.name, batch_size_s=300)
        finally:
            try:
                Path(tmp.name).unlink()
            except Exception:
                pass
    else:
        res = model.generate(input=str(wav_path), batch_size_s=300)
    if not res:
        return None
    item = res[0] if isinstance(res, list) else res
    emb = item.get("spk_embedding") if isinstance(item, dict) else None
    if emb is None:
        return None
    vec = np.asarray(emb, dtype="float32").reshape(-1)
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-8:
        return None
    return (vec / nrm).tolist()


def _extract_with_pyannote(wav_path, start_s, end_s, status_cb=None):
    """用 pyannote wespeaker 提取（与 pyannote_chunk_worker 同一模型）。"""
    import numpy as np
    import torch
    import soundfile as sf
    from pyannote.audio import Model
    from pyannote.audio.core.inference import Inference

    info = sf.info(str(wav_path))
    sr = info.samplerate
    s0 = int((start_s or 0) * sr)
    n = int((end_s - (start_s or 0)) * sr) if end_s else info.frames - s0
    wav_np, _ = sf.read(str(wav_path), dtype="float32", start=s0, frames=n,
                        always_2d=True)
    waveform = torch.from_numpy(wav_np.T).contiguous()
    audio_in = {"waveform": waveform, "sample_rate": int(sr)}

    emb_model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    inference = Inference(emb_model, window="sliding", step=1.0,
                          device=torch.device("cpu"))
    feat = inference(audio_in)
    data = np.asarray(feat.data, dtype="float32")
    if data.ndim != 2 or data.shape[0] == 0:
        return None
    vec = data.mean(axis=0)
    nrm = float(np.linalg.norm(vec))
    if nrm < 1e-8:
        return None
    return (vec / nrm).tolist()
