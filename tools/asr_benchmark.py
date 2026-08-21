"""ASR 模型选型评测工具（开发用，不参与 app 运行）。

设计要点：指标函数与 ASR 引擎**解耦**。
  - 指标部分（normalize_text / cer / term_recall）零第三方依赖，
    任何环境都能 import 并单测；
  - 引擎适配器**惰性导入**，缺哪个跳哪个，不影响其余引擎与单测。

用法：
    # 只跑指标自检
    python tools/asr_benchmark.py --selftest

    # 评测单个音频（需已装对应引擎）
    python tools/asr_benchmark.py --audio a.wav --ref a.txt \\
        --engines faster-whisper:small,funasr:SenseVoiceSmall \\
        --terms 西工智财,回款,ERP

    # 批量：目录下 *.wav 各自配同名 .txt 参考文本
    python tools/asr_benchmark.py --dir benchmark/samples --engines ...

参考文本（.txt）为人工校对的正确转写；无参考文本时只测速度不算 CER。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
import wave
import contextlib
from pathlib import Path

# ────────────────────────────── 指标 ──────────────────────────────
# 这一段刻意不依赖任何第三方库，保证在任何 venv 下都能 import 与单测。

_PUNCT_RE = re.compile(
    r"[\s，。、；：？！“”‘’（）【】《》—…·,.;:?!\"'()\[\]<>/\\|~`@#$%^&*+=_\-]+"
)


def normalize_text(text: str) -> str:
    """归一化：全角→半角、去标点空白、英文小写。

    CER 必须在归一化后计算，否则标点差异会淹没真实识别错误
    （ASR 加标点策略各不相同，不该算进字错率）。
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _PUNCT_RE.sub("", text)
    return text.lower()


def edit_distance(a: str, b: str) -> int:
    """Levenshtein 距离，滚动数组实现（O(min(len)) 空间）。"""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,          # 删除
                cur[j - 1] + 1,       # 插入
                prev[j - 1] + (ca != cb),  # 替换
            ))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    """字错率 = 编辑距离 / 参考文本字数。

    返回值可 >1（插入过多时），不做截断——截断会掩盖模型胡说八道的程度。
    参考文本为空时返回 0.0（无从计算，交由调用方判断是否跳过）。
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return 0.0
    return edit_distance(ref, hyp) / len(ref)


def term_recall(reference: str, hypothesis: str, terms: list[str]) -> dict:
    """术语召回：参考文本中出现的术语，有多少在识别结果里也出现。

    这是热词功能的直接检验指标，且**标注成本远低于全量 CER**——
    只需标「这段里该出现哪些术语」，不必逐字校对。
    只统计确实出现在参考文本中的术语，避免用无关术语稀释分母。
    """
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    detail = {}
    hit = total = 0
    for t in terms:
        nt = normalize_text(t)
        if not nt:
            continue
        expected = ref.count(nt)
        if expected == 0:
            continue                      # 参考里没有，不计入
        got = hyp.count(nt)
        detail[t] = {"expected": expected, "got": got}
        total += expected
        hit += min(got, expected)
    return {
        "hit": hit,
        "total": total,
        "recall": (hit / total) if total else None,
        "detail": detail,
    }


def audio_duration_s(path: str | Path) -> float:
    """读 WAV 时长；非 WAV 交由引擎处理，返回 0 表示未知。"""
    try:
        with contextlib.closing(wave.open(str(path), "rb")) as w:
            return w.getnframes() / w.getframerate()
    except Exception:
        return 0.0


# ──────────────────────────── 引擎适配 ────────────────────────────
# 全部惰性导入：缺依赖只跳过该引擎，不影响其他引擎和指标自检。

# 模型实例缓存：**RTF 必须只计推理耗时，不含模型加载**。
# 否则短音频上加载时间会完全淹没推理时间——实测 5.5s 音频，
# funasr 连加载一起算得 RTF 3.499，而其内部实测推理 RTF 仅 0.177，差约 20 倍。
_MODEL_CACHE: dict = {}
_LOAD_TIMES: dict = {}


def _cache_key(engine_name: str, kwargs: dict) -> tuple:
    """同一引擎+同一模型参数复用实例，不同参数各自缓存。"""
    return (engine_name,) + tuple(sorted(kwargs.items()))


def _cached(key: tuple, factory):
    """取缓存模型；首次构造并记录加载耗时。"""
    if key not in _MODEL_CACHE:
        t0 = time.time()
        _MODEL_CACHE[key] = factory()
        _LOAD_TIMES[key] = time.time() - t0
    return _MODEL_CACHE[key]

def run_faster_whisper(audio: str, model_size: str = "small",
                       compute_type: str = "int8", beam_size: int = 5) -> str:
    from faster_whisper import WhisperModel
    model = _cached(
        ("faster-whisper", model_size, compute_type),
        lambda: WhisperModel(model_size, device="cpu", compute_type=compute_type),
    )
    segments, _info = model.transcribe(
        audio, language="zh", beam_size=beam_size, vad_filter=True
    )
    return "".join(s.text for s in segments)


def run_funasr(audio: str, model_name: str = "iic/SenseVoiceSmall") -> str:
    from funasr import AutoModel
    model = _cached(
        ("funasr", model_name),
        lambda: AutoModel(model=model_name, disable_update=True),
    )
    res = model.generate(input=audio, batch_size_s=300, use_itn=True)
    if not res:
        return ""
    text = res[0].get("text", "")
    return re.sub(r"<\s*\|[^>]*?\|\s*>", "", text).strip()


ENGINES = {
    "faster-whisper": run_faster_whisper,
    "funasr": run_funasr,
}


def parse_engine_spec(spec: str):
    """'faster-whisper:small' → (name, kwargs)"""
    if ":" in spec:
        name, arg = spec.split(":", 1)
    else:
        name, arg = spec, None
    name = name.strip()
    if name not in ENGINES:
        raise SystemExit(f"未知引擎 {name!r}，可选: {', '.join(ENGINES)}")
    if not arg:
        return name, {}
    return name, ({"model_size": arg} if name == "faster-whisper"
                  else {"model_name": arg})


def evaluate(audio: str, ref_text: str | None, engine_specs: list[str],
             terms: list[str], warmup: bool = True) -> list[dict]:
    dur = audio_duration_s(audio)
    rows = []
    for spec in engine_specs:
        name, kwargs = parse_engine_spec(spec)
        row = {"engine": spec, "audio": Path(audio).name, "duration_s": round(dur, 2)}
        try:
            # 预热一次：把模型载入并缓存，使正式计时只含推理耗时，
            # 同时避开首次调用的 lazy init / JIT 开销污染数字。
            if warmup:
                ENGINES[name](audio, **kwargs)
            t0 = time.time()
            hyp = ENGINES[name](audio, **kwargs)
            elapsed = time.time() - t0
        except ImportError as e:
            row["skipped"] = f"依赖缺失: {e.name}"
            rows.append(row)
            continue
        except Exception as e:                      # 引擎自身报错不该中断整轮评测
            row["error"] = f"{type(e).__name__}: {e}"
            rows.append(row)
            continue
        row["elapsed_s"] = round(elapsed, 2)
        row["rtf"] = round(elapsed / dur, 3) if dur else None
        row["load_s"] = round(
            next((v for k, v in _LOAD_TIMES.items() if k[0] == name), 0.0), 2)
        row["chars"] = len(normalize_text(hyp))
        row["text"] = hyp
        if ref_text:
            row["cer"] = round(cer(ref_text, hyp), 4)
        if terms:
            row["terms"] = term_recall(ref_text or "", hyp, terms)
        rows.append(row)
    return rows


# ───────────────────────────── 自检 ─────────────────────────────

def selftest() -> int:
    """指标函数自检。用可控的合成输入验证算法本身，不依赖任何模型。"""
    failed = []

    def check(name, got, expect):
        ok = got == expect
        print(("  OK   " if ok else "  FAIL ") + name
              + ("" if ok else f"\n         got={got!r} expect={expect!r}"))
        if not ok:
            failed.append(name)

    print("=== normalize_text ===")
    check("去中文标点", normalize_text("你好，世界。"), "你好世界")
    check("去空白", normalize_text(" a b\tc\n"), "abc")
    check("英文转小写", normalize_text("ERP System"), "erpsystem")
    check("全角转半角", normalize_text("ＥＲＰ"), "erp")
    check("空输入", normalize_text(""), "")
    check("None 安全", normalize_text(None), "")

    print()
    print("=== edit_distance ===")
    check("相同串距离 0", edit_distance("abc", "abc"), 0)
    check("单字替换", edit_distance("abc", "abd"), 1)
    check("单字删除", edit_distance("abc", "ab"), 1)
    check("空串", edit_distance("", "abc"), 3)
    check("参数顺序无关", edit_distance("abc", "a"), edit_distance("a", "abc"))

    print()
    print("=== cer ===")
    check("完全正确 CER=0", cer("达摩院推出", "达摩院推出"), 0.0)
    # 5 字里错 2 字（达摩→打磨）
    check("错 2/5 字", round(cer("达摩院推出", "打磨院推出"), 4), 0.4)
    check("标点差异不计入", cer("你好世界", "你好，世界。"), 0.0)
    check("空参考返回 0", cer("", "任意"), 0.0)
    check("识别为空则全错", cer("abc", ""), 1.0)
    # 插入过多允许 >1，不截断
    check("插入过多可 >1", cer("ab", "abcdef") > 1.0, True)

    print()
    print("=== term_recall ===")
    r = term_recall("西工智财的回款", "西工智财的回款", ["西工智财", "回款"])
    check("全命中 recall=1", r["recall"], 1.0)
    r = term_recall("西工智财的回款", "西工只财的回款", ["西工智财", "回款"])
    check("漏 1 个 recall=0.5", r["recall"], 0.5)
    r = term_recall("今天开会", "今天开会", ["西工智财"])
    check("参考中无该术语则不计入分母", r["total"], 0)
    check("分母为 0 时 recall=None", r["recall"], None)
    r = term_recall("ERP 上线", "erp上线", ["ERP"])
    check("大小写/空格归一化后仍命中", r["recall"], 1.0)
    r = term_recall("回款回款", "回款", ["回款"])
    check("重复出现按次数统计", (r["hit"], r["total"]), (1, 2))

    print()
    if failed:
        print(f"失败 {len(failed)} 项: {', '.join(failed)}")
        return 1
    print("指标自检全部通过")
    return 0


def main():
    ap = argparse.ArgumentParser(description="ASR 模型选型评测")
    ap.add_argument("--selftest", action="store_true", help="只跑指标自检")
    ap.add_argument("--audio", help="单个音频文件")
    ap.add_argument("--ref", help="参考文本文件（人工校对的正确转写）")
    ap.add_argument("--dir", help="批量目录：*.wav 配同名 .txt")
    ap.add_argument("--engines", default="faster-whisper:small",
                    help="逗号分隔，如 faster-whisper:small,funasr:iic/SenseVoiceSmall")
    ap.add_argument("--terms", default="", help="逗号分隔的术语/热词表")
    ap.add_argument("--out", help="结果写入 JSON")
    args = ap.parse_args()

    if args.selftest:
        return selftest()

    engines = [s for s in (x.strip() for x in args.engines.split(",")) if s]
    terms = [s for s in (x.strip() for x in args.terms.split(",")) if s]

    jobs = []
    if args.dir:
        for wav in sorted(Path(args.dir).glob("*.wav")):
            txt = wav.with_suffix(".txt")
            jobs.append((str(wav), txt.read_text(encoding="utf-8") if txt.exists() else None))
    elif args.audio:
        ref = Path(args.ref).read_text(encoding="utf-8") if args.ref else None
        jobs.append((args.audio, ref))
    else:
        ap.error("需要 --audio 或 --dir（或 --selftest）")

    if not jobs:
        print("没有找到音频文件")
        return 1

    all_rows = []
    for audio, ref in jobs:
        print(f"\n### {Path(audio).name}  ({audio_duration_s(audio):.1f}s)"
              f"{'  [无参考文本，仅测速]' if not ref else ''}")
        rows = evaluate(audio, ref, engines, terms)
        for r in rows:
            if "skipped" in r:
                print(f"  {r['engine']:<34} 跳过 — {r['skipped']}")
                continue
            if "error" in r:
                print(f"  {r['engine']:<34} 出错 — {r['error']}")
                continue
            line = (f"  {r['engine']:<34} RTF {r['rtf']}  推理 {r['elapsed_s']}s"
                    f"  (加载 {r.get('load_s', 0)}s)")
            if "cer" in r:
                line += f"  CER {r['cer']:.1%}"
            t = r.get("terms")
            if t and t["recall"] is not None:
                line += f"  术语 {t['hit']}/{t['total']}"
            print(line)
        all_rows.extend(rows)

    # 汇总对比
    scored = [r for r in all_rows if "cer" in r]
    if scored:
        print("\n=== 按引擎汇总（CER 越低越好）===")
        by_engine = {}
        for r in scored:
            by_engine.setdefault(r["engine"], []).append(r)
        for eng, rs in sorted(by_engine.items(),
                              key=lambda kv: sum(x["cer"] for x in kv[1]) / len(kv[1])):
            avg_cer = sum(x["cer"] for x in rs) / len(rs)
            rtfs = [x["rtf"] for x in rs if x.get("rtf")]
            avg_rtf = sum(rtfs) / len(rtfs) if rtfs else None
            print(f"  {eng:<34} CER {avg_cer:.2%}"
                  + (f"  RTF {avg_rtf:.3f}" if avg_rtf else "")
                  + f"  ({len(rs)} 条)")

    if args.out:
        Path(args.out).write_text(
            json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已写入 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
