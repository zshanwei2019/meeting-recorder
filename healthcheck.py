# -*- coding: utf-8 -*-
"""环境自检与模型就绪检查（会后处理之外的独立模块）。

设计原则（与 postmeeting.py 一致）：
- 模块顶层只依赖标准库；sounddevice 等重依赖一律惰性 import，
  保证在没装音频库 / 没插声卡的环境也能跑出模型/磁盘部分的检查。
- 所有路径检查函数都接受显式 base 参数，便于单测注入临时目录。

对外主入口：run_health_check(config=None) -> dict
"""

from __future__ import annotations

import os
import shutil
import sys
import time
import threading
import platform
from pathlib import Path

# ── 缓存目录 ──────────────────────────────────────────────────
def _home() -> Path:
    return Path.home()


def modelscope_base() -> Path:
    return _home() / ".cache" / "modelscope" / "models"


def hf_hub_base() -> Path:
    return _home() / ".cache" / "huggingface" / "hub"


# 模型权重候选扩展名（.pt/.bin/.safetensors/.onnx/.ckpt/.pth）
_WEIGHT_EXTS = (".pt", ".bin", ".safetensors", ".onnx", ".ckpt", ".pth")
# 认为是“真权重”的最小体积（1MB），避免把 0 字节占位文件当成下载完成
_MIN_WEIGHT_BYTES = 1 * 1024 * 1024


# ── 模型清单 ──────────────────────────────────────────────────
# kind: "core" 为默认转写链路必需；"optional" 为特定功能才需要。
# provider: modelscope 缓存在 ~/.cache/modelscope/models（'/'->'--'）；
#           huggingface 缓存在 ~/.cache/huggingface/hub（models--org--name）。
MODEL_REGISTRY = [
    {
        "id": "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        "label": "Paraformer-Large 中文转写（核心）",
        "kind": "core", "provider": "modelscope",
    },
    {
        "id": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        "label": "FSMN 语音端点检测 VAD（核心）",
        "kind": "core", "provider": "modelscope",
    },
    {
        "id": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        "label": "CT-Transformer 标点恢复（核心）",
        "kind": "core", "provider": "modelscope",
    },
    {
        "id": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
        "label": "Paraformer 实时流式转写（实时听写用）",
        "kind": "optional", "provider": "modelscope",
    },
    {
        "id": "iic/SenseVoiceSmall",
        "label": "SenseVoice 多语种转写（可选引擎）",
        "kind": "optional", "provider": "modelscope",
    },
    {
        "id": "iic/speech_campplus_sv_zh-cn_16k-common",
        "label": "CAM++ 说话人模型（FunASR 分离兜底）",
        "kind": "optional", "provider": "modelscope",
    },
    {
        "id": "pyannote/speaker-diarization-3.1",
        "label": "pyannote 说话人分离主模型（需 HF 授权）",
        "kind": "optional", "provider": "huggingface",
        # 该仓库是纯 pipeline 配置（只有 config.yaml），权重在它引用的
        # segmentation-3.0 / wespeaker 子模型里，不能用“有无权重文件”判定。
        "configOnly": True,
    },
    {
        "id": "pyannote/segmentation-3.0",
        "label": "pyannote 分段模型（需 HF 授权）",
        "kind": "optional", "provider": "huggingface",
    },
    {
        "id": "pyannote/wespeaker-voxceleb-resnet34-LM",
        "label": "pyannote 声纹嵌入模型（跨块聚类）",
        "kind": "optional", "provider": "huggingface",
    },
]


def _dir_size_mb(path: Path) -> float:
    total = 0
    try:
        for root, _dirs, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
    except OSError:
        pass
    return round(total / (1024 * 1024), 1)


def _find_weight_file(snapshot: Path):
    """在 snapshot 目录里找一个“像样的”权重文件，返回 (Path|None, 数量)。"""
    weights = []
    if not snapshot.is_dir():
        return None, 0
    for root, _dirs, files in os.walk(snapshot):
        for f in files:
            if f.lower().endswith(_WEIGHT_EXTS):
                fp = Path(root) / f
                try:
                    if fp.stat().st_size >= _MIN_WEIGHT_BYTES:
                        weights.append(fp)
                except OSError:
                    pass
    return (weights[0] if weights else None), len(weights)


def modelscope_snapshot(model_id: str, base: Path | None = None) -> Path:
    base = base or modelscope_base()
    return base / model_id.replace("/", "--") / "snapshots" / "master"


def hf_snapshot(model_id: str, base: Path | None = None) -> Path | None:
    """HuggingFace hub 缓存布局：models--org--name/snapshots/<rev>/。返回首个 rev 目录。"""
    base = base or hf_hub_base()
    repo = base / ("models--" + model_id.replace("/", "--"))
    snap_root = repo / "snapshots"
    if not snap_root.is_dir():
        return None
    revs = sorted([p for p in snap_root.iterdir() if p.is_dir()])
    return revs[-1] if revs else None


def _has_config_file(snapshot: Path) -> bool:
    """snapshot 里是否存在 pipeline/tokenizer 等配置文件（yaml/json）。"""
    if not snapshot.is_dir():
        return False
    for root, _dirs, files in os.walk(snapshot):
        for f in files:
            if f.lower().endswith((".yaml", ".yml", ".json")) and f != ".gitkeep":
                try:
                    if (Path(root) / f).stat().st_size > 0:
                        return True
                except OSError:
                    pass
    return False


def check_model(entry: dict, ms_base: Path | None = None,
                hf_base: Path | None = None) -> dict:
    """检查单个模型的本地就绪状态。

    status:
      ok       —— snapshot 存在且含真权重（configOnly 仓库则为含有效配置文件）
      partial  —— 目录存在但没有合格权重/配置（典型：下载中断 / 只剩元数据）
      missing  —— 完全没有
    """
    mid = entry["id"]
    if entry["provider"] == "huggingface":
        snap = hf_snapshot(mid, hf_base)
    else:
        snap = modelscope_snapshot(mid, ms_base)
        if not snap.is_dir():
            # 兼容：极少数缓存没有 snapshots/master 层，直接看模型根目录
            root = (ms_base or modelscope_base()) / mid.replace("/", "--")
            if root.is_dir() and any(root.rglob("*")):
                snap = root

    size = _dir_size_mb(snap) if snap and snap.is_dir() else 0.0

    # 纯配置型仓库（如 pyannote pipeline 主仓库，权重在子模型里）：
    # 只要存在非空配置文件即算就绪。
    if entry.get("configOnly"):
        ready = bool(snap and snap.is_dir() and _has_config_file(snap))
        if ready:
            status, detail = "ok", f"已就绪（配置型，{size:g} MB）"
        elif snap and snap.is_dir():
            status, detail = "partial", "下载不完整（缺少配置文件），建议重新下载"
        else:
            status, detail = "missing", "未下载"
        return {
            "id": mid, "label": entry["label"], "kind": entry["kind"],
            "provider": entry["provider"], "status": status, "sizeMb": size,
            "weightCount": 0, "detail": detail, "configOnly": True,
        }

    weight, wcount = _find_weight_file(snap) if snap else (None, 0)

    if weight is not None:
        status = "ok"
        detail = f"已就绪（{size:g} MB）"
    elif snap and snap.is_dir():
        status = "partial"
        detail = "下载不完整（缺少模型权重），建议重新下载"
    else:
        status = "missing"
        detail = "未下载"

    return {
        "id": mid,
        "label": entry["label"],
        "kind": entry["kind"],
        "provider": entry["provider"],
        "status": status,
        "sizeMb": size,
        "weightCount": wcount,
        "detail": detail,
    }


def check_all_models(ms_base: Path | None = None, hf_base: Path | None = None) -> list:
    return [check_model(e, ms_base, hf_base) for e in MODEL_REGISTRY]


# ── 音频设备 ──────────────────────────────────────────────────
# 系统回放环回设备名关键词（中英文）
_LOOPBACK_KEYWORDS = ("立体声混音", "立体声混合", "stereo mix", "wave out mix",
                      "what u hear", "voice meeter", "vb-audio", "virtual cable",
                      "cable input", "回环", "loopback")


def _is_loopback(name: str) -> bool:
    n = (name or "").lower()
    return any(k in n for k in _LOOPBACK_KEYWORDS)


def check_audio_devices():
    """返回输入设备列表与系统环回设备检测结果。sounddevice 缺失时给出明确提示。"""
    result = {
        "status": "warn",
        "inputs": [],
        "hasLoopback": False,
        "loopbackNames": [],
        "detail": "",
    }
    try:
        import sounddevice as sd  # noqa
    except Exception as e:  # noqa: BLE001
        result["status"] = "error"
        result["detail"] = f"音频库 sounddevice 不可用：{e}"
        return result

    try:
        inputs = []
        for i, dev in enumerate(sd.query_devices()):
            try:
                if dev.get("max_input_channels", 0) > 0:
                    inputs.append({
                        "id": i,
                        "name": str(dev.get("name", "")),
                        "channels": int(dev.get("max_input_channels", 0)),
                        "sampleRate": int(dev.get("default_samplerate", 0)),
                    })
            except Exception:  # noqa: BLE001
                continue
        result["inputs"] = inputs
        loops = [d["name"] for d in inputs if _is_loopback(d["name"])]
        result["loopbackNames"] = loops
        result["hasLoopback"] = bool(loops)
        mics = [d for d in inputs if not _is_loopback(d["name"])]
        if loops:
            result["status"] = "ok"
            result["detail"] = f"检测到 {len(loops)} 个系统环回设备、{len(mics)} 个麦克风"
        elif mics:
            result["status"] = "warn"
            result["detail"] = ("未发现「立体声混音/VB-Cable」等系统环回设备，"
                                "无法内录会议声音；麦克风可用。可在声音设置启用，或安装 VB-Audio Virtual Cable")
        else:
            result["status"] = "error"
            result["detail"] = "没有任何可用的音频输入设备"
    except Exception as e:  # noqa: BLE001
        result["status"] = "error"
        result["detail"] = f"枚举音频设备失败：{e}"
    return result


# ── 磁盘 / 数据目录 ───────────────────────────────────────────
def check_disk_and_dirs(data_dir: Path | None = None):
    data_dir = data_dir or (_home() / "MeetingRecorder")
    out = {"dataDir": str(data_dir), "writable": False, "freeGb": None, "status": "warn"}
    try:
        usage = shutil.disk_usage(str(data_dir if data_dir.exists() else _home()))
        out["freeGb"] = round(usage.free / (1024 ** 3), 1)
        if usage.free < 5 * 1024 ** 3:
            out["status"] = "warn"
            out["detail"] = f"剩余磁盘空间仅 {out['freeGb']} GB，模型+录音建议预留 20GB 以上"
        else:
            out["status"] = "ok"
            out["detail"] = f"剩余磁盘 {out['freeGb']} GB"
    except Exception as e:  # noqa: BLE001
        out["detail"] = f"无法读取磁盘空间：{e}"

    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        probe = data_dir / ".health_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        out["writable"] = True
    except Exception as e:  # noqa: BLE001
        out["status"] = "error"
        out["detail"] = f"数据目录不可写：{e}"
    return out


# ── 汇总 ──────────────────────────────────────────────────────
def run_health_check(config: dict | None = None,
                     ms_base: Path | None = None,
                     hf_base: Path | None = None,
                     data_dir: Path | None = None) -> dict:
    """跑全部检查，返回给前端的健康报告。"""
    config = config or {}
    models = check_all_models(ms_base, hf_base)
    devices = check_audio_devices()
    disk = check_disk_and_dirs(data_dir)

    checks = []

    # 模型检查项
    core_missing = [m for m in models if m["kind"] == "core" and m["status"] != "ok"]
    for m in models:
        # 非核心且完全缺失、且当前配置用不到时降级为 skip 提示，不影响总判定
        checks.append({
            "id": "model:" + m["id"],
            "group": "models",
            "label": m["label"],
            "status": m["status"],
            "detail": m["detail"],
            "modelId": m["id"],
            "provider": m["provider"],
            "kind": m["kind"],
            "sizeMb": m["sizeMb"],
        })

    checks.append({
        "id": "devices", "group": "devices",
        "label": "音频输入设备（系统环回 / 麦克风）",
        "status": devices["status"], "detail": devices["detail"],
    })
    checks.append({
        "id": "disk", "group": "system",
        "label": "数据目录与磁盘空间",
        "status": disk["status"], "detail": disk["detail"],
    })

    # pyannote 授权提示（选了 pyannote 后端但没 token 属于配置提醒）
    backend = str(config.get("diarization_backend", "pyannote"))
    hf_token = str(config.get("hf_token", "") or "").strip()
    if config.get("speaker_diarization") and backend == "pyannote" and not hf_token:
        checks.append({
            "id": "hf_token", "group": "config",
            "label": "pyannote 说话人分离",
            "status": "warn",
            "detail": "已选 pyannote 后端但未填 HuggingFace Token，将回退到 FunASR 内置分离",
        })

    def severity(s):
        return {"error": 3, "partial": 2, "warn": 1, "missing": 0, "ok": 0, "skip": 0}.get(s, 0)

    n_error = sum(1 for c in checks if c["status"] == "error")
    n_partial = sum(1 for c in checks if c["status"] == "partial")
    n_warn = sum(1 for c in checks if c["status"] == "warn")
    # 总判定：核心模型缺失/残缺 或 有 error 项 => not ok
    ok = (not core_missing) and n_error == 0

    return {
        "ok": ok,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "summary": {
            "coreModelsReady": not core_missing,
            "coreMissing": [m["id"] for m in core_missing],
            "errors": n_error,
            "partial": n_partial,
            "warnings": n_warn,
        },
        "models": models,
        "devices": devices,
        "disk": disk,
        "checks": checks,
    }


# ── 缺失模型下载 ─────────────────────────────────────────────
# modelscope 公开模型可直接拉取；pyannote 是 HuggingFace gated 模型，必须
# 本人先在网页接受条款 + 提供有效 token（授权失败时给出具体模型页面链接）。

def _hf_accept_url(model_id: str) -> str:
    return "https://huggingface.co/" + model_id


def _is_auth_error(exc: Exception) -> bool:
    name = type(exc).__name__.lower()
    if any(k in name for k in ("gated", "unauthorized", "authentication", "permission")):
        return True
    status = getattr(exc, "response", None)
    code = getattr(getattr(status, "status_code", None), "value", lambda: None)
    try:
        sc = status.status_code if status is not None else None
    except Exception:
        sc = None
    if sc in (401, 403):
        return True
    low = str(exc).lower()
    return any(k in low for k in ("401", "403", "gated", "unauthorized", "access denied"))


class ModelDownloader:
    """后台串行下载缺失模型（modelscope 公开 + HuggingFace gated），内存跟踪进度。

    可注入下载函数便于单测：downloader_fn(model_id) 用于 modelscope、
    hf_downloader_fn(model_id, token) 用于 HuggingFace；生产传 None 则惰性 import。
    """

    def __init__(self, downloader_fn=None, hf_downloader_fn=None):
        # RLock：start() 持锁后还要调 status()，同线程可重入，避免自锁死锁
        self._lock = threading.RLock()
        self._jobs = {}          # model_id -> 状态 dict
        self._queue = []         # 待下载任务 [{id, provider, token}]
        self._worker_started = False
        self._downloader_fn = downloader_fn
        self._hf_downloader_fn = hf_downloader_fn

    # ── 真正的下载函数（惰性 import，避免健康检查强依赖模型库）──
    def _real_download_ms(self, model_id: str) -> str:
        from modelscope import snapshot_download
        return snapshot_download(model_id)

    def _real_download_hf(self, model_id: str, token: str) -> str:
        from huggingface_hub import snapshot_download as hf_snapshot_download
        return hf_snapshot_download(model_id, token=token)

    def _download_ms(self, model_id: str) -> str:
        return self._downloader_fn(model_id) if self._downloader_fn is not None \
            else self._real_download_ms(model_id)

    def _download_hf(self, model_id: str, token: str) -> str:
        if self._hf_downloader_fn is not None:
            return self._hf_downloader_fn(model_id, token)
        return self._real_download_hf(model_id, token)

    def _registry(self):
        return {m["id"]: m for m in MODEL_REGISTRY}

    def start(self, model_ids, ms_base: Path | None = None, hf_base: Path | None = None,
              hf_token: str = "") -> dict:
        """请求下载一批模型。返回 {started, skipped, jobs}。

        hf_token 用于 HuggingFace gated 模型；modelscope 模型忽略它。
        """
        registry = self._registry()
        hf_token = (hf_token or "").strip()
        started, skipped = [], []
        should_start_worker = False
        with self._lock:
            for raw in model_ids or []:
                mid = str(raw or "").strip()
                entry = registry.get(mid)
                if entry is None:
                    skipped.append({"id": mid, "reason": "未知模型，不支持下载"})
                    continue
                provider = entry["provider"]
                st = self._jobs.get(mid)
                if st and st["status"] in ("queued", "downloading"):
                    skipped.append({"id": mid, "reason": "已在下载队列中"})
                    continue
                # 已就绪的不必重下
                chk = check_model(entry, ms_base, hf_base)
                if chk["status"] == "ok":
                    skipped.append({"id": mid, "reason": "已就绪"})
                    continue
                # gated 模型必须先有 token（条款是否接受只能下载时才知道）
                if provider == "huggingface" and not hf_token:
                    skipped.append({
                        "id": mid,
                        "reason": "需先在设置里填写 HuggingFace Token，并在网页接受模型条款",
                        "acceptUrl": _hf_accept_url(mid),
                    })
                    continue
                self._jobs[mid] = {
                    "id": mid, "status": "queued",
                    "message": "排队等待下载…", "provider": provider,
                    "startedAt": time.time(),
                }
                if not any(q["id"] == mid for q in self._queue):
                    self._queue.append({"id": mid, "provider": provider, "token": hf_token})
                started.append(mid)
            if started and not self._worker_started:
                self._worker_started = True
                should_start_worker = True
        # 线程启动与 status() 都放到锁外，避免持锁做阻塞/重入操作
        if should_start_worker:
            threading.Thread(target=self._worker, daemon=True,
                             args=(ms_base, hf_base)).start()
        return {"started": started, "skipped": skipped,
                "jobs": self.status(ms_base=ms_base, hf_base=hf_base)}

    def _worker(self, ms_base=None, hf_base=None):
        while True:
            with self._lock:
                if not self._queue:
                    self._worker_started = False
                    return
                item = self._queue.pop(0)
                mid = item["id"]
                self._jobs[mid] = {**self._jobs.get(mid, {}), "id": mid,
                                   "status": "downloading",
                                   "message": "正在下载，请保持网络畅通…"}
            try:
                if item["provider"] == "huggingface":
                    path = self._download_hf(mid, item.get("token", ""))
                else:
                    path = self._download_ms(mid)
                with self._lock:
                    self._jobs[mid] = {**self._jobs.get(mid, {}), "id": mid,
                                       "status": "done", "message": f"下载完成：{path}",
                                       "finishedAt": time.time()}
            except Exception as e:  # noqa: BLE001
                msg = self._friendly_error(mid, e)
                with self._lock:
                    self._jobs[mid] = {**self._jobs.get(mid, {}), "id": mid,
                                       "status": "error", "message": msg,
                                       "finishedAt": time.time()}

    def _friendly_error(self, mid: str, exc: Exception) -> str:
        if _is_auth_error(exc):
            return ("授权失败：请确认 HuggingFace Token 有效，且已在网页点 “Agree” 接受该模型条款后重试："
                    + _hf_accept_url(mid) + f"（{str(exc)[:100]}）")
        return f"下载失败：{exc}"

    def status(self, ms_base: Path | None = None, hf_base: Path | None = None) -> list:
        """返回所有任务状态；done 的再用 check_model 复核（防止假成功）。"""
        with self._lock:
            jobs = [dict(v) for v in self._jobs.values()]
        for j in jobs:
            if j["status"] == "done":
                entry = self._registry().get(j["id"])
                if entry is not None:
                    chk = check_model(entry, ms_base, hf_base)
                    if chk["status"] != "ok":
                        j["status"] = "error"
                        j["message"] = "下载结束但未检测到完整文件，请重试"
        # 稳定排序：进行中在前，再按开始时间
        order = {"downloading": 0, "queued": 1, "error": 2, "done": 3}
        jobs.sort(key=lambda j: (order.get(j["status"], 9), j.get("startedAt", 0)))
        return jobs
