# -*- coding: utf-8 -*-
"""云端录音文件识别（ASR）统一适配层。

支持三家：
- aliyun  阿里云百炼 Paraformer（paraformer-v2，异步任务，音频需公网 URL → OSS 预签名）
- volc    火山引擎豆包大模型（极速版 flash 一次请求直出；>2h 走标准版 submit/query）
- tencent 腾讯云录音文件识别（CreateRecTask / DescribeTaskStatus，TC3 签名）

设计约束：
- 重依赖（av / oss2）惰性 import，模块本身只依赖 requests + 标准库，
  没装库也能 import、跑纯逻辑与单测。
- HTTP / 上传函数全部可注入（post/get/put 参数），单测不联网。
- 返回结构与本地 FunASR 对齐：
    {"text": str, "sentence_info": [{"text","start","end","spk",...}], "speaker_count": int}
  时间戳单位毫秒；未开说话人分离时 sentence_info 为空（与 app.py 本地引擎行为一致）。

文档依据（2026-09 核对官方文档）：
- 阿里: help.aliyun.com/zh/model-studio/paraformer-recorded-speech-recognition-python-sdk
- 火山: docs.volcengine.com/docs/6561/1631584（极速版）/1354868（标准版）
- 腾讯: cloud.tencent.com/document/product/1093/37823（CreateRecTask）/37822（查询）
"""

import base64
import hashlib
import hmac
import json
import os
import time
import uuid
from pathlib import Path

import requests

try:
    from app_logging import get_logger
except Exception:  # pragma: no cover - 极端环境缺模块时退回空日志
    import logging
    def get_logger(name=""):
        lg = logging.getLogger(f"meeting_recorder.{name}" if name else "meeting_recorder")
        if not lg.handlers:
            lg.addHandler(logging.NullHandler())
        return lg

log = get_logger("cloud_asr")

# ─── 常量 ───────────────────────────────────────────────────────────────────

CLOUD_ENGINES = ("aliyun", "volc", "tencent")
CLOUD_ENGINE_LABELS = {
    "aliyun": "阿里云 Paraformer",
    "volc": "火山豆包",
    "tencent": "腾讯云",
}

# 云端任务整体超时（含排队）。官方说法通常数分钟，留足 1 小时余量。
ASR_TASK_TIMEOUT_S = 3600
POLL_INTERVAL_S = 5

# 腾讯 CreateRecTask 的 base64 直传上限 5MB；留安全余量按 4.7MB 原始文件判定。
TENCENT_DIRECT_MAX_BYTES = 4_700_000
# 火山极速版硬限制 100MB / 2 小时（mp3 压缩后体积基本不会触顶，主要卡时长）。
VOLC_FLASH_MAX_BYTES = 100 * 1024 * 1024
VOLC_FLASH_MAX_SECONDS = 2 * 3600

MP3_BIT_RATE = 32_000  # 16k 单声道人声 32kbps 足够，1 小时约 14MB

# HTTP 重试（仅对网络错误 / 429 / 5xx；4xx 参数/鉴权错误立即失败，不重试）
RETRY_MAX_ATTEMPTS = 3
RETRY_BASE_DELAY = 0.5
RETRY_MAX_DELAY = 8.0


def _retry_sleep(attempt):
    """指数退避；测试可 monkeypatch 本函数实现零等待。attempt 从 0 起。"""
    time.sleep(min(RETRY_MAX_DELAY, RETRY_BASE_DELAY * (2 ** attempt)))


def _is_retryable_status(code):
    return code == 429 or 500 <= code < 600


def classify_http_error(status, body=None, label=""):
    """把 HTTP 状态码/响应体翻译成人话，供用户排查（密钥/权限/限流/服务端）。"""
    prefix = f"{label}：" if label else ""
    if status in (401, 403):
        return (f"{prefix}鉴权失败（HTTP {status}），请检查 API Key / SecretId / SecretKey "
                "是否正确、对应云服务是否已开通、密钥是否有该资源权限")
    if status == 404:
        return f"{prefix}接口或资源不存在（HTTP 404），请确认服务已开通或音频 URL 可被公网下载"
    if status == 429:
        return f"{prefix}触发云端限流（HTTP 429），请稍后重试或降低并发"
    if status == 413:
        return f"{prefix}音频过大（HTTP 413）"
    if 400 <= status < 500:
        msg = ""
        if isinstance(body, dict):
            msg = body.get("message") or (body.get("Response") or {}).get("Error", {}).get("Message") or ""
        return f"{prefix}请求被拒绝（HTTP {status}）{('：' + msg) if msg else ''}"
    if 500 <= status < 600:
        return f"{prefix}云端服务暂时不可用（HTTP {status}），已重试仍失败，请稍后再试"
    return f"{prefix}HTTP {status}"


class CloudASRError(Exception):
    """云端转写业务错误（鉴权/参数/任务失败），文案可直接展示给用户。"""


class CloudASRCancelled(Exception):
    """用户主动取消。

    三家录音文件识别均无可靠的公开“撤销已提交任务”接口，取消只能做到：
    立即停止本地轮询/等待、清理临时托管对象；云端任务可能仍执行并计费。
    单独成类以便上层给出与“失败”不同的提示，且不当作错误写红日志。
    """
    def __init__(self, provider=""):
        self.provider = provider
        label = CLOUD_ENGINE_LABELS.get(provider, "云端") if provider else "云端"
        super().__init__(
            f"已取消{label}转写等待。注意：任务可能已提交到云端，"
            "厂商仍可能执行本次识别并计费，结果不再回填。")


def _check_cancel(cancel_event):
    """轮询循环里调用；事件置位则抛 CloudASRCancelled 尽快退出。"""
    if cancel_event is not None and cancel_event.is_set():
        raise CloudASRCancelled()


# ─── 音频预处理：16k 单声道 WAV → mp3 ────────────────────────────────────────

def encode_mp3_for_cloud(wav_path, mp3_path=None, bit_rate=MP3_BIT_RATE):
    """把录音 WAV 重采样为 16k 单声道 mp3，返回 (mp3路径, 时长秒)。

    三家云端都收 mp3；16k 单声道 int16 WAV 约 32KB/s（1 小时 115MB），
    压成 32kbps mp3 后 1 小时约 14MB，base64 直传/上传都快得多。
    PyAV（av 包）惰性 import。
    """
    import av

    wav_path = str(wav_path)
    if mp3_path is None:
        mp3_path = Path(wav_path).with_suffix(".cloud.mp3")
    mp3_path = str(mp3_path)

    in_container = av.open(wav_path)
    try:
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        out_container = av.open(mp3_path, "w", format="mp3")
        try:
            stream = out_container.add_stream("libmp3lame", rate=16000)
            stream.bit_rate = bit_rate
            total_samples = 0
            for frame in in_container.decode(audio=0):
                for rf in resampler.resample(frame):
                    total_samples += rf.samples
                    for packet in stream.encode(rf):
                        out_container.mux(packet)
            for packet in stream.encode(None):
                out_container.mux(packet)
        finally:
            out_container.close()
    finally:
        in_container.close()

    duration_s = total_samples / 16000.0 if total_samples else 0.0
    return mp3_path, duration_s


# ─── 杂项 ───────────────────────────────────────────────────────────────────

def split_hot_words(raw, limit=None):
    """把设置里的热词串（| ／ ｜ ／ 换行分隔）拆成去重列表。"""
    if not raw:
        return []
    parts = []
    for chunk in str(raw).replace("｜", "|").replace("\r", "\n").replace("|", "\n").split("\n"):
        w = chunk.strip()
        if w and w not in parts:
            parts.append(w)
    return parts[:limit] if limit else parts


def _http(method, url, *, inject=None, timeout=60, retry=True, label="", request_fn=None, **kwargs):
    """发起 HTTP 请求。

    - inject：注入假函数时直接返回、完全绕过重试（供既有业务单测快速跑）；
    - request_fn：替换 requests.<method> 但仍走重试/分类逻辑（供重试机制单测）；
    真实请求对网络错误 / 429 / 5xx 做有限次指数退避重试，非重试类 4xx 原样返回。
    """
    fn = (inject or {}).get(method.lower())
    if fn is not None:
        return fn(url, **kwargs)
    caller = request_fn or getattr(requests, method.lower())
    attempts = RETRY_MAX_ATTEMPTS if retry else 1
    last_exc = None
    for attempt in range(attempts):
        try:
            resp = caller(url, timeout=timeout, **kwargs)
        except requests.RequestException as e:
            last_exc = e
            if attempt < attempts - 1:
                log.warning("%s 请求异常（第 %d 次）：%s，将重试", label or url, attempt + 1, e)
                _retry_sleep(attempt)
                continue
            raise CloudASRError(
                f"{label + '：' if label else ''}网络连接失败：{e}（已重试 {attempts} 次），"
                "请检查网络/代理后重试") from e
        if retry and _is_retryable_status(resp.status_code) and attempt < attempts - 1:
            log.warning("%s 返回 %s（第 %d 次），将重试", label or url, resp.status_code, attempt + 1)
            _retry_sleep(attempt)
            continue
        if retry and _is_retryable_status(resp.status_code):
            raise CloudASRError(classify_http_error(resp.status_code, _safe_json(resp), label))
        return resp
    # 理论不可达
    raise CloudASRError(f"{label} 请求失败：{last_exc}")


def _safe_json(resp):
    try:
        return resp.json()
    except Exception:
        return None


def _as_json(resp):
    """从真假两种响应里取 JSON（fake 直接返回 dict）。"""
    if isinstance(resp, dict):
        return resp
    try:
        return resp.json()
    except Exception:
        return {}


def _headers(resp, key, default=""):
    if isinstance(resp, dict):
        return resp.get("headers", {}).get(key, default)
    return resp.headers.get(key, default)


def _normalize_sentences(raw_sentences, diarization):
    """把各家分句统一成 app 的 sentence_info：{text,start,end,spk}（毫秒）。

    未开分离时 spk 一律 0；开了则把云端返回的说话人标识压缩成连续 0..N-1。
    """
    out = []
    spk_map = {}
    for s in raw_sentences or []:
        text = (s.get("text") or s.get("FinalSentence") or "").strip()
        if not text:
            continue
        start = int(s.get("begin_time") or s.get("StartMs") or s.get("start_time") or 0)
        end = int(s.get("end_time") or s.get("EndMs") or 0)
        spk = 0
        raw_spk = s.get("speaker_id", s.get("SpeakerId"))
        if diarization and raw_spk is not None:
            key = str(raw_spk)
            if key not in spk_map:
                spk_map[key] = len(spk_map)
            spk = spk_map[key]
        out.append({"text": text, "start": start, "end": end, "spk": spk})
    speaker_count = len(spk_map) if diarization and spk_map else 0
    return out, speaker_count


# ─── 阿里云 OSS 预签名投递 ───────────────────────────────────────────────────

def oss_upload(wav_or_bytes, config, key, *, put_fn=None, mp3_path=None):
    """上传到阿里 OSS 并返回一个 24h 有效的 GET 预签名 URL。

    config 需要：oss_access_key_id / oss_access_key_secret / oss_bucket /
    oss_endpoint（如 oss-cn-beijing.aliyuncs.com）。
    """
    try:
        import oss2
    except ImportError as e:
        raise CloudASRError("缺少 oss2 依赖，无法上传 OSS，请 pip install oss2") from e

    ak = config.get("oss_access_key_id", "").strip()
    sk = config.get("oss_access_key_secret", "").strip()
    bucket_name = config.get("oss_bucket", "").strip()
    endpoint = config.get("oss_endpoint", "").strip().replace("https://", "").replace("http://", "")
    missing = [n for n, v in (("AccessKey ID", ak), ("AccessKey Secret", sk),
                              ("Bucket", bucket_name), ("Endpoint", endpoint)) if not v]
    if missing:
        raise CloudASRError("阿里云转写需要公网可下载的音频 URL，请在设置里补全 OSS 配置（缺：%s）"
                            % "、".join(missing))

    if isinstance(wav_or_bytes, (bytes, bytearray)):
        data = bytes(wav_or_bytes)
    else:
        mp3, _ = encode_mp3_for_cloud(wav_or_bytes, mp3_path)
        with open(mp3, "rb") as f:
            data = f.read()

    auth = oss2.Auth(ak, sk)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    put_url = bucket.sign_url("PUT", key, 600, slash_safe=True)
    log.info("OSS 上传开始：bucket=%s key=%s 大小=%.1fMB", bucket_name, key, len(data) / 1048576)
    r = _http("put", put_url, inject=( {"put": put_fn} if put_fn is not None else None),
              data=data, headers={"Content-Type": "audio/mpeg"},
              timeout=300, label="OSS 上传")
    status = r.status_code if hasattr(r, "status_code") else r.get("status_code", 200)
    if status >= 300:
        raise CloudASRError(classify_http_error(status, _safe_json(r), "OSS 上传")
                            + "，请检查 Bucket/Endpoint/权限")
    log.info("OSS 上传成功：%s", key)
    return bucket.sign_url("GET", key, 24 * 3600, slash_safe=True), data


def _oss_cleanup(config, key):
    """转写结束后尽力清理临时文件，失败不影响主流程，仅记日志。"""
    try:
        import oss2
        auth = oss2.Auth(config["oss_access_key_id"].strip(), config["oss_access_key_secret"].strip())
        bucket = oss2.Bucket(auth,
                             config["oss_endpoint"].strip().replace("https://", "").replace("http://", ""),
                             config["oss_bucket"].strip())
        bucket.delete_object(key)
        log.info("OSS 临时对象已清理：%s", key)
    except Exception as e:
        log.warning("OSS 临时对象清理失败（预签名 24h 后失效）：%s %s", key, e)


# ─── 阿里云百炼 Paraformer（paraformer-v2）──────────────────────────────────

def aliyun_transcribe(file_url, config, *, status_callback=None, http=None,
                      diarization=False, speaker_count=0, cancel_event=None):
    """提交 Paraformer 录音文件识别任务并轮询到结束。file_url 必须公网可下载。"""
    api_key = (config.get("aliyun_asr_api_key") or config.get("dashscope_api_key") or "").strip()
    if not api_key:
        raise CloudASRError("请先在设置里填写阿里云百炼 API Key（DashScope API Key）")

    base = "https://dashscope.aliyuncs.com/api/v1"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    parameters = {"language_hints": ["zh", "en"]}
    # paraformer-v2 文件转写不支持请求内直传热词，只能传百炼控制台热词表 ID
    vocabulary_id = (config.get("aliyun_vocabulary_id") or "").strip()
    if vocabulary_id:
        parameters["vocabulary_id"] = vocabulary_id
    if diarization:
        parameters["diarization_enabled"] = True
        if speaker_count and int(speaker_count) > 0:
            parameters["speaker_count"] = int(speaker_count)

    def cb(msg):
        if status_callback:
            status_callback("transcribing", msg)

    t0 = time.time()
    cb("正在提交阿里云 Paraformer 任务…")
    body = {
        "model": "paraformer-v2",
        "input": {"file_urls": [file_url]},
        "parameters": parameters,
    }
    submit_headers = dict(headers, **{"X-DashScope-Async": "enable"})
    resp = _http("post", f"{base}/services/audio/asr/transcription",
                 inject=http, headers=submit_headers, json=body, label="阿里云任务提交")
    data = _as_json(resp)
    if not isinstance(resp, dict) and getattr(resp, "status_code", 200) >= 400:
        raise CloudASRError(f"阿里云任务提交失败：{data.get('message') or _brief(resp)}")
    output = data.get("output") or {}
    task_id = output.get("task_id")
    if not task_id:
        raise CloudASRError(f"阿里云未返回任务 ID：{data.get('message') or data}")
    log.info("阿里云任务已提交 task_id=%s diarization=%s", task_id, diarization)

    deadline = time.time() + ASR_TASK_TIMEOUT_S
    last_log = 0.0
    while time.time() < deadline:
        _check_cancel(cancel_event)
        resp = _http("get", f"{base}/tasks/{task_id}", inject=http, headers=headers)
        data = _as_json(resp)
        output = data.get("output") or {}
        status = output.get("task_status")
        if status in ("SUCCEEDED", "FAILED"):
            break
        if time.time() - last_log > 15:
            cb(f"阿里云转写中（{status or '排队中'}）…")
            last_log = time.time()
        time.sleep(POLL_INTERVAL_S)
    else:
        raise CloudASRError("阿里云转写超时（超过 1 小时），任务仍在排队/处理，可稍后重转")

    if status == "FAILED":
        results = output.get("results") or []
        msg = (results[0].get("message") if results else None) or output.get("message") or "任务失败"
        log.error("阿里云任务失败 task_id=%s msg=%s 耗时=%.1fs", task_id, msg, time.time() - t0)
        raise CloudASRError(f"阿里云转写失败：{msg}")

    results = output.get("results") or []
    if not results or results[0].get("subtask_status") == "FAILED":
        msg = (results[0].get("message") if results else None) or "未返回识别结果"
        log.error("阿里云子任务失败 task_id=%s msg=%s", task_id, msg)
        raise CloudASRError(f"阿里云转写失败：{msg}")
    transcription_url = results[0].get("transcription_url")
    if not transcription_url:
        raise CloudASRError("阿里云未返回结果下载地址")

    cb("正在下载识别结果…")
    r = _http("get", transcription_url, inject=http)
    result_json = _as_json(r)
    transcripts = result_json.get("transcripts") or []
    text = (transcripts[0].get("content") if transcripts else "") or ""
    sentence_info, spk_count = _normalize_sentences(result_json.get("sentences"), diarization)
    log.info("阿里云转写成功 task_id=%s 字数=%d 说话人=%d 耗时=%.1fs",
             task_id, len(text), spk_count, time.time() - t0)
    return {"text": text.strip(), "sentence_info": sentence_info, "speaker_count": spk_count}


# ─── 火山引擎豆包（极速版 flash / 标准版 submit+query）───────────────────────

def _volc_headers(config, request_id, *, sequence=None,
                  resource_id="volc.bigasr.auc_turbo"):
    """新版控制台只需 X-Api-Key；旧版用 APP ID + Access Token。"""
    api_key = (config.get("volc_asr_api_key") or "").strip()
    h = {
        "Content-Type": "application/json",
        "X-Api-Resource-Id": resource_id,
        "X-Api-Request-Id": request_id,
    }
    if sequence is not None:
        h["X-Api-Sequence"] = str(sequence)
    if api_key:
        h["X-Api-Key"] = api_key
        return h, api_key
    app_key = (config.get("volc_asr_app_key") or "").strip()
    access_key = (config.get("volc_asr_access_key") or "").strip()
    if not app_key or not access_key:
        raise CloudASRError("请先在设置里填写火山语音的 API Key（新版控制台），或 APP ID + Access Token（旧版）")
    h["X-Api-App-Key"] = app_key
    h["X-Api-Access-Key"] = access_key
    return h, app_key


def _volc_request_body(audio_field, uid, *, diarization=False, hot_words=None, model_name="bigmodel"):
    request = {
        "model_name": model_name,
        "enable_itn": True,
        "enable_punc": True,
        "show_utterances": True,
    }
    if diarization:
        request["enable_speaker_info"] = True
        # 说话人分离 2.0（仅中英场景生效）
        request["ssd_version"] = "200"
    words = split_hot_words(hot_words)
    if words:
        request["context"] = json.dumps({"hotwords": [{"word": w} for w in words[:5000]]},
                                        ensure_ascii=False)
    return {
        "user": {"uid": uid or "meeting-recorder"},
        "audio": audio_field,
        "request": request,
    }


def _volc_parse_result(data, diarization):
    result = (data or {}).get("result") or {}
    text = (result.get("text") or "").strip()
    raw_sents = []
    for u in result.get("utterances") or []:
        item = {
            "text": u.get("text") or "",
            "start_time": u.get("start_time"),
            "end_time": u.get("end_time"),
        }
        additions = u.get("additions") or {}
        if "speaker_id" in additions:
            item["speaker_id"] = additions["speaker_id"]
        # 部分版本把 speaker_id 直接放在 utterance 上
        if "speaker_id" in u:
            item["speaker_id"] = u["speaker_id"]
        raw_sents.append(item)
    sentence_info, spk_count = _normalize_sentences(raw_sents, diarization)
    return {"text": text, "sentence_info": sentence_info, "speaker_count": spk_count}


def volc_transcribe_flash(mp3_bytes, config, *, status_callback=None, http=None,
                          diarization=False, hot_words="", cancel_event=None):
    """火山极速版：一次 HTTP 请求直接返回全文（≤2h、≤100MB）。"""
    t0 = time.time()
    request_id = str(uuid.uuid4())
    headers, uid = _volc_headers(config, request_id, sequence=-1)
    audio_field = {"data": base64.b64encode(mp3_bytes).decode("ascii")}
    body = _volc_request_body(audio_field, uid, diarization=diarization, hot_words=hot_words)

    if status_callback:
        status_callback("transcribing", "正在调用火山豆包极速版转写…")
    _check_cancel(cancel_event)
    log.info("火山极速版提交 request_id=%s 大小=%.1fMB diarization=%s",
             request_id, len(mp3_bytes) / 1048576, diarization)
    url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/recognize/flash"
    resp = _http("post", url, inject=http, headers=headers, json=body, timeout=600,
                 label="火山极速版")
    code = _headers(resp, "X-Api-Status-Code")
    message = _headers(resp, "X-Api-Message")
    if code == "20000003":
        log.info("火山极速版静音音频 request_id=%s", request_id)
        return {"text": "", "sentence_info": [], "speaker_count": 0}
    if code != "20000000":
        logid = _headers(resp, "X-Tt-Logid")
        log.error("火山极速版失败 code=%s msg=%s logid=%s", code, message, logid)
        raise CloudASRError(f"火山转写失败（{code or 'HTTP错误'} {message}），logid: {logid}")
    out = _volc_parse_result(_as_json(resp), diarization)
    log.info("火山极速版成功 request_id=%s 字数=%d 耗时=%.1fs logid=%s",
             request_id, len(out["text"]), time.time() - t0, _headers(resp, "X-Tt-Logid"))
    return out


def volc_transcribe_standard(file_url, config, *, status_callback=None, http=None,
                             diarization=False, hot_words="", cancel_event=None):
    """火山标准版：submit 音频链接 + query 轮询，用于 >2h 长录音。"""
    request_id = str(uuid.uuid4())
    # 标准版资源 ID 与极速版不同：豆包录音文件识别 1.0 = volc.bigasr.auc
    headers, uid = _volc_headers(config, request_id, sequence=-1,
                                 resource_id="volc.bigasr.auc")
    audio_field = {"url": file_url, "format": "mp3"}
    body = _volc_request_body(audio_field, uid, diarization=diarization, hot_words=hot_words)

    def cb(msg):
        if status_callback:
            status_callback("transcribing", msg)

    cb("正在提交火山标准版转写任务…")
    submit_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/submit"
    resp = _http("post", submit_url, inject=http, headers=headers, json=body)
    code = _headers(resp, "X-Api-Status-Code")
    if code != "20000000":
        raise CloudASRError(
            f"火山任务提交失败（{code} {_headers(resp, 'X-Api-Message')}），"
            f"logid: {_headers(resp, 'X-Tt-Logid')}")

    query_headers, _ = _volc_headers(config, request_id, resource_id="volc.bigasr.auc")
    query_url = "https://openspeech.bytedance.com/api/v3/auc/bigmodel/query"
    deadline = time.time() + ASR_TASK_TIMEOUT_S
    last_log = 0.0
    while time.time() < deadline:
        _check_cancel(cancel_event)
        resp = _http("post", query_url, inject=http, headers=query_headers, json={})
        code = _headers(resp, "X-Api-Status-Code")
        if code == "20000000":
            return _volc_parse_result(_as_json(resp), diarization)
        if code not in ("20000001", "20000002"):
            raise CloudASRError(
                f"火山转写失败（{code} {_headers(resp, 'X-Api-Message')}），"
                f"logid: {_headers(resp, 'X-Tt-Logid')}")
        if time.time() - last_log > 15:
            cb("火山标准版转写中…")
            last_log = time.time()
        time.sleep(POLL_INTERVAL_S)
    raise CloudASRError("火山标准版转写超时（超过 1 小时）")


# ─── 腾讯云录音文件识别（TC3-HMAC-SHA256 签名）──────────────────────────────

def _tc3_sign(key, msg):
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def tencent_call(action, payload_dict, secret_id, secret_key, region, *, http=None,
                 timeout=60, label=""):
    """调用 asr.tencentcloudapi.com 的 TC3 v3 接口，返回 Response 对象内容。"""
    host = "asr.tencentcloudapi.com"
    endpoint = f"https://{host}"
    service = "asr"
    timestamp = int(time.time())
    date = time.strftime("%Y-%m-%d", time.gmtime(timestamp))
    payload = json.dumps(payload_dict, ensure_ascii=False, separators=(",", ":"))

    content_type = "application/json; charset=utf-8"
    canonical_headers = (f"content-type:{content_type}\n"
                         f"host:{host}\n"
                         f"x-tc-action:{action.lower()}\n")
    signed_headers = "content-type;host;x-tc-action"
    hashed_payload = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    canonical_request = f"POST\n/\n\n{canonical_headers}\n{signed_headers}\n{hashed_payload}"

    credential_scope = f"{date}/{service}/tc3_request"
    string_to_sign = (f"TC3-HMAC-SHA256\n{timestamp}\n{credential_scope}\n"
                      f"{hashlib.sha256(canonical_request.encode('utf-8')).hexdigest()}")
    secret_date = _tc3_sign(("TC3" + secret_key).encode("utf-8"), date)
    secret_service = _tc3_sign(secret_date, service)
    secret_signing = _tc3_sign(secret_service, "tc3_request")
    signature = hmac.new(secret_signing, string_to_sign.encode("utf-8"),
                         hashlib.sha256).hexdigest()
    authorization = (f"TC3-HMAC-SHA256 Credential={secret_id}/{credential_scope}, "
                     f"SignedHeaders={signed_headers}, Signature={signature}")

    headers = {
        "Authorization": authorization,
        "Content-Type": content_type,
        "Host": host,
        "X-TC-Action": action,
        "X-TC-Version": "2019-06-14",
        "X-TC-Timestamp": str(timestamp),
        "X-TC-Region": region,
    }
    resp = _http("post", endpoint, inject=http, headers=headers, data=payload.encode("utf-8"),
                 timeout=timeout, label=label or f"腾讯云 {action}")
    data = _as_json(resp)
    response = data.get("Response", data)
    if "Error" in response:
        err = response["Error"]
        rid = response.get("RequestId", "")
        log.error("腾讯云接口错误 action=%s code=%s msg=%s requestId=%s",
                  action, err.get('Code'), err.get('Message'), rid)
        raise CloudASRError(
            f"腾讯云接口错误：{err.get('Code')} {err.get('Message')}"
            + (f"（RequestId: {rid}）" if rid else ""))
    return response.get("Data", response)


def _cos_presigned_url(method, bucket, region, secret_id, secret_key, key, expires=600):
    """生成腾讯云 COS 对象的预签名 URL（HMAC-SHA1 简单签名）。

    注意签名与 HTTP 方法绑定：上传要签 PUT、腾讯服务器下载要签 GET，
    不能拿 PUT 的签名 URL 直接当下载地址。
    """
    method = method.lower()
    host = f"{bucket}.cos.{region}.myqcloud.com"
    now = int(time.time())
    key_time = f"{now};{now + expires}"
    # 无签名 header / param：CanonicalHeaders 与 CanonicalQueryString 均为空串
    canonical_request = f"{method}\n/{key}\n\n\n"
    sign_key = hmac.new(secret_key.encode("utf-8"), key_time.encode("utf-8"),
                        hashlib.sha1).hexdigest()
    string_to_sign = (f"sha1\n{key_time}\n"
                      f"{hashlib.sha1(canonical_request.encode('utf-8')).hexdigest()}\n")
    signature = hmac.new(sign_key.encode("utf-8"), string_to_sign.encode("utf-8"),
                         hashlib.sha1).hexdigest()
    from urllib.parse import quote
    return ("https://" + host + "/" + quote(key, safe="/") +
            "?q-sign-algorithm=sha1"
            f"&q-ak={secret_id}"
            f"&q-sign-time={key_time}"
            f"&q-key-time={key_time}"
            "&q-header-list="
            "&q-url-param-list="
            f"&q-signature={signature}")


def tencent_transcribe(mp3_bytes, config, *, status_callback=None, http=None,
                       diarization=False, speaker_count=0, hot_words="", put_fn=None,
                       cancel_event=None):
    """腾讯云录音文件识别：小文件 base64 直传，大文件走 COS 预签名 URL。"""
    secret_id = (config.get("tencent_secret_id") or "").strip()
    secret_key = (config.get("tencent_secret_key") or "").strip()
    region = (config.get("tencent_asr_region") or "ap-beijing").strip()
    if not secret_id or not secret_key:
        raise CloudASRError("请先在设置里填写腾讯云 SecretId / SecretKey（访问管理 → API密钥）")

    def cb(msg):
        if status_callback:
            status_callback("transcribing", msg)

    if len(mp3_bytes) <= TENCENT_DIRECT_MAX_BYTES:
        source_type = 1
        audio_field = {"Data": base64.b64encode(mp3_bytes).decode("ascii"),
                       "DataLen": len(mp3_bytes)}
    else:
        bucket = (config.get("cos_bucket") or "").strip()
        cos_region = (config.get("cos_region") or region).strip()
        if not bucket:
            raise CloudASRError(
                "压缩后音频仍超过 4.7MB，腾讯云直传上限 5MB；"
                "请在设置里配置 COS 存储桶（名称含 APPID，如 asr-1250000000）和地域后重试")
        key = f"{(config.get('cloud_tmp_prefix') or 'asr-temp/').strip().strip('/')}/" \
              f"{time.strftime('%Y%m%d')}/{uuid.uuid4().hex}.mp3"
        put_url = _cos_presigned_url("put", bucket, cos_region, secret_id, secret_key, key)
        # 给腾讯服务器下载用：24h 有效的 GET 签名
        get_url = _cos_presigned_url("get", bucket, cos_region, secret_id, secret_key,
                                     key, expires=24 * 3600)
        cb("正在上传音频到腾讯云 COS…")
        log.info("COS 上传开始：bucket=%s key=%s 大小=%.1fMB", bucket, key,
                 len(mp3_bytes) / 1048576)
        r = _http("put", put_url,
                  inject=({"put": put_fn} if put_fn is not None else None),
                  data=mp3_bytes, headers={"Content-Type": "audio/mpeg"},
                  timeout=300, label="COS 上传")
        status = r.status_code if hasattr(r, "status_code") else r.get("status_code", 200)
        if status >= 300:
            raise CloudASRError(classify_http_error(status, _safe_json(r), "COS 上传")
                                + "，请检查存储桶名称/地域/密钥权限")
        log.info("COS 上传成功：%s", key)
        source_type = 0
        audio_field = {"Url": get_url}

    # 多人会议 + 开分离 → 会议大模型；否则中英大模型 2.0
    engine_model = "16k_zh_en_meeting" if diarization else "16k_zh_en_2.0"
    params = {
        "EngineModelType": engine_model,
        "ChannelNum": 1,
        "ResTextFormat": 3,  # 带词时间戳+标点，按标点分段
        "SourceType": source_type,
        "ConvertNumMode": 1,
    }
    params.update(audio_field)
    if diarization:
        params["SpeakerDiarization"] = 1
        # 16k 引擎不支持指定人数，仅 8k 支持，这里不传 SpeakerNumber
    words = split_hot_words(hot_words)
    if words:
        params["HotwordList"] = ",".join(f"{w[:30]}|5" for w in words[:128])

    cb("正在提交腾讯云转写任务…")
    t0 = time.time()
    data = tencent_call("CreateRecTask", params, secret_id, secret_key, region, http=http)
    task_id = data.get("TaskId")
    if not task_id:
        raise CloudASRError(f"腾讯云未返回任务 ID：{data}")
    log.info("腾讯云任务已提交 task_id=%s engine=%s source=%s diarization=%s",
             task_id, engine_model, source_type, diarization)

    deadline = time.time() + ASR_TASK_TIMEOUT_S
    last_log = 0.0
    while time.time() < deadline:
        _check_cancel(cancel_event)
        data = tencent_call("DescribeTaskStatus", {"TaskId": task_id},
                            secret_id, secret_key, region, http=http)
        status = data.get("Status")
        if status == 2:
            break
        if status == 3:
            log.error("腾讯云任务失败 task_id=%s msg=%s 耗时=%.1fs",
                      task_id, data.get('ErrorMsg'), time.time() - t0)
            raise CloudASRError(f"腾讯云转写失败：{data.get('ErrorMsg') or '未知错误'}")
        if time.time() - last_log > 15:
            cb(f"腾讯云转写中（{data.get('StatusStr') or '排队中'}）…")
            last_log = time.time()
        time.sleep(POLL_INTERVAL_S)
    else:
        raise CloudASRError("腾讯云转写超时（超过 1 小时），可稍后用同一录音重转")

    text = (data.get("Result") or "").strip()
    raw_detail = []
    detail = data.get("ResultDetail")
    if isinstance(detail, str) and detail.strip():
        try:
            raw_detail = json.loads(detail)
        except json.JSONDecodeError:
            raw_detail = []
    elif isinstance(detail, list):
        raw_detail = detail
    sentence_info, spk_count = _normalize_sentences(raw_detail, diarization)
    if not text and sentence_info:
        text = "".join(s["text"] for s in sentence_info)
    log.info("腾讯云转写成功 task_id=%s 字数=%d 说话人=%d 耗时=%.1fs",
             task_id, len(text), spk_count, time.time() - t0)
    return {"text": text, "sentence_info": sentence_info, "speaker_count": spk_count}


# ─── 统一入口 ───────────────────────────────────────────────────────────────

def transcribe(provider, wav_path, config, *, status_callback=None,
               http=None, put_fn=None, cancel_event=None):
    """云端文件转写统一入口。

    provider: aliyun / volc / tencent
    wav_path: 16k 单声道 WAV（内部压成 mp3 再投递）
    config:   state.config
    cancel_event: 可选 threading.Event，置位后协作式取消（停止轮询/等待）。
    返回 {"text", "sentence_info", "speaker_count"}；失败抛 CloudASRError，
    用户取消抛 CloudASRCancelled。
    """
    provider = (provider or "").strip().lower()
    if provider not in CLOUD_ENGINES:
        raise CloudASRError(f"未知云端引擎：{provider}")

    diarization = bool(config.get("speaker_diarization", False))
    speaker_count = int(config.get("preset_spk_num", 0) or 0)
    hot_words = config.get("hot_words", "")
    label = CLOUD_ENGINE_LABELS.get(provider, provider)
    t_all = time.time()
    log.info("云端转写开始 provider=%s 文件=%s 分离=%s 热词=%d个",
             provider, os.path.basename(str(wav_path)), diarization,
             len(split_hot_words(hot_words)))

    if status_callback:
        status_callback("loading", "正在压缩音频为 mp3…")
    key_prefix = (config.get("cloud_tmp_prefix") or "asr-temp/").strip().strip("/")
    key = f"{key_prefix}/{time.strftime('%Y%m%d')}/{uuid.uuid4().hex}.mp3"
    tmp_mp3 = str(Path(wav_path).with_suffix(f".{uuid.uuid4().hex[:8]}.cloud.mp3"))
    try:
        mp3_path, duration_s = encode_mp3_for_cloud(wav_path, tmp_mp3)
        with open(mp3_path, "rb") as f:
            mp3_bytes = f.read()

        if provider == "volc":
            if duration_s > VOLC_FLASH_MAX_SECONDS:
                # 超长录音走标准版（必须公网 URL → 复用 OSS 配置）
                if not config.get("oss_bucket"):
                    raise CloudASRError(
                        "录音超过 2 小时，火山极速版上限 2 小时；标准版需要音频公网 URL，"
                        "请在设置里配置阿里云 OSS（用于临时托管音频）后重试，"
                        "或先用本地 FunASR 引擎转写")
                if status_callback:
                    status_callback("loading", "正在上传音频到 OSS…")
                file_url, _ = oss_upload(mp3_bytes, config, key, put_fn=put_fn)
                try:
                    result = volc_transcribe_standard(
                        file_url, config, status_callback=status_callback, http=http,
                        diarization=diarization, hot_words=hot_words,
                        cancel_event=cancel_event)
                finally:
                    _oss_cleanup(config, key)
            else:
                result = volc_transcribe_flash(
                    mp3_bytes, config, status_callback=status_callback, http=http,
                    diarization=diarization, hot_words=hot_words,
                    cancel_event=cancel_event)

        elif provider == "tencent":
            result = tencent_transcribe(
                mp3_bytes, config, status_callback=status_callback, http=http,
                diarization=diarization, speaker_count=speaker_count,
                hot_words=hot_words, put_fn=put_fn, cancel_event=cancel_event)

        elif provider == "aliyun":
            if status_callback:
                status_callback("loading", "正在上传音频到 OSS…")
            file_url, _ = oss_upload(mp3_bytes, config, key, put_fn=put_fn)
            try:
                result = aliyun_transcribe(
                    file_url, config, status_callback=status_callback, http=http,
                    diarization=diarization, speaker_count=speaker_count,
                    cancel_event=cancel_event)
            finally:
                _oss_cleanup(config, key)

        # 与本地引擎行为对齐：未开说话人分离时不回吐分句（前端按纯文本渲染）
        if not diarization:
            result["sentence_info"] = []
            result["speaker_count"] = 0
        log.info("云端转写结束 provider=%s 字数=%d 说话人=%d 总耗时=%.1fs",
                 provider, len(result.get("text", "")), result.get("speaker_count", 0),
                 time.time() - t_all)
        return result

    finally:
        try:
            if os.path.exists(tmp_mp3):
                os.remove(tmp_mp3)
        except OSError:
            pass


def _brief(resp):
    try:
        return getattr(resp, "text", "")[:300]
    except Exception:
        return ""
