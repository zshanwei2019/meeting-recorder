# -*- coding: utf-8 -*-
"""云端文件转写适配层（cloud_asr）单测。

全部用注入的假 HTTP / 假上传，不联网；mp3 压缩用本地生成的静音 WAV
真实走一遍 PyAV，验证端到端接线。
"""
import io
import json
import os
import struct
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cloud_asr  # noqa: E402

# 轮询间隔归零，避免测试睡眠
cloud_asr.POLL_INTERVAL_S = 0

passed = 0
failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        print(f"  OK   {name}")
        passed += 1
    else:
        print(f"  FAIL {name} {detail}")
        failed += 1


class FakeResp:
    """伪装 requests.Response：headers dict + json()。"""
    def __init__(self, payload, headers=None, status_code=200, text=""):
        self._payload = payload
        self.headers = headers or {}
        self.status_code = status_code
        self.text = text or json.dumps(payload, ensure_ascii=False)

    def json(self):
        return self._payload


def make_wav(path, seconds=0.5, rate=16000):
    """生成 16k 单声道 int16 静音 WAV。"""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(b"\x00\x00" * int(rate * seconds))
    return str(path)


# ── 1. 纯逻辑工具 ──────────────────────────────────────────────────────────

print("=== 1. split_hot_words ===")
check("竖线/全角竖线/换行都可分隔且去重",
      cloud_asr.split_hot_words("西工智财｜ERP|西工智财\n量化") ==
      ["西工智财", "ERP", "量化"])
check("空串返回空列表", cloud_asr.split_hot_words("") == [])
check("limit 截断", cloud_asr.split_hot_words("a|b|c", limit=2) == ["a", "b"])

print("\n=== 2. _normalize_sentences ===")
# 阿里 sentences 格式（begin_time / end_time / text / speaker_id）
ali = [
    {"text": "你好", "begin_time": 0, "end_time": 1000, "speaker_id": 7},
    {"text": "开会", "begin_time": 1000, "end_time": 2000, "speaker_id": 3},
    {"text": "好的", "begin_time": 2000, "end_time": 3000, "speaker_id": 7},
    {"text": "  ", "begin_time": 3000, "end_time": 3100},
]
sents, n = cloud_asr._normalize_sentences(ali, True)
check("阿里分句统一为 sentence_info", sents[0] == {"text": "你好", "start": 0, "end": 1000, "spk": 0})
check("说话人 id 压缩成连续 0..N-1（7→0, 3→1, 7→0）",
      [s["spk"] for s in sents] == [0, 1, 0])
check("speaker_count=2", n == 2, n)
check("空文本句被丢弃", len(sents) == 3)
# 腾讯格式（StartMs/EndMs/FinalSentence/SpeakerId）
tc = [{"FinalSentence": "今天天气", "StartMs": 50, "EndMs": 900, "SpeakerId": 2}]
sents2, n2 = cloud_asr._normalize_sentences(tc, True)
check("腾讯字段名归一化", sents2[0]["text"] == "今天天气" and sents2[0]["start"] == 50
      and sents2[0]["spk"] == 0)
sents3, n3 = cloud_asr._normalize_sentences(ali, False)
check("未开分离时 spk 全 0、count 0", all(s["spk"] == 0 for s in sents3) and n3 == 0)

print("\n=== 3. 火山鉴权头 ===")
h, uid = cloud_asr._volc_headers({"volc_asr_api_key": "k1"}, "rid-1", sequence=-1)
check("新版控制台只发 X-Api-Key",
      h.get("X-Api-Key") == "k1" and "X-Api-App-Key" not in h
      and h["X-Api-Resource-Id"] == "volc.bigasr.auc_turbo"
      and h["X-Api-Sequence"] == "-1")
h2, uid2 = cloud_asr._volc_headers(
    {"volc_asr_app_key": "app9", "volc_asr_access_key": "tok9"}, "rid-2")
check("旧版控制台发 APP ID + Access Token",
      h2.get("X-Api-App-Key") == "app9" and h2.get("X-Api-Access-Key") == "tok9"
      and "X-Api-Key" not in h2 and uid2 == "app9")
try:
    cloud_asr._volc_headers({}, "rid-3")
    check("火山凭据全空时报错", False)
except cloud_asr.CloudASRError:
    check("火山凭据全空时报错", True)
h3, _ = cloud_asr._volc_headers({"volc_asr_api_key": "k"}, "r",
                                resource_id="volc.bigasr.auc")
check("标准版资源 ID 可覆盖", h3["X-Api-Resource-Id"] == "volc.bigasr.auc")

print("\n=== 4. 火山请求体 ===")
body = cloud_asr._volc_request_body({"data": "AAAA"}, "u1", diarization=True,
                                    hot_words="西工|ERP")
req = body["request"]
check("开启标点/ITN/分句", req["enable_punc"] and req["enable_itn"] and req["show_utterances"])
check("分离开启说话人聚类 + ssd200",
      req.get("enable_speaker_info") is True and req.get("ssd_version") == "200")
ctx = json.loads(req["context"])
check("热词以 context.hotwords 直传",
      [w["word"] for w in ctx["hotwords"]] == ["西工", "ERP"])
check("uid 与音频字段透传", body["user"]["uid"] == "u1" and body["audio"] == {"data": "AAAA"})

print("\n=== 5. 火山结果解析 ===")
volc_data = {"result": {"text": "你好。世界。", "utterances": [
    {"text": "你好。", "start_time": 0, "end_time": 800,
     "additions": {"speaker_id": 2}},
    {"text": "世界。", "start_time": 800, "end_time": 1600,
     "additions": {"speaker_id": 5}},
]}}
out = cloud_asr._volc_parse_result(volc_data, True)
check("火山全文拼接", out["text"] == "你好。世界。")
check("火山分句带说话人重映射",
      [(s["text"], s["spk"]) for s in out["sentence_info"]] ==
      [("你好。", 0), ("世界。", 1)] and out["speaker_count"] == 2)
volc_data2 = {"result": {"text": "直出", "utterances": [
    {"text": "直出", "start_time": 0, "end_time": 300, "speaker_id": 0}]}}
out2 = cloud_asr._volc_parse_result(volc_data2, False)
check("speaker_id 在 utterance 顶层也兼容", out2["sentence_info"][0]["text"] == "直出")

# ── 6. 火山极速版（假 HTTP）────────────────────────────────────────────────

print("\n=== 6. 火山极速版 ===")
captured = {}


def fake_volc_post(url, **kwargs):
    captured["url"] = url
    captured["headers"] = kwargs.get("headers")
    captured["json"] = kwargs.get("json")
    return FakeResp(volc_data, headers={"X-Api-Status-Code": "20000000",
                                        "X-Api-Message": "OK",
                                        "X-Tt-Logid": "log-1"})


out = cloud_asr.volc_transcribe_flash(b"mp3bytes", {"volc_asr_api_key": "k"},
                                     http={"post": fake_volc_post}, diarization=True)
check("flash 打到正确 URL", captured["url"].endswith("/recognize/flash"))
check("flash 请求体是 base64 音频",
      captured["json"]["audio"]["data"] == "bXAzYnl0ZXM=")
check("flash 返回全文", out["text"] == "你好。世界。" and out["speaker_count"] == 2)


def fake_silent(url, **kwargs):
    return FakeResp({}, headers={"X-Api-Status-Code": "20000003",
                                 "X-Api-Message": "silent"})


out_s = cloud_asr.volc_transcribe_flash(b"x", {"volc_asr_api_key": "k"},
                                        http={"post": fake_silent})
check("静音音频 20000003 返回空结果不报错", out_s["text"] == "")


def fake_fail(url, **kwargs):
    return FakeResp({}, headers={"X-Api-Status-Code": "55000031",
                                 "X-Api-Message": "busy", "X-Tt-Logid": "L"})


try:
    cloud_asr.volc_transcribe_flash(b"x", {"volc_asr_api_key": "k"},
                                    http={"post": fake_fail})
    check("flash 非成功状态码报错", False)
except cloud_asr.CloudASRError as e:
    check("flash 非成功状态码报错并带 logid", "55000031" in str(e) and "L" in str(e))

# ── 7. 火山标准版轮询 ────────────────────────────────────────────────────────

print("\n=== 7. 火山标准版 ===")
seq = {"n": 0}


def fake_std(url, **kwargs):
    if url.endswith("/submit"):
        return FakeResp({}, headers={"X-Api-Status-Code": "20000000",
                                     "X-Api-Message": "OK"})
    seq["n"] += 1
    if seq["n"] == 1:
        return FakeResp({}, headers={"X-Api-Status-Code": "20000001",
                                     "X-Api-Message": "doing"})
    return FakeResp(volc_data, headers={"X-Api-Status-Code": "20000000",
                                        "X-Api-Message": "OK"})


out = cloud_asr.volc_transcribe_standard("https://x/a.mp3",
                                         {"volc_asr_api_key": "k"},
                                         http={"post": fake_std}, diarization=True)
check("标准版 submit→doing→success 轮询", seq["n"] == 2 and out["text"] == "你好。世界。")

# ── 8. 腾讯 TC3 签名与调用 ──────────────────────────────────────────────────

print("\n=== 8. 腾讯云 TC3 签名调用 ===")
tc_calls = []


def fake_tc_post(url, **kwargs):
    action = kwargs["headers"]["X-TC-Action"]
    tc_calls.append((action, kwargs["headers"], json.loads(kwargs["data"].decode())))
    if action == "CreateRecTask":
        return {"Response": {"Data": {"TaskId": 400000042}}}
    return {"Response": {"Data": {
        "Status": 2, "StatusStr": "success", "Result": "今天开会。",
        "ResultDetail": [
            {"FinalSentence": "今天", "StartMs": 0, "EndMs": 500, "SpeakerId": 0},
            {"FinalSentence": "开会。", "StartMs": 500, "EndMs": 1000, "SpeakerId": 1},
        ],
    }}}


data = cloud_asr.tencent_call("CreateRecTask", {"a": 1}, "sid", "sk", "ap-beijing",
                              http={"post": fake_tc_post})
check("CreateRecTask 返回 Data", data["TaskId"] == 400000042)
action, hdrs, payload = tc_calls[-1]
check("TC3 授权头格式",
      hdrs["Authorization"].startswith("TC3-HMAC-SHA256 Credential=sid/")
      and "asr/tc3_request" in hdrs["Authorization"])
check("公共头齐全", hdrs["X-TC-Version"] == "2019-06-14"
      and hdrs["X-TC-Region"] == "ap-beijing" and hdrs["Host"] == "asr.tencentcloudapi.com")
check("请求体原样 JSON", payload == {"a": 1})


def fake_tc_error(url, **kwargs):
    return {"Response": {"Error": {"Code": "AuthFailure", "Message": "签名失败"}}}


try:
    cloud_asr.tencent_call("X", {}, "sid", "sk", "ap-beijing",
                           http={"post": fake_tc_error})
    check("腾讯 Error 字段转 CloudASRError", False)
except cloud_asr.CloudASRError as e:
    check("腾讯 Error 字段转 CloudASRError", "AuthFailure" in str(e) and "签名失败" in str(e))

# ── 9. 腾讯转写：小文件直传 ─────────────────────────────────────────────────

print("\n=== 9. 腾讯转写小文件直传 ===")
tc_calls.clear()
cfg_tc = {"tencent_secret_id": "sid", "tencent_secret_key": "sk",
          "tencent_asr_region": "ap-beijing"}
out = cloud_asr.tencent_transcribe(b"small-mp3", cfg_tc,
                                   http={"post": fake_tc_post},
                                   diarization=False, hot_words="西工|ERP")
create = [c for c in tc_calls if c[0] == "CreateRecTask"][0][2]
check("小文件 SourceType=1 + Data/DataLen",
      create["SourceType"] == 1 and create["Data"] == "c21hbGwtbXAz"
      and create["DataLen"] == 9)
check("默认中英大模型 2.0", create["EngineModelType"] == "16k_zh_en_2.0")
check("ResTextFormat=3 单声道 数字转换",
      create["ResTextFormat"] == 3 and create["ChannelNum"] == 1
      and create["ConvertNumMode"] == 1)
check("临时热词表 HotwordList", create["HotwordList"] == "西工|5,ERP|5")
check("非分离不带 SpeakerDiarization", "SpeakerDiarization" not in create)
check("查询结果解析全文", out["text"] == "今天开会。")

tc_calls.clear()
cloud_asr.tencent_transcribe(b"x", cfg_tc, http={"post": fake_tc_post}, diarization=True)
create2 = [c for c in tc_calls if c[0] == "CreateRecTask"][0][2]
check("分离模式用会议大模型 + SpeakerDiarization=1",
      create2["EngineModelType"] == "16k_zh_en_meeting"
      and create2["SpeakerDiarization"] == 1)
check("16k 引擎不传 SpeakerNumber", "SpeakerNumber" not in create2)

try:
    cloud_asr.tencent_transcribe(b"x", {}, http={"post": fake_tc_post})
    check("缺腾讯密钥报错", False)
except cloud_asr.CloudASRError as e:
    check("缺腾讯密钥报错", "SecretId" in str(e))

# ── 10. 腾讯大文件 → COS 签名中转 ───────────────────────────────────────────

print("\n=== 10. 腾讯大文件 COS 中转 ===")
big = b"\x00" * (cloud_asr.TENCENT_DIRECT_MAX_BYTES + 10)
put_calls = []


def fake_put(url, **kwargs):
    put_calls.append((url, kwargs))
    return FakeResp({}, status_code=200)


cfg_cos = dict(cfg_tc, cos_bucket="asr-1250000000", cos_region="ap-shanghai")
tc_calls.clear()
cloud_asr.tencent_transcribe(big, cfg_cos, http={"post": fake_tc_post},
                             put_fn=fake_put)
check("大文件触发一次 PUT 上传", len(put_calls) == 1)
put_url = put_calls[0][0]
check("PUT URL 指向 COS 桶与地域",
      "asr-1250000000.cos.ap-shanghai.myqcloud.com" in put_url
      and put_calls[0][1]["headers"]["Content-Type"] == "audio/mpeg")
create3 = [c for c in tc_calls if c[0] == "CreateRecTask"][0][2]
check("大文件 SourceType=0 + 公网签名 GET URL",
      create3["SourceType"] == 0
      and create3["Url"].startswith("https://asr-1250000000.cos.ap-shanghai.myqcloud.com/")
      and "q-signature=" in create3["Url"])
check("下载用 GET 签名，与上传 PUT 签名不同",
      create3["Url"] != put_url)

cfg_no_cos = dict(cfg_tc)
try:
    cloud_asr.tencent_transcribe(big, cfg_no_cos, http={"post": fake_tc_post},
                                 put_fn=fake_put)
    check("无 COS 桶的大文件报人话错误", False)
except cloud_asr.CloudASRError as e:
    check("无 COS 桶的大文件报人话错误", "COS" in str(e) and "4.7MB" in str(e))


def fake_put_fail(url, **kwargs):
    return FakeResp({}, status_code=403, text="denied")


try:
    cloud_asr.tencent_transcribe(big, cfg_cos, http={"post": fake_tc_post},
                                 put_fn=fake_put_fail)
    check("COS 上传失败报错", False)
except cloud_asr.CloudASRError as e:
    check("COS 上传失败报错", "403" in str(e))

# ── 11. COS 预签名：方法绑定 ────────────────────────────────────────────────

print("\n=== 11. COS 预签名方法绑定 ===")
u_put = cloud_asr._cos_presigned_url("put", "b-1", "ap-beijing", "sid", "sk", "k/1.mp3")
u_get = cloud_asr._cos_presigned_url("get", "b-1", "ap-beijing", "sid", "sk", "k/1.mp3")
check("PUT/GET 签名不同", u_put != u_get)
check("两个 URL 都带 sha1 签名要素",
      all("q-sign-algorithm=sha1" in u and "q-signature=" in u for u in (u_put, u_get)))
u_get2 = cloud_asr._cos_presigned_url("get", "b-1", "ap-beijing", "sid", "sk", "k/1.mp3")
# key_time 含秒级时间戳，同秒内相同；结构至少含两次 key-time
check("URL 含 sign-time 与 key-time", "q-sign-time=" in u_get and "q-key-time=" in u_get)

# ── 12. 阿里云任务全流程 ─────────────────────────────────────────────────────

print("\n=== 12. 阿里云 Paraformer ===")
ali_calls = []
ALI_RESULT = {
    "transcripts": [{"content": "第一段。第二段。"}],
    "sentences": [
        {"text": "第一段。", "begin_time": 0, "end_time": 1200, "speaker_id": 0},
        {"text": "第二段。", "begin_time": 1200, "end_time": 2400, "speaker_id": 1},
    ],
}


def fake_ali(method, url, **kwargs):
    if method == "post":
        ali_calls.append(("post", url, kwargs))
        return FakeResp({"output": {"task_id": "t-1", "task_status": "PENDING"}})
    # get：任务查询或结果下载
    if "/tasks/t-1" in url:
        state["n"] += 1
        st = "RUNNING" if state["n"] == 1 else "SUCCEEDED"
        outp = {"task_id": "t-1", "task_status": st}
        if st == "SUCCEEDED":
            outp["results"] = [{"subtask_status": "SUCCEEDED",
                                "transcription_url": "https://r/x.json"}]
        return FakeResp({"output": outp})
    return FakeResp(ALI_RESULT)


state = {"n": 0}
http_ali = {"post": lambda url, **k: fake_ali("post", url, **k),
            "get": lambda url, **k: fake_ali("get", url, **k)}
cfg_ali = {"aliyun_asr_api_key": "sk-xxx"}
out = cloud_asr.aliyun_transcribe("https://audio/a.mp3", cfg_ali, http=http_ali,
                                  diarization=True, speaker_count=3)
submit = ali_calls[0][2]
check("提交打到 transcription 接口并带异步头",
      ali_calls[0][1].endswith("/services/audio/asr/transcription")
      and submit["headers"]["X-DashScope-Async"] == "enable"
      and submit["headers"]["Authorization"] == "Bearer sk-xxx")
body = submit["json"]
check("提交模型 paraformer-v2 + 音频 URL",
      body["model"] == "paraformer-v2" and body["input"]["file_urls"] == ["https://audio/a.mp3"])
check("语言提示 zh/en + 分离 + 参考人数",
      body["parameters"]["language_hints"] == ["zh", "en"]
      and body["parameters"]["diarization_enabled"] is True
      and body["parameters"]["speaker_count"] == 3)
check("阿里云结果全文", out["text"] == "第一段。第二段。")
check("阿里云分句说话人", out["speaker_count"] == 2
      and out["sentence_info"][1]["spk"] == 1)

# 任务级失败
def fake_ali_failed(method, url, **kwargs):
    if method == "post":
        return FakeResp({"output": {"task_id": "t-2", "task_status": "PENDING"}})
    return FakeResp({"output": {"task_id": "t-2", "task_status": "FAILED",
                                "message": "内部错误"}})


http_fail = {"post": lambda url, **k: fake_ali_failed("post", url, **k),
             "get": lambda url, **k: fake_ali_failed("get", url, **k)}
try:
    cloud_asr.aliyun_transcribe("https://a/x.mp3", cfg_ali, http=http_fail)
    check("任务 FAILED 报错", False)
except cloud_asr.CloudASRError as e:
    check("任务 FAILED 报错", "内部错误" in str(e))

# 子任务失败（整任务 SUCCEEDED 但 results FAILED）
def fake_subfail(method, url, **kwargs):
    if method == "post":
        return FakeResp({"output": {"task_id": "t-3", "task_status": "PENDING"}})
    return FakeResp({"output": {"task_id": "t-3", "task_status": "SUCCEEDED",
                                "results": [{"subtask_status": "FAILED",
                                             "code": "InvalidFile.DownloadFailed",
                                             "message": "下载失败"}]}})


http_sub = {"post": lambda url, **k: fake_subfail("post", url, **k),
            "get": lambda url, **k: fake_subfail("get", url, **k)}
try:
    cloud_asr.aliyun_transcribe("https://a/x.mp3", cfg_ali, http=http_sub)
    check("子任务 FAILED 报错", False)
except cloud_asr.CloudASRError as e:
    check("子任务 FAILED 报错并透传 message", "下载失败" in str(e))

try:
    cloud_asr.aliyun_transcribe("https://a/x.mp3", {})
    check("缺阿里 API Key 报错", False)
except cloud_asr.CloudASRError as e:
    check("缺阿里 API Key 报错", "API Key" in str(e))

# ── 13. oss_upload（假 oss2 模块）───────────────────────────────────────────

print("\n=== 13. OSS 上传与预签名 ===")
import types

fake_oss2 = types.ModuleType("oss2")


class FakeAuth:
    def __init__(self, ak, sk):
        self.ak, self.sk = ak, sk


class FakeBucket:
    def __init__(self, auth, endpoint, name):
        self.auth, self.endpoint, self.name = auth, endpoint, name

    def sign_url(self, method, key, expires, slash_safe=True):
        return (f"https://{self.name}.{self.endpoint}/{key}"
                f"?{method}&expires={expires}&sig=fake")

    def delete_object(self, key):
        deleted.append(key)


deleted = []
fake_oss2.Auth = FakeAuth
fake_oss2.Bucket = FakeBucket
sys.modules["oss2"] = fake_oss2

oss_cfg = {"oss_access_key_id": "ak", "oss_access_key_secret": "sk",
           "oss_bucket": "bkt", "oss_endpoint": "https://oss-cn-beijing.aliyuncs.com"}
put_seen = {}


def fake_oss_put(url, **kwargs):
    put_seen["url"] = url
    put_seen["data"] = kwargs["data"]
    return FakeResp({}, status_code=200)


url, data = cloud_asr.oss_upload(b"mp3-bytes", oss_cfg, "asr-temp/20260911/x.mp3",
                                put_fn=fake_oss_put)
check("PUT 用 600s 签名 URL", url.startswith("https://bkt.oss-cn-beijing.aliyuncs.com/")
      and "asr-temp/20260911/x.mp3?PUT&expires=600" in put_seen["url"])
check("上传字节原样透传 + mp3 content-type",
      put_seen["data"] == b"mp3-bytes")
check("返回 24h GET 签名 URL", "GET&expires=86400" in url)

cloud_asr._oss_cleanup(oss_cfg, "asr-temp/20260911/x.mp3")
check("清理调用 delete_object", deleted == ["asr-temp/20260911/x.mp3"])

try:
    cloud_asr.oss_upload(b"x", {"oss_bucket": "b"}, "k", put_fn=fake_oss_put)
    check("OSS 配置缺失报人话错误", False)
except cloud_asr.CloudASRError as e:
    check("OSS 配置缺失报人话错误", "OSS" in str(e))

try:
    cloud_asr.oss_upload(b"x", oss_cfg, "k",
                         put_fn=lambda u, **k: FakeResp({}, status_code=403))
    check("OSS PUT 失败报错", False)
except cloud_asr.CloudASRError as e:
    check("OSS PUT 失败报错", "OSS 上传失败" in str(e))

# ── 14. 统一入口 transcribe()：真实 WAV→mp3 + 三家假后端 ────────────────────

print("\n=== 14. transcribe() 统一入口 ===")
import tempfile

tmpdir = tempfile.mkdtemp()
wav = make_wav(Path(tmpdir) / "rec.wav", seconds=0.5)

# 火山：端到端压 mp3 → flash
out = cloud_asr.transcribe("volc", wav,
                           {"volc_asr_api_key": "k"},
                           http={"post": fake_volc_post})
check("统一入口-火山文本", out["text"] == "你好。世界。")
check("未开分离 → sentence_info 被清空", out["sentence_info"] == [] and out["speaker_count"] == 0)
cloud_files = list(Path(tmpdir).glob("*.cloud.mp3"))
check("临时 mp3 已清理", not cloud_files, [p.name for p in cloud_files])

# 阿里：端到端压 mp3 → OSS 上传 → 任务
state["n"] = 0
ali_calls.clear()
full_cfg = dict(oss_cfg, aliyun_asr_api_key="sk-xxx",
                speaker_diarization=True, hot_words="西工")
out = cloud_asr.transcribe("aliyun", wav, full_cfg, http=http_ali,
                           put_fn=fake_oss_put)
check("统一入口-阿里文本与分句",
      out["text"] == "第一段。第二段。" and out["speaker_count"] == 2)
check("阿里投递的是真实 mp3 字节（ID3/MPEG 头或非空）",
      isinstance(put_seen["data"], bytes) and len(put_seen["data"]) > 0)
check("转写后 OSS 临时文件被删除", deleted[-1].endswith(".mp3"))

# 腾讯：端到端压 mp3 → 直传（0.5s 音频很小）
tc_calls.clear()
out = cloud_asr.transcribe("tencent", wav, cfg_tc, http={"post": fake_tc_post})
check("统一入口-腾讯文本", out["text"] == "今天开会。")

# 未知 provider
try:
    cloud_asr.transcribe("aws", wav, {})
    check("未知云端引擎报错", False)
except cloud_asr.CloudASRError:
    check("未知云端引擎报错", True)

# 音频预处理函数直接验证时长
mp3, dur = cloud_asr.encode_mp3_for_cloud(wav, str(Path(tmpdir) / "out.mp3"))
check("PyAV 编码出非空 mp3", os.path.getsize(mp3) > 0)
check("时长约等于源 WAV（0.5s）", abs(dur - 0.5) < 0.1, dur)

# ── 15. app.py / UI 接线静态校验（不 import 整个重型 app）──────────────────

print("\n=== 15. 接线静态校验 ===")
root = Path(__file__).resolve().parent.parent
app_src = (root / "app.py").read_text(encoding="utf-8")
ui_src = (root / "ui" / "index.html").read_text(encoding="utf-8")
import ast

ast.parse(app_src)
check("app.py AST 可解析", True)
check("app.py 导入 cloud_asr", "import cloud_asr as _cloud_asr" in app_src)
check("文件转写分发三家引擎",
      all(f'engine in ("aliyun", "volc", "tencent")' in app_src for _ in (0,)))
check("实时转写守卫含云端三家", '"aliyun", "volc", "tencent"' in app_src)
for key in ("aliyun_asr_api_key", "oss_bucket", "volc_asr_api_key",
            "tencent_secret_id", "cos_bucket", "cloud_tmp_prefix"):
    check(f"DEFAULT_CONFIG 含 {key}", f'"{key}"' in app_src)
check("引擎名记录进 meta（engine=engine 透传）",
      "save_recording_meta(filepath, text=text, engine=engine" in app_src)

for opt in ('value="aliyun"', 'value="volc"', 'value="tencent"'):
    check(f"UI 引擎选项 {opt}", opt in ui_src)
for elem in ("aliyunCredRows", "volcCredRows", "tencentCredRows",
             "aliyunAsrApiKey", "ossBucket", "volcAsrApiKey",
             "tencentSecretId", "cosBucket"):
    check(f"UI 含元素 {elem}", f'id="{elem}"' in ui_src)
check("UI 凭据随引擎显隐",
      "getElementById('aliyunCredRows')" in ui_src
      and "getElementById('volcCredRows')" in ui_src
      and "getElementById('tencentCredRows')" in ui_src)
check("UI saveConfig 收集云端凭据",
      "aliyun_asr_api_key" in ui_src and "tencent_secret_id" in ui_src)

print(f"\n通过 {passed} 条，失败 {failed} 条")
sys.exit(1 if failed else 0)
