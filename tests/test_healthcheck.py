#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""healthcheck.py 的纯逻辑回归测试。

只测文件系统判定（不碰声卡 / torch）：用临时目录模拟模型
missing / partial / ok 三态、环回设备识别、磁盘与数据目录检查、汇总总判定。

运行：venv\\Scripts\\python.exe tests\\test_healthcheck.py
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import healthcheck as hc  # noqa: E402

PASS = []
FAIL = []


def check(name, cond, hint=""):
    if cond:
        PASS.append(name)
        print("  OK   ", name)
    else:
        FAIL.append(name)
        print("  FAIL ", name, hint)


print("=== 1. modelscope 模型三态 ===")
with tempfile.TemporaryDirectory() as td:
    ms = Path(td) / "ms"
    hf = Path(td) / "hf"

    ok_id = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
    bad_id = "iic/punc_ct-transformer_cn-en-common-vocab471067-large"
    none_id = "iic/not-here-at-all"
    entry_ok = {"id": ok_id, "label": "a", "kind": "core", "provider": "modelscope"}
    entry_bad = {"id": bad_id, "label": "b", "kind": "core", "provider": "modelscope"}
    entry_none = {"id": none_id, "label": "c", "kind": "core", "provider": "modelscope"}

    # 三态初始都应是 missing
    for e in (entry_ok, entry_bad, entry_none):
        check(f"初始缺失 {e['id']}", hc.check_model(e, ms, hf)["status"] == "missing")

    # 完整：snapshots/master/model.pt 2MB
    snap = hc.modelscope_snapshot(ok_id, ms)
    snap.mkdir(parents=True)
    (snap / "model.pt").write_bytes(b"x" * (2 * 1024 * 1024))
    (snap / "config.yaml").write_text("cfg", encoding="utf-8")
    r = hc.check_model(entry_ok, ms, hf)
    check("完整模型=ok", r["status"] == "ok", str(r))
    check("完整模型体积≥2MB", r["sizeMb"] >= 1.9, str(r["sizeMb"]))

    # 残缺：目录存在但只有配置、无权重
    bsnap = hc.modelscope_snapshot(bad_id, ms)
    bsnap.mkdir(parents=True)
    (bsnap / "config.yaml").write_text("cfg", encoding="utf-8")
    rb = hc.check_model(entry_bad, ms, hf)
    check("无权重=partial", rb["status"] == "partial", str(rb))

    # 0 字节权重不算就绪（下载中断占位）
    (bsnap / "model.pt").write_bytes(b"")
    rb2 = hc.check_model(entry_bad, ms, hf)
    check("0字节权重仍=partial", rb2["status"] == "partial", str(rb2))

    # 缺失不变
    check("不存在=missing", hc.check_model(entry_none, ms, hf)["status"] == "missing")

    print("=== 2. HuggingFace 模型三态 ===")
    py_id = "pyannote/speaker-diarization-3.1"
    entry_py = {"id": py_id, "label": "p", "kind": "optional", "provider": "huggingface"}
    check("HF 初始 missing", hc.check_model(entry_py, ms, hf)["status"] == "missing")
    pysnap = hc.hf_snapshot(py_id, hf)
    check("HF 无快照时返回 None", pysnap is None)
    # 造 HF 布局 models--pyannote--.../snapshots/<rev>/pytorch_model.bin
    rev = hf / "models--pyannote--speaker-diarization-3.1" / "snapshots" / "abc123"
    rev.mkdir(parents=True)
    (rev / "pytorch_model.bin").write_bytes(b"y" * (3 * 1024 * 1024))
    got = hc.hf_snapshot(py_id, hf)
    check("HF 找到 rev 目录", got is not None and got.name == "abc123", str(got))
    rp = hc.check_model(entry_py, ms, hf)
    check("HF 完整=ok", rp["status"] == "ok" and rp["sizeMb"] >= 2.9, str(rp))

    # configOnly 仓库（pyannote pipeline 主仓库天生只有 config.yaml，权重在子模型）
    cfg_id = "pyannote/speaker-diarization-3.1"
    entry_cfg = {"id": cfg_id, "label": "p", "kind": "optional",
                 "provider": "huggingface", "configOnly": True}
    cfg_rev = hf / "models--pyannote--speaker-diarization-3.1" / "snapshots" / "cfg001"
    cfg_rev.mkdir(parents=True)
    check("configOnly 空快照=partial", hc.check_model(entry_cfg, ms, hf)["status"] == "partial")
    (cfg_rev / "config.yaml").write_text("version: 3.1.0\n", encoding="utf-8")
    rc = hc.check_model(entry_cfg, ms, hf)
    check("configOnly 仅 config.yaml 即=ok（不误报残缺）", rc["status"] == "ok", str(rc))
    check("configOnly 标记透传", rc.get("configOnly") is True)
    # 0 字节配置不算就绪
    (cfg_rev / "config.yaml").write_bytes(b"")
    check("configOnly 0字节配置仍=partial", hc.check_model(entry_cfg, ms, hf)["status"] == "partial")

    print("=== 3. run_health_check 总判定 ===")
    # CI/无头环境无真实音频设备，注入假设备检查，保证总判定只反映模型状态
    def _fake_devices_ok():
        return {"status": "ok", "inputs": [], "hasLoopback": True,
                "loopbackNames": ["fake"], "detail": "注入"}
    rep = hc.run_health_check(config={}, ms_base=ms, hf_base=hf,
                              data_dir=Path(td) / "data", device_check=_fake_devices_ok)
    # 临时目录里只有 fsmn 一个核心模型完整，punct 残缺 => 总体不 ok
    check("核心缺失时 ok=False", rep["ok"] is False, str(rep["summary"]))
    check("summary 记录 coreMissing", bad_id in rep["summary"]["coreMissing"], str(rep["summary"]))
    check("磁盘检查在临时目录可写", rep["disk"]["writable"] is True)

    # 把另一个核心模型也补全 + 去掉残缺 => ok=True
    (bsnap / "model.pt").write_bytes(b"z" * (2 * 1024 * 1024))
    # 注册表有 3 个核心模型，还需补上 paraformer-large-vad-punc
    pld = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
    plsnap = hc.modelscope_snapshot(pld, ms)
    plsnap.mkdir(parents=True)
    (plsnap / "model.pt").write_bytes(b"q" * (2 * 1024 * 1024))
    rep2 = hc.run_health_check(config={}, ms_base=ms, hf_base=hf,
                               data_dir=Path(td) / "data2", device_check=_fake_devices_ok)
    check("核心齐全时 ok=True", rep2["ok"] is True, str(rep2["summary"]))

print("=== 4. 环回设备关键词识别 ===")
for name, want in [
    ("立体声混音 (Realtek HD Audio Stereo input)", True),
    ("Stereo Mix (Realtek)", True),
    ("CABLE Input (VB-Audio Virtual Cable)", True),
    ("麦克风阵列 (适用于数字麦克风的英特尔® 智能声音技术)", False),
    ("Microsoft 声音映射器 - Input", False),
]:
    check(f"环回识别 '{name[:14]}'", hc._is_loopback(name) == want)

print("=== 5. 音频检查在无 sounddevice 环境降级（不崩）===")
# sounddevice 在测试 venv 里可能装了；这里只验证函数返回结构完整、永不抛异常
dev = hc.check_audio_devices()
check("设备检查返回 status", dev["status"] in ("ok", "warn", "error"))
check("设备检查含 inputs 列表", isinstance(dev["inputs"], list))
check("设备检查含 detail", isinstance(dev["detail"], str))

print("=== 6. 模型注册表完整性 ===")
ids = [m["id"] for m in hc.MODEL_REGISTRY]
check("注册表无重复 id", len(ids) == len(set(ids)))
check("含 3 个核心模型", sum(1 for m in hc.MODEL_REGISTRY if m["kind"] == "core") == 3)
check("provider 仅 modelscope/huggingface",
      all(m["provider"] in ("modelscope", "huggingface") for m in hc.MODEL_REGISTRY))

print("=== 7. ModelDownloader（注入假下载，不联网）===")
import time as _time

with tempfile.TemporaryDirectory() as td2:
    dms = Path(td2) / "ms"

    def _wait_jobs(dl, timeout=8):
        t0 = _time.time()
        while _time.time() - t0 < timeout:
            st = dl.status(ms_base=dms)
            if st and all(j["status"] in ("done", "error") for j in st):
                return st
            _time.sleep(0.03)
        return dl.status(ms_base=dms)

    # 7.1 假下载成功：真的在缓存里造出权重文件
    def fake_ok(mid):
        snap = hc.modelscope_snapshot(mid, dms)
        snap.mkdir(parents=True, exist_ok=True)
        (snap / "model.pt").write_bytes(b"w" * (2 * 1024 * 1024))
        return str(snap)

    dl = hc.ModelDownloader(downloader_fn=fake_ok)
    res = dl.start(["iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"], ms_base=dms)
    check("合法模型进入 started", len(res["started"]) == 1, str(res))
    jobs = _wait_jobs(dl)
    check("假下载后状态 done", jobs and jobs[0]["status"] == "done", str(jobs))
    # done 后 check_model 复核应真的 ok
    chk = hc.check_model(
        {"id": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", "label": "x",
         "kind": "core", "provider": "modelscope"}, dms, None)
    check("下载后自检复核 ok", chk["status"] == "ok", str(chk))

    # 7.2 已就绪的再请求 -> skipped，不开任务
    res2 = dl.start(["iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"], ms_base=dms)
    check("已就绪被跳过", len(res2["started"]) == 0 and len(res2["skipped"]) == 1, str(res2))

    # 7.3 gated / 未知模型被拒绝
    res3 = dl.start(["pyannote/speaker-diarization-3.1", "iic/not-exist"], ms_base=dms)
    check("gated+未知模型不入队", len(res3["started"]) == 0 and len(res3["skipped"]) == 2, str(res3))

    # 7.4 下载失败 -> error，不崩
    def fake_fail(mid):
        raise RuntimeError("network down")
    dlf = hc.ModelDownloader(downloader_fn=fake_fail)
    dlf.start(["iic/SenseVoiceSmall"], ms_base=dms)
    fjobs = _wait_jobs(dlf)
    check("下载异常记为 error", fjobs and fjobs[0]["status"] == "error" and "network down" in fjobs[0]["message"], str(fjobs))

    # 7.5 假成功但没造出权重 -> 复核降级 error（防空报成功）
    dlf2 = hc.ModelDownloader(downloader_fn=lambda mid: "whatever")
    dlf2.start(["iic/SenseVoiceSmall"], ms_base=dms)
    f2 = _wait_jobs(dlf2)
    check("无权重时 done 被复核改 error", f2 and f2[0]["status"] == "error", str(f2))

print("=== 8. HuggingFace gated 模型下载（注入假 HF 下载器，不联网）===")

class _FakeGatedError(Exception):
    """类名含 Gated，_is_auth_error 据此识别为授权失败。"""


with tempfile.TemporaryDirectory() as td3:
    dhf = Path(td3) / "hf"

    def _wait_hf(dl, timeout=8):
        t0 = _time.time()
        while _time.time() - t0 < timeout:
            st = dl.status(hf_base=dhf)
            if st and all(j["status"] in ("done", "error") for j in st):
                return st
            _time.sleep(0.03)
        return dl.status(hf_base=dhf)

    SEG = "pyannote/segmentation-3.0"
    CFG = "pyannote/speaker-diarization-3.1"

    # 8.1 无 token：拒绝并入队，给出条款接受链接
    dl0 = hc.ModelDownloader(hf_downloader_fn=lambda m, t: "x")
    r0 = dl0.start([SEG], hf_base=dhf, hf_token="")
    check("HF 无 token 不入队", r0["started"] == [] and len(r0["skipped"]) == 1, str(r0))
    check("HF 无 token 给条款链接",
          r0["skipped"][0].get("acceptUrl", "").endswith(SEG), str(r0["skipped"]))
    check("无 token 时不调用 HF 下载器",
          dl0.status(hf_base=dhf) == [])

    # 8.2 带 token 成功：下载器收到 token，造出权重文件后 done 且复核 ok
    seen = {}

    def hf_ok(mid, token):
        seen["token"] = token
        rev = hc.hf_snapshot(mid, dhf)
        if rev is None:
            rev = dhf / ("models--" + mid.replace("/", "--")) / "snapshots" / "main"
        rev.mkdir(parents=True, exist_ok=True)
        (rev / "pytorch_model.bin").write_bytes(b"w" * (2 * 1024 * 1024))
        return str(rev)

    dl1 = hc.ModelDownloader(hf_downloader_fn=hf_ok)
    r1 = dl1.start([SEG], hf_base=dhf, hf_token="hf_secret_123")
    check("HF 带 token 入队", r1["started"] == [SEG], str(r1))
    j1 = _wait_hf(dl1)
    check("HF 下载成功 done", j1 and j1[0]["status"] == "done", str(j1))
    check("HF 下载器确实收到 token", seen.get("token") == "hf_secret_123", str(seen))
    chk1 = hc.check_model(
        {"id": SEG, "label": "x", "kind": "optional", "provider": "huggingface"},
        None, dhf)
    check("HF 下载后复核 ok", chk1["status"] == "ok", str(chk1))

    # 8.3 授权失败（GatedRepoError / 403）：error 信息含条款链接与中文指引
    def hf_gated(mid, token):
        raise _FakeGatedError("403 Client Error: Cannot access gated repo")

    dl2 = hc.ModelDownloader(hf_downloader_fn=hf_gated)
    dl2.start(["pyannote/wespeaker-voxceleb-resnet34-LM"], hf_base=dhf, hf_token="hf_x")
    j2 = _wait_hf(dl2)
    msg2 = j2[0]["message"] if j2 else ""
    check("HF 授权失败记为 error", j2 and j2[0]["status"] == "error", str(j2))
    check("授权错误提示去网页接受条款", "条款" in msg2 and "Agree" in msg2, msg2)
    check("授权错误给出模型页面链接", "huggingface.co/pyannote/wespeaker" in msg2, msg2)
    check("_is_auth_error 识别 401/403/gated",
          hc._is_auth_error(_FakeGatedError("401")) and hc._is_auth_error(RuntimeError("403 forbidden")))

    # 8.4 configOnly 模型成功：只造 config.yaml 也应 done 且复核 ok
    def hf_cfg_ok(mid, token):
        rev = dhf / ("models--" + mid.replace("/", "--")) / "snapshots" / "c1"
        rev.mkdir(parents=True, exist_ok=True)
        (rev / "config.yaml").write_text("version: 3.1.0\n", encoding="utf-8")
        return str(rev)

    dl3 = hc.ModelDownloader(hf_downloader_fn=hf_cfg_ok)
    dl3.start([CFG], hf_base=dhf, hf_token="hf_z")
    j3 = _wait_hf(dl3)
    check("configOnly HF 下载 done", j3 and j3[0]["status"] == "done", str(j3))
    chk3 = hc.check_model(
        {"id": CFG, "label": "x", "kind": "optional",
         "provider": "huggingface", "configOnly": True}, None, dhf)
    check("configOnly HF 下载后复核 ok", chk3["status"] == "ok", str(chk3))

print()
print(f"通过 {len(PASS)} 项，失败 {len(FAIL)} 项")
if FAIL:
    print("失败项:", FAIL)
    print("存在失败 FAIL")
    sys.exit(1)
print("全部通过 PASS")
