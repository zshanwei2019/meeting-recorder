# -*- coding: utf-8 -*-
"""Qwen3ASRTranscriber 设备/精度自动选择（bf16 / Arc XPU）+ 健壮解码。

CI（requirements-ci.txt）故意不装 torch/funasr/librosa/soundfile，
所以本测试不 import torch：核心决策走纯函数 _qwen3_device_choice（返回字符串 tag），
接线用源码断言；torch dtype 映射与真实解码仅在本机有这些包时才额外验证。
"""
import ast
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app   # noqa: E402

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        print(f"  OK   {name}")
        passed += 1
    else:
        print(f"  FAIL {name} {detail}")
        failed += 1


def choice(has_xpu, bf16_ok, **env):
    """调用无 torch 依赖的纯决策函数。"""
    return app._qwen3_device_choice(has_xpu, bf16_ok, env=env)


print("=== 1. 纯决策：dtype=fp32 优先级最高 ===")
check("无 XPU、支持 bf16、默认 -> cpu/bf16",
      choice(False, True) == ("cpu", "bf16"), choice(False, True))
check("无 XPU、不支持 bf16、默认 -> cpu/fp32",
      choice(False, False) == ("cpu", "fp32"), choice(False, False))
check("有 XPU、支持 bf16、默认 -> xpu/fp16",
      choice(True, True) == ("xpu", "fp16"), choice(True, True))
check("有 XPU、不支持 bf16、默认仍 -> xpu/fp16（XPU 不用 CPU bf16）",
      choice(True, False) == ("xpu", "fp16"), choice(True, False))

print("\n=== 2. QWEN3_ASR_DEVICE 强制 ===")
check("强制 cpu（即便有 XPU）-> cpu/bf16",
      choice(True, True, QWEN3_ASR_DEVICE="cpu") == ("cpu", "bf16"))
check("强制 cpu 且不支持 bf16 -> cpu/fp32",
      choice(True, False, QWEN3_ASR_DEVICE="cpu") == ("cpu", "fp32"))
check("强制 xpu（即便探测无 XPU，尊重用户）-> xpu/fp16",
      choice(False, True, QWEN3_ASR_DEVICE="xpu") == ("xpu", "fp16"))
check("大小写/空白容错",
      choice(True, True, QWEN3_ASR_DEVICE=" CPU ") == ("cpu", "bf16"))

print("\n=== 3. QWEN3_ASR_DTYPE=fp32 覆盖一切 ===")
check("有 XPU 且强制 xpu，但 dtype=fp32 -> cpu/fp32",
      choice(True, True, QWEN3_ASR_DEVICE="xpu", QWEN3_ASR_DTYPE="fp32")
      == ("cpu", "fp32"))
check("默认设备 + dtype=fp32 -> cpu/fp32",
      choice(True, True, QWEN3_ASR_DTYPE="fp32") == ("cpu", "fp32"))

print("\n=== 4. 决策结果 tag 恒为合法集合 ===")
valid = {("xpu", "fp16"), ("cpu", "bf16"), ("cpu", "fp32")}
for hx in (False, True):
    for bo in (False, True):
        for dev in (None, "cpu", "xpu"):
            for dt in (None, "fp32"):
                env = {}
                if dev:
                    env["QWEN3_ASR_DEVICE"] = dev
                if dt:
                    env["QWEN3_ASR_DTYPE"] = dt
                r = choice(hx, bo, **env)
                check(f"合法组合 {r}", r in valid, r)

print("\n=== 5. torch 存在时：纯决策映射为真实 dtype（本机才有）===")
try:
    import torch  # noqa: E402
    torch_ok = True
except Exception:
    torch_ok = False
if torch_ok:
    # 用 monkeypatch 掉能力探测，验证 _select_qwen3_device 的 dtype 映射
    for hx, bo, expect in [(False, True, torch.bfloat16),
                           (False, False, torch.float32),
                           (True, True, torch.float16)]:
        app._CPU_BF16_OK = bo
        orig = getattr(torch.xpu, "is_available", None) if hasattr(torch, "xpu") else None
        if hasattr(torch, "xpu"):
            torch.xpu.is_available = lambda hx=hx: hx
        os.environ.pop("QWEN3_ASR_DEVICE", None)
        os.environ.pop("QWEN3_ASR_DTYPE", None)
        try:
            dev, dtype = app._select_qwen3_device()
            tag = {torch.float16: "fp16", torch.bfloat16: "bf16",
                   torch.float32: "fp32"}[dtype]
            check(f"映射 {dev}/{tag} 正确", dtype == expect, (dev, dtype))
        finally:
            if orig is not None:
                torch.xpu.is_available = orig
    app._CPU_BF16_OK = None
else:
    print("  (skip: CI 精简环境无 torch —— 真实 dtype 映射由纯函数覆盖)")

print("\n=== 6. 接线（源码断言）===")
src = io.open(
    Path(__file__).resolve().parent.parent / "app.py", encoding="utf-8").read()
check("load 调用 _select_qwen3_device",
      "device, dtype = _select_qwen3_device()" in src)
check("from_pretrained 不再硬编码 device_map=cpu/float32",
      'device_map="cpu"' not in src)
check("模型 .to(device).eval()", ").to(device).eval()" in src)
check("transcribe 把 inputs 搬到 device/dtype",
      "inputs.to(self._device, self._dtype)" in src)
check("XPU 路径有 synchronize（加载与生成各一次）",
      src.count("torch.xpu.synchronize()") >= 2)
check("实例保存 _device/_dtype",
      "self._device = device" in src and "self._dtype = dtype" in src)
check("解码绕开 torchcodec：存在 _load_audio_for_qwen3",
      "def _load_audio_for_qwen3" in src)
check("解码有 soundfile 直读 + ffmpeg 兜底",
      "sf.read(" in src and "ffmpeg" in src)
check("不再把文件路径直接丢给 processor",
      "apply_transcription_request(audio=filepath)" not in src)

print("\n=== 7. 解码函数（有 soundfile 才实测，CI 无则跳过）===")
try:
    import numpy as np  # noqa: E402
    import soundfile as sf  # noqa: E402
    import tempfile  # noqa: E402
    dec_ok = True
except Exception:
    dec_ok = False
if dec_ok:
    # 合成 1s / 22050Hz / 立体声 PCM16 wav，验证重采样 16k + 混单声道
    sr0 = 22050
    tt = np.arange(sr0) / sr0
    tone = 0.3 * np.sin(2 * np.pi * 440 * tt)
    stereo = np.stack([tone, tone * 0.5], axis=1)
    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(wav_path, stereo, sr0, subtype="PCM_16")
        w, srr = app._load_audio_for_qwen3(wav_path)
        check("解码返回 numpy 波形", isinstance(w, np.ndarray), type(w))
        check("重采样到 16000 Hz", srr == 16000, srr)
        check("混为单声道", getattr(w, "ndim", 9) == 1, getattr(w, "shape", None))
        check("长度约 1s", 15000 < len(w) < 17000, len(w))
        check("dtype float32", w.dtype == np.float32, w.dtype)
    finally:
        os.remove(wav_path)
else:
    print("  (skip: CI 精简环境无 soundfile/librosa —— 真实解码由本机冒烟覆盖)")

print("\n=== 8. AST 校验 ===")
ast.parse(src)
check("app.py AST 解析通过", True)

print(f"\n通过 {passed} 条，失败 {failed} 条")
sys.exit(1 if failed else 0)
