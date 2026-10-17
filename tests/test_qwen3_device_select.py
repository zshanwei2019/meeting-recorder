# -*- coding: utf-8 -*-
"""Qwen3ASRTranscriber 设备/精度自动选择（bf16 / Arc XPU）。

不加载任何模型，只验证 _select_qwen3_device 在各环境变量组合下的决策，
以及 Qwen3ASRTranscriber 把 device/dtype 传给 from_pretrained/to/generate 的接线。

背景（2026-10-17，Core Ultra X7 358H + Arc B390 实测）：
- CPU bf16 比旧 fp32 快约 35 倍，转写字词一致；
- Arc XPU 需 xpu 版 torch>=2.14.1（2.9.1 的 LLM 算子回退单核 CPU 会卡死），fp16 约 3 s/30s 片。
"""
import ast
import io
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
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


def select_with(**env):
    """以指定环境变量调用一次选择函数（返回前清理，避免污染其它用例）。"""
    keys = ("QWEN3_ASR_DEVICE", "QWEN3_ASR_DTYPE")
    old = {k: os.environ.get(k) for k in keys}
    app._CPU_BF16_OK = None
    try:
        for k in keys:
            if env.get(k) is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = env[k]
        return app._select_qwen3_device()
    finally:
        app._CPU_BF16_OK = None
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


print("=== 1. 环境变量覆盖逻辑 ===")
# 强制 fp32 回退（最高优先级，即便同时强制 xpu 也应是 cpu/fp32）
dev, dt = select_with(QWEN3_ASR_DTYPE="fp32", QWEN3_ASR_DEVICE="xpu")
check("QWEN3_ASR_DTYPE=fp32 覆盖一切 -> cpu/fp32",
      dev == "cpu" and dt == torch.float32, (dev, dt))

# 强制 cpu（生产 venv 无 xpu，最常用路径）
dev, dt = select_with(QWEN3_ASR_DEVICE="cpu")
check("QWEN3_ASR_DEVICE=cpu -> cpu，且为 bf16 或 fp32",
      dev == "cpu" and dt in (torch.bfloat16, torch.float32), (dev, dt))

# 强制 xpu：即使 torch 是 cpu 构建也尊重用户选择（返回 xpu/fp16；
# 真正加载时若环境无 XPU 会在 .to('xpu') 处报错并被 load 的 except 捕获）
dev, dt = select_with(QWEN3_ASR_DEVICE="xpu")
check("QWEN3_ASR_DEVICE=xpu -> xpu/fp16",
      dev == "xpu" and dt == torch.float16, (dev, dt))

print("\n=== 2. 自动选择（不强制设备）===")
dev, dt = select_with()
if hasattr(torch, "xpu") and torch.xpu.is_available():
    check("xpu torch 且可用 -> 自动 xpu/fp16",
          dev == "xpu" and dt == torch.float16, (dev, dt))
else:
    check("cpu torch -> 自动 cpu 且为 bf16 或 fp32",
          dev == "cpu" and dt in (torch.bfloat16, torch.float32), (dev, dt))
check("自动选择永远不会在 CPU 上返回 fp16",
      not (dev == "cpu" and dt == torch.float16), (dev, dt))

print("\n=== 3. bf16 能力探测可缓存且返回 bool ===")
app._CPU_BF16_OK = None
ok1 = app._cpu_bf16_supported(torch)
ok2 = app._cpu_bf16_supported(torch)
check("_cpu_bf16_supported 返回 bool", isinstance(ok1, bool), type(ok1))
check("探测结果两次一致（已缓存）", ok1 == ok2 == app._CPU_BF16_OK)
app._CPU_BF16_OK = None

print("\n=== 4. 接线：load 把 device/dtype 传下去，transcribe 搬运 inputs ===")
src = io.open(
    Path(__file__).resolve().parent.parent / "app.py", encoding="utf-8").read()
check("load 调用 _select_qwen3_device",
    "device, dtype = _select_qwen3_device()" in src)
check("from_pretrained 不再硬编码 device_map=cpu/float32",
    'device_map="cpu"' not in src and "dtype=torch.float32,\n                )" not in src)
check("模型 .to(device)", ").to(device).eval()" in src)
check("transcribe 把 inputs 搬到 device/dtype",
    "inputs.to(self._device, self._dtype)" in src)
check("XPU 路径有 synchronize（计时/收尾正确）",
    src.count("torch.xpu.synchronize()") >= 2)
check("实例保存 _device/_dtype",
    "self._device = device" in src and "self._dtype = dtype" in src)

print("\n=== 5. _load_audio_for_qwen3 解码（合成 wav，不依赖 torchcodec）===")
import tempfile, wave, struct  # noqa: E402
try:
    import soundfile  # noqa: F401
    sf_ok = True
except Exception:
    sf_ok = False

if sf_ok:
    # 造一个 1s、22050Hz、立体声的 16-bit PCM wav，验证重采样到 16k + 混单声道
    sr0, secs = 22050, 1
    t_axis = np.arange(sr0 * secs) / sr0
    tone = 0.3 * np.sin(2 * np.pi * 440 * t_axis)
    stereo = np.stack([tone, tone * 0.5], axis=1)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        wav_path = f.name
    try:
        import soundfile as sf
        sf.write(wav_path, stereo, sr0, subtype="PCM_16")
        w, srr = app._load_audio_for_qwen3(wav_path)
        check("解码返回 numpy 波形", isinstance(w, np.ndarray), type(w))
        check("重采样到 16000 Hz", srr == 16000, srr)
        check("混为单声道", getattr(w, "ndim", 9) == 1, getattr(w, "shape", None))
        check("长度约 1s（15000~17000 样本）", 15000 < len(w) < 17000, len(w))
        check("dtype float32", w.dtype == np.float32, w.dtype)
    finally:
        os.remove(wav_path)
else:
    print("  (skip: soundfile 不可用)")

print("\n=== 6. AST 校验 ===")
ast.parse(src)
check("app.py AST 解析通过", True)

print(f"\n通过 {passed} 条，失败 {failed} 条")
sys.exit(1 if failed else 0)
