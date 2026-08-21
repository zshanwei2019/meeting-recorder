# -*- coding: utf-8 -*-
"""文件转写默认模型切换：SenseVoiceSmall -> paraformer-large-vad-punc。

背景：在 11 分钟真实录音（多说话人+背景音）上实测：
- paraformer-large-vad-punc 同音字、数字、语种漂移全面优于 SenseVoiceSmall；
- CPU RTF 反而更低（0.12 vs 0.13）；
- 该模型本就是说话人分离 pipeline 的 ASR 骨干，权重已在本地。

本测试不加载任何模型，只验证接线与默认值。
"""
import ast
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402

PARA = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
SENSE = "iic/SenseVoiceSmall"

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


print("=== 1. 模型常量与别名解析 ===")
check("默认文件模型 = paraformer-large-vad-punc",
      app.FILE_MODEL_PARAFORMER_LARGE == PARA, app.FILE_MODEL_PARAFORMER_LARGE)
check("SenseVoice 常量保留", app.FILE_MODEL_SENSEVOICE == SENSE)
check("短名 paraformer-large 归一化", app._resolve_file_model("paraformer-large") == PARA)
check("短名 sensevoice 归一化", app._resolve_file_model("sensevoice") == SENSE)
check("短名大小写不敏感", app._resolve_file_model("PARAFORMER-LARGE") == PARA)
check("完整 ID 原样返回", app._resolve_file_model(PARA) == PARA)
check("完整 ID SenseVoice 原样返回", app._resolve_file_model(SENSE) == SENSE)
check("None / 空 -> 默认 paraformer", app._resolve_file_model(None) == PARA)
check("空字符串 -> 默认 paraformer", app._resolve_file_model("") == PARA)
check("未知值原样透传（假定是完整 ID）",
      app._resolve_file_model("iic/some_other_model") == "iic/some_other_model")

print("\n=== 2. DEFAULT_CONFIG 默认值 ===")
check("DEFAULT_CONFIG.funasr_model 是 paraformer",
      app.DEFAULT_CONFIG["funasr_model"] == PARA,
      app.DEFAULT_CONFIG["funasr_model"])

print("\n=== 3. load_file_model 默认参数已是 paraformer ===")
import inspect
sig = inspect.signature(app.FunASRTranscriber.load_file_model)
default_model = sig.parameters["model_name"].default
check("load_file_model 默认 model_name = paraformer",
      default_model == PARA, default_model)

print("\n=== 4. transcribe_file 接受 model_name 且解析后传给 load_model ===")
sig2 = inspect.signature(app.FunASRTranscriber.transcribe_file)
check("transcribe_file 有 model_name 参数",
      "model_name" in sig2.parameters, list(sig2.parameters))

# 用桩替换 load_model / file_model.generate，捕获传入的模型名
captured = {}


class StubResult:
    def __len__(self):
        return 1
    def __getitem__(self, i):
        return {"text": "<|zh|>你好<|NEUTRAL|>"}


t = app.FunASRTranscriber()


def fake_load_model(self, model_name=None, status_callback=None):
    captured["loaded"] = model_name
    return True


def fake_generate(self, **kwargs):
    captured["kwargs"] = kwargs
    return StubResult()

t.load_model = lambda *a, **k: fake_load_model(t, *a, **k)
t.file_model = type("M", (), {"generate": fake_generate})()

# 传短名 sensevoice 应被解析为完整 ID 并加载
text = t.transcribe_file("x.wav", model_name="sensevoice")
check("传短名 sensevoice -> 加载完整 SenseVoice ID",
      captured.get("loaded") == SENSE, captured.get("loaded"))
check("特殊 token 被清洗", text == "你好", repr(text))

# 传短名 paraformer-large
captured.clear()
text = t.transcribe_file("x.wav", model_name="paraformer-large")
check("传短名 paraformer-large -> 加载完整 paraformer ID",
      captured.get("loaded") == PARA, captured.get("loaded"))

# 不传 model_name 时走默认（paraformer）
captured.clear()
text = t.transcribe_file("x.wav")
check("不传 model_name -> 加载默认 paraformer",
      captured.get("loaded") == PARA, captured.get("loaded"))

print("\n=== 5. 后端转写任务读取 config.funasr_model（不再硬编码 SenseVoice）===")
src = io.open("app.py", encoding="utf-8").read()
# _transcribe_file_task 里应读取 config 的 funasr_model
check("_transcribe_file_task 读取 config.get('funasr_model')",
      "state.config.get(\"funasr_model\")" in src or
      "state.config.get('funasr_model')" in src)
# transcribe_file 调用处应把 model_name 传进去
check("transcribe_file(...) 传了 model_name=file_model",
      "model_name=file_model" in src)
# 启动预加载不再硬编码 SenseVoice
check("启动预加载不再硬编码 iic/SenseVoiceSmall",
      'load_file_model("iic/SenseVoiceSmall")' not in src)
# 但 diarization 仍用 paraformer-large（不受影响）
check("说话人分离仍用 paraformer-large-vad-punc",
      PARA in src)

print("\n=== 6. AST 校验：源码可解析 ===")
ast.parse(src)
check("app.py AST 解析通过", True)

print(f"\n通过 {passed} 条，失败 {failed} 条")
sys.exit(1 if failed else 0)
