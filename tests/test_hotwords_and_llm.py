"""热词解析 / LLM 请求体注入 的回归测试。

直接从 app.py 导入被测对象，避免测试与实现漂移。
不发起任何真实网络请求。

运行：.venv\\Scripts\\python.exe tests\\test_hotwords_and_llm.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402

FAILED = []


def check(name, got, expect):
    if got == expect:
        print("OK   {}".format(name))
    else:
        print("FAIL {}\n       got={!r}\n     expect={!r}".format(name, got, expect))
        FAILED.append(name)


print("=== 热词解析 (FunASRTranscriber._parse_hot_words) ===")
parse = app.FunASRTranscriber._parse_hot_words
check("空值返回空列表", parse(""), [])
check("None 返回空列表", parse(None), [])
check("单个热词", parse("西工智财"), ["西工智财"])
check("英文逗号分隔", parse("西工智财,南通,ERP"), ["西工智财", "南通", "ERP"])
check("中文逗号/顿号/分号/空格混用",
      parse("西工智财，南通、ERP； 回款"),
      ["西工智财", "南通", "ERP", "回款"])
check("换行分隔", parse("甲方\n乙方"), ["甲方", "乙方"])
check("去重且保序", parse("A,A,B"), ["A", "B"])
check("接受列表输入", parse(["X", " Y ", "X"]), ["X", "Y"])

print()
print("=== 热词参数构造 (build_hotword_kwargs) ===")
tr = app.FunASRTranscriber()
check("无热词时不注入任何参数", tr.build_hotword_kwargs(""), {})
kw = tr.build_hotword_kwargs("西工智财,南通")
check("hotword 为空格分隔字符串", kw.get("hotword"), "西工智财 南通")
if app._POSTPROCESS_HOTWORDS_AVAILABLE:
    check("已装 pypinyin -> 启用拼音级纠正", kw.get("postprocess_hotwords"), ["西工智财", "南通"])
    check("阈值默认 0.85", kw.get("postprocess_hotword_threshold"), 0.85)
else:
    check("未装 pypinyin -> 降级，不注入后处理参数",
          "postprocess_hotwords" in kw, False)
    print("     (提示：pypinyin 未安装，拼音级热词纠正已跳过)")

print()
print("=== LLM 请求体注入 ===")


def _inject_openai(body, messages, max_tokens):
    body["messages"] = messages
    body["max_tokens"] = max_tokens


def _inject_dashscope(body, messages, max_tokens):
    body.setdefault("input", {})["messages"] = messages
    body.setdefault("parameters", {})["max_tokens"] = max_tokens


msgs = [{"role": "user", "content": "PROMPT"}]

body = {"model": "deepseek-chat", "messages": [], "temperature": 0.3, "max_tokens": 8192}
_inject_openai(body, msgs, 4096)
check("OpenAI 兼容: messages 写在顶层", body["messages"], msgs)
check("OpenAI 兼容: max_tokens 被覆盖", body["max_tokens"], 4096)

body = {"model": "qwen-plus", "input": {"messages": []},
        "parameters": {"temperature": 0.3, "result_format": "message"}}
_inject_dashscope(body, msgs, 4096)
check("DashScope: messages 写在 input 下", body["input"]["messages"], msgs)
check("DashScope: max_tokens 写在 parameters 下", body["parameters"]["max_tokens"], 4096)
check("DashScope: 顶层不应出现 messages", "messages" in body, False)
print("     " + json.dumps(body, ensure_ascii=False))

print()
print("=== 旧实现缺陷复现（证明修复必要）===")
old = {"model": "qwen-turbo", "input": {"messages": []}, "parameters": {"temperature": 0.3}}
old["messages"] = msgs                 # 旧 call_llm：一律写顶层
if "max_tokens" in old:                # qwen body 无此键 -> 分支永不进入
    old["max_tokens"] = 4096
check("旧实现 input.messages 为空 -> 千问收不到 prompt", old["input"]["messages"], [])
check("旧实现 max_tokens 从未生效", "max_tokens" in old, False)

print()
if FAILED:
    print("失败 {} 项: {}".format(len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("全部通过")
