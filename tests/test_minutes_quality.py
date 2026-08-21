"""纪要质量相关的回归测试：智能分段 / 说话人标注 / 领域化 prompt。

直接从 app.py 导入被测对象，避免测试与实现漂移。不发起任何真实网络请求。

运行：.venv\\Scripts\\python.exe tests\\test_minutes_quality.py
"""
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


def check_true(name, cond):
    check(name, bool(cond), True)


print("=== 长文本分段 (_split_text_for_llm) ===")
split = app._split_text_for_llm

check("空文本返回空列表", split("", 100), [])
check("短于 chunk_size 时不切分", split("这是一句话。", 100), ["这是一句话。"])

# 每句 10 字，共 30 句 = 300 字；chunk_size=100 → 应在句末切
sentences = ["第{:02d}句话内容啊。".format(i) for i in range(30)]
long_text = "".join(sentences)
chunks = split(long_text, 100)
check("无字符丢失（拼回等于原文）", "".join(chunks), long_text)
check_true("每段都不超过 chunk_size", all(len(c) <= 100 for c in chunks))
check_true("每段都以句末标点结尾（最后一段除外）",
           all(c[-1] in app._SENTENCE_END_PUNCTS for c in chunks[:-1]))

print()
print("=== 旧硬切实现的缺陷复现（证明修复必要）===")
CH = 100
old_chunks = [long_text[i:i + CH] for i in range(0, len(long_text), CH)]
check("旧实现：第一段结尾被劈在句子中间（非句末标点）",
      old_chunks[0][-1] in app._SENTENCE_END_PUNCTS, False)
check("新实现：第一段结尾落在句末标点",
      chunks[0][-1] in app._SENTENCE_END_PUNCTS, True)

print()
print("=== 无标点极端输入不得死循环 ===")
no_punct = "啊" * 250
np_chunks = split(no_punct, 100)
check("无标点时退化为硬切", [len(c) for c in np_chunks], [100, 100, 50])
check("无标点时内容无损", "".join(np_chunks), no_punct)

print()
print("=== 说话人标注 (_build_annotated_transcript) ===")
build = app._build_annotated_transcript
check("空 sentence_info 返回空串", build([]), "")
check("None 返回空串", build(None), "")

si = [
    {"text": "今年回款目标定在三千万。", "start": 0, "end": 3000, "spk": 0},
    {"text": "我认为偏高了。", "start": 3200, "end": 5000, "spk": 1},
    {"text": "那按两千八百万执行。", "start": 65000, "end": 68000, "spk": 0},
]
annotated = build(si)
print("     " + annotated.replace("\n", "\n     "))
check("说话人切换与长停顿共产生 3 段", len(annotated.splitlines()), 3)
check_true("含说话人1 标签", "说话人1]" in annotated)
check_true("含说话人2 标签", "说话人2]" in annotated)
check_true("含 MM:SS 时间戳", "[00:00 " in annotated)
check_true("超过一分钟用 01:05 形式", "[01:05 " in annotated)
check_true("原文内容保留", "三千万" in annotated and "两千八百万" in annotated)
check("max_chars 截断生效", len(build(si, max_chars=40).splitlines()), 1)

print()
print("=== 领域化提示 (_domain_guidance) ===")
g = app._domain_guidance
check_true("金融领域给出授信/风控等关注点", "风控" in g("金融"))
check_true("法院领域强调原告被告区分", "原告" in g("法院"))
check_true("医疗领域强调剂量不可推测", "剂量" in g("医疗"))
check_true("科技领域要求保留英文缩写", "API" in g("科技"))
check("未知领域退化为通用而不抛异常", g("玄学"), g("玄学"))
check_true("未知领域仍带原始领域名", "玄学" in g("玄学"))
check_true("通用领域也有关注点", "决策" in g("通用"))

print()
print("=== 纪要 prompt (_build_final_prompt) ===")
p_plain = app._build_final_prompt("正文", "金融")
check_true("prompt 内嵌领域化提示", "风控" in p_plain)
check_true("无说话人时不出现说话人段", "【说话人信息】" not in p_plain)

p_spk = app._build_final_prompt("正文", "金融", has_speakers=True)
check_true("有说话人时出现说话人段", "【说话人信息】" in p_spk)
check_true("要求负责人不得猜测", "不得根据猜测填写" in p_spk)
check_true("保留五段式结构", "五、风险与建议" in p_spk)

p_sum = app._build_final_prompt("摘要", "通用", is_summary=True)
check_true("汇总模式标签为各段摘要", "各段摘要" in p_sum)

print()
print("=== 签名兼容性 ===")
import inspect  # noqa: E402
sig = inspect.signature(app._generate_minutes_task)
check("sentence_info 为可选参数（旧调用不破）",
      sig.parameters["sentence_info"].default, None)

print()
if FAILED:
    print("失败 {} 项: {}".format(len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("全部通过")
