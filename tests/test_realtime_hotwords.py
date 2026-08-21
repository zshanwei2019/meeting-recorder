"""实时热词纠正的回归测试（拼音级近音纠错）。

直接从 app.py 导入被测对象，避免测试与实现漂移。不发起任何真实网络请求，
不加载任何 ASR 模型。

运行：.venv\\Scripts\\python.exe tests\\test_realtime_hotwords.py
"""
import sys
import time
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


fix = app.correct_hotwords_by_pinyin

print("=== 依赖可用性 ===")
print("     pypinyin+rapidfuzz available =", app._POSTPROCESS_HOTWORDS_AVAILABLE)
if not app._POSTPROCESS_HOTWORDS_AVAILABLE:
    print("     依赖缺失，仅验证降级行为")
    check("降级时原文返回", fix("西工只财的回款", "西工智财"), "西工只财的回款")
    print("全部通过（降级路径）")
    sys.exit(0)

print()
print("=== 逐字拼音对齐 (_pinyin_syllables) ===")
syls = app._pinyin_syllables("abc中文")
check("非汉字不被合并，字与音节一一对应", len(syls), 5)
check("整段文本长度与音节数始终相等", len(app._pinyin_syllables("A1你好b")), 5)
check("单字拼音正确", app._char_pinyin("智"), "zhi")
check("缓存命中返回同值", app._char_pinyin("智"), "zhi")

print()
print("=== 近音纠正（应当纠） ===")
check("同音异字整词纠正", fix("西工只财的回款", "西工智财"), "西工智财的回款")
check("声母混淆 zh/z", fix("这个项目由西工知才负责", "西工智财"), "这个项目由西工智财负责")
check("多个热词同时纠正",
      fix("西工只财和南通分公司", "西工智财,南通"),
      "西工智财和南通分公司")
check("已经正确的词不被改动", fix("西工智财很好", "西工智财"), "西工智财很好")
check("英文热词精确匹配保留", fix("我们用ERP系统", "ERP"), "我们用ERP系统")

print()
print("=== 不该纠（防误伤，比能纠更重要） ===")
check("完全不相关文本不动", fix("今天天气很好", "西工智财"), "今天天气很好")
check("拼音差异大不动", fix("公司业务发展", "西工智财"), "公司业务发展")
check("空文本", fix("", "西工智财"), "")
check("空热词", fix("西工只财", ""), "西工只财")
check("None 热词", fix("西工只财", None), "西工只财")
check("单字热词被忽略（同音字太多，歧义大）", fix("他说的对", "队"), "他说的对")
check("不跨标点纠正", fix("西工。智财", "西工智财"), "西工。智财")
check("不跨空格纠正", fix("西工 智财", "西工智财"), "西工 智财")
check("不跨换行纠正", fix("西工\n智财", "西工智财"), "西工\n智财")

print()
print("=== 阈值行为（作用于拼音相似度，非字面）===")
# 注意："西工只财" 与 "西工智财" 拼音完全相同（均为 xigongzhicai），
# 属于同音异字，拼音相似度恒为 1.0 → 任何阈值下都应纠正。
check("同音异字：阈值 1.0 仍纠正（拼音完全一致）",
      fix("西工只财", "西工智财", threshold=1.0), "西工智财")
check("同音异字的拼音串确实相等",
      "".join(app._pinyin_syllables("西工只财")),
      "".join(app._pinyin_syllables("西工智财")))

# "系统直采" = xitongzhicai，与 xigongzhicai 近似但不等 → 阈值真正生效的场景
check("近音但不同音：宽阈值 0.85 下纠正",
      fix("系统直采报表", "西工智财", threshold=0.85), "西工智财报表")
check("近音但不同音：严阈值 0.95 下不纠",
      fix("系统直采报表", "西工智财", threshold=0.95), "系统直采报表")
check("近音但不同音：阈值 1.0 下不纠",
      fix("系统直采报表", "西工智财", threshold=1.0), "系统直采报表")

print()
print("=== 重叠与优先级 ===")
check("长热词优先于短热词",
      fix("西工只财报表", "西工智财,智财"), "西工智财报表")
check("纠正区间不重叠、无字符重复",
      fix("西工只财和西工只财", "西工智财"), "西工智财和西工智财")

print()
print("=== 长度容忍（ASR 多字/少字） ===")
check_true("ASR 少一字仍可纠",
           "西工智财" in fix("西工只才", "西工智财"))

print()
print("=== 幂等性 ===")
once = fix("西工只财的回款目标", "西工智财")
twice = fix(once, "西工智财")
check("纠正结果再纠一次不变（幂等）", twice, once)

print()
print("=== 性能（长会议不能卡住 UI） ===")
long_text = "西工只财今年的回款目标定在三千万元。" * 600   # 约 1 万字
t0 = time.time()
out = fix(long_text, "西工智财,南通,回款,ERP")
elapsed = time.time() - t0
print("     1 万字耗时 {:.3f}s".format(elapsed))
check_true("1 万字应在 2 秒内完成", elapsed < 2.0)
check("长文本全部命中被纠正", out.count("西工智财"), 600)
check("长文本未被截断（长度一致）", len(out), len(long_text))

print()
print("=== 实时路径接线检查 ===")
import inspect  # noqa: E402
src = inspect.getsource(app._realtime_transcribe_task)
check_true("实时任务读取 hot_words 配置", 'state.config.get("hot_words"' in src)
check_true("实时任务调用热词纠正", "correct_hotwords_by_pinyin" in src)
check_true("纠正在加标点之后（避免跨句误匹配）",
           src.index("add_punctuation") < src.index("correct_hotwords_by_pinyin"))
check("最终 flush 也做纠正（共两处调用）",
      src.count("correct_hotwords_by_pinyin("), 2)

print()
print("=== 旧代码 NameError 隐患已修复 ===")
# 旧实现把 _async_punctuate 定义在 `if audio_data is None:` 分支内部，
# 却在分支外调用；若首次循环即取到音频（不进该分支），直接 NameError。
# 锂点选得小心：不能用分支/循环字面量——注释里包含 `if audio_data is None:`，
# 而 `while state.is_realtime:` 在 xfyun 分支也出现过（确实在 def 之前）。
# 用只属于 FunASR 主循环的真实语句作锂点。
def_pos = src.index("def _async_punctuate")
loop_pos = src.index("audio_data = state.recorder.get_audio_chunk")
check("锂点在源码中唯一", src.count("audio_data = state.recorder.get_audio_chunk"), 1)
check_true("_async_punctuate 定义在 FunASR 主循环之前", def_pos < loop_pos)
check("全文仅定义一次（未重复定义）", src.count("def _async_punctuate"), 1)
check("两处调用均在定义之后",
      all(p > def_pos for p in
          [i for i in range(len(src))
           if src.startswith("target=_async_punctuate", i)]),
      True)

print()
if FAILED:
    print("失败 {} 项: {}".format(len(FAILED), ", ".join(FAILED)))
    sys.exit(1)
print("全部通过")
