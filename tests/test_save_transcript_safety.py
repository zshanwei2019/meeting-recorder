"""save_transcript 路径与格式加固的回归测试（直接 import app.py）。

覆盖 probe 实测确认的三处问题：
  1. filename 裸拼进路径 → ../../../evil 真的写到 TRANSCRIPTS_DIR 外
  2. fmt 无白名单 → 可落盘 .bat / .ps1 / .html 等可执行或可双击打开的文件
  3. txt 分支缺 try/except → 异常冒泡到 websocket_endpoint 兜底，UI 直接失联

不加载 ASR 模型、不打开音频设备、不发网络请求。
运行：.venv\\Scripts\\python.exe tests\\test_save_transcript_safety.py
"""
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app  # noqa: E402

PASS = []
FAIL = []


def check(name, got, expect):
    if got == expect:
        PASS.append(name)
        print(f"  OK   {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL {name}\n         got={got!r}\n         expect={expect!r}")


def check_true(name, cond, hint=""):
    check(name + (f" ({hint})" if hint else ""), bool(cond), True)


BASE = app.TRANSCRIPTS_DIR

print("=== 1. 路径穿越必须被挡住（旧 bug 复现）===")
# 旧写法： TRANSCRIPTS_DIR / f"{filename}.{fmt}"
# probe 实测 "../../../evil" 会 resolve 到 C:\Users\evil.txt —— 逃出目录
old_escape = (BASE / "../../../evil.txt").resolve()
check_true("旧写法确实逃出目录（证明修复必要）",
           old_escape.parent != BASE.resolve(), f"旧结果={old_escape}")

ATTACKS = [
    "../../../evil",
    "..\\..\\..\\evil",
    "../" * 8 + "AppData/Roaming/Microsoft/Windows/Start Menu/Programs/Startup/x",
    "C:/Windows/Temp/evil",
    "C:evil",
    "/etc/passwd",
    "sub/dir/deep",
    "....//....//evil",
]
for atk in ATTACKS:
    p = app._resolve_output_path(BASE, atk, "txt")
    inside = p.parent == BASE.resolve()
    check_true(f"挡住 {atk[:34]}", inside, f"-> {p.name}")

print()
print("=== 2. 文件名净化细节 ===")
check("目录成分被剥离", app._sanitize_filename("a/b/c"), "c")
check("反斜杠同样处理", app._sanitize_filename("a\\b\\c"), "c")
check("盘符被剥离", app._sanitize_filename("C:evil"), "evil")
check("纯 .. 回退到默认名", app._sanitize_filename(".."), "transcript")
check("单点回退", app._sanitize_filename("."), "transcript")
check("空串回退", app._sanitize_filename(""), "transcript")
check("None 回退", app._sanitize_filename(None), "transcript")
check("前导点去除（不产生隐藏文件）", app._sanitize_filename(".hidden"), "hidden")
check("中文保留", app._sanitize_filename("会议记录_0821"), "会议记录_0821")
check("连字符与点保留", app._sanitize_filename("2026-08-21.v2"), "2026-08-21.v2")
check("非法字符替换为下划线", app._sanitize_filename('a<>:|?*b'), "a______b")
# 旧写法用 split(":")[-1] 剔盘符，会把所有冒号都当盘符，
# "10:30会议" 静默丢成 "30会议"——安全上无害，但属数据丢失。
check("时间冒号不丢失数据", app._sanitize_filename("10:30会议"), "10_30会议")
# 边界：开头“单字母 + 冒号”符合 Windows 盘符语义（A: 是合法盘符），
# 故 "a:b:c" 按“A 盘下的相对路径”处理。且 Windows 本就无法创建含
# 冒号的文件名（实测 FileNotFoundError），剔掉不损失可用性。
check("开头单字母+冒号按盘符剔除", app._sanitize_filename("a:b:c"), "b_c")
check("两字母开头不算盘符", app._sanitize_filename("ab:c"), "ab_c")
check("仅开头盘符被剔", app._sanitize_filename("C:evil"), "evil")
check("非盘符单字母冒号不当盘符",
      app._sanitize_filename("第一节:总结"), "第一节_总结")
check("全角冒号正常替换", app._sanitize_filename("会议：讨论"), "会议_讨论")
check_true("超长名被截断", len(app._sanitize_filename("x" * 400)) <= 120)
# Windows 保留设备名
check("CON 被前缀化", app._sanitize_filename("CON"), "_CON")
check("con 小写同样处理", app._sanitize_filename("con"), "_con")
check("COM1 被前缀化", app._sanitize_filename("COM1"), "_COM1")
check("con.txt 也算保留名", app._sanitize_filename("con.txt"), "_con.txt")
check_true("正常名不被误伤", app._sanitize_filename("transcript_2026") == "transcript_2026")

print()
print("=== 3. 格式白名单 ===")
check("白名单与前端下拉一致",
      sorted(app.ALLOWED_SAVE_FORMATS), ["docx", "json", "srt", "txt", "vtt"])
for bad in ("bat", "ps1", "html", "py", "exe", "js", "cmd", "vbs"):
    check_true(f"拒绝 .{bad}", bad not in app.ALLOWED_SAVE_FORMATS)

print()
print("=== 4. 扩展名不可被 filename 二次注入 ===")
# 即便 filename 里带别的后缀，最终扩展名仍由白名单里的 fmt 决定
p = app._resolve_output_path(BASE, "evil.bat", "txt")
check("filename 带 .bat 时仍以 .txt 结尾", p.suffix, ".txt")
check_true("路径仍在目录内", p.parent == BASE.resolve())

print()
print("=== 5. 静态断言：防回归 ===")
src = io.open(ROOT / "app.py", encoding="utf-8").read()
check("不再裸拼 fmt 路径",
      src.count('TRANSCRIPTS_DIR / f"{filename}.{fmt}"'), 0)
check("不再裸拼 docx 路径",
      src.count('TRANSCRIPTS_DIR / f"{filename}.docx"'), 0)
check_true("改用 _resolve_output_path", "_resolve_output_path(TRANSCRIPTS_DIR" in src)
check_true("有格式白名单判断", "if fmt not in ALLOWED_SAVE_FORMATS:" in src)
# txt 分支的 write_text 必须在 try 内
i_w = src.index('filepath.write_text(text, encoding="utf-8")')
before = src[:i_w]
i_try = before.rindex("try:")
i_else = before.rindex("else:")
check_true("write_text 位于 try 之后", i_try > i_else)
check_true("write_text 后有 except 兜底",
           "except Exception as e:" in src[i_w:i_w + 400])

print()
print(f"通过 {len(PASS)} 条，失败 {len(FAIL)} 条")
if FAIL:
    for f in FAIL:
        print(f"  - {f}")
sys.exit(1 if FAIL else 0)
