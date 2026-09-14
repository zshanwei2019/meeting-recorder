# -*- coding: utf-8 -*-
"""AI 纪要 LLM 错误人性化映射回归测试。

背景：点“AI 纪要”时若服务商报错（实测 DeepSeek 余额不足返回 402 Insufficient
Balance），旧版只把错误写进默认收起的日志面板，界面毫无反应。修复后后端发
minutes_error 弹红条，且把 402/401/429/网络错误翻译成用户能看懂的话。
本测试只验证纯函数 _friendly_llm_error 的映射，不起服务/不联网/不加载 ASR。

运行：.venv\\Scripts\\python.exe tests\\test_llm_error_feedback.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app  # noqa: E402

passed = 0
failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        print(f"  OK   {name}")
        passed += 1
    else:
        failed += 1
        print(f"  FAIL {name} {detail}")


f = app._friendly_llm_error

print("=== 402 余额不足 ===")
m = f("API返回402: Insufficient Balance")
check("含 402", "402" in m, m)
check("提示余额/额度", ("余额" in m or "额度" in m), m)
check("提示充值/换 Key", ("充值" in m or "Key" in m), m)

print("\n=== 401/403 Key 无效 ===")
m = f("API返回401: Authentication Fails, invalid api key")
check("提示 Key 无效/无权限", ("Key" in m and ("无效" in m or "无权限" in m)), m)
m403 = f("API返回403: forbidden")
check("403 也归入鉴权问题", ("Key" in m403), m403)

print("\n=== 429 限流 ===")
m = f("API返回429: Rate Limit Reached, too many requests")
check("提示限流/稍后重试", ("限流" in m and "稍后" in m), m)

print("\n=== 网络/超时 ===")
m = f("HTTPSConnectionPool: Max retries exceeded; connection timed out")
check("提示网络/超时", ("网络" in m or "超时" in m), m)

print("\n=== 未知错误兜底 ===")
m = f("some weird internal error XYZ")
check("保留原始信息", "some weird internal error XYZ" in m, m)
m2 = f("   ")
check("空错误给兜底文案", "未知错误" in m2, m2)

print(f"\n通过 {passed} 条，失败 {failed} 条")
sys.exit(1 if failed else 0)
