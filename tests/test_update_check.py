# -*- coding: utf-8 -*-
"""update_check（GitHub Releases 新版本检查）单测。全部注入假 http_get，不联网。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import update_check as uc  # noqa: E402

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
    def __init__(self, data, status_code=200):
        self._data = data
        self.status_code = status_code

    def json(self):
        return self._data


uc.reset_cache()

print("=== 1. 版本号规范化 ===")
check("v3.3.0 -> (3,3,0)", uc.normalize_version("v3.3.0") == (3, 3, 0))
check("3.3.0 去空格", uc.normalize_version("  3.3.0 ") == (3, 3, 0))
check("3.3 补全两位", uc.normalize_version("3.3") == (3, 3))
check("release-3.10.2 提取数字", uc.normalize_version("release-3.10.2") == (3, 10, 2))
check("非字符串返回空", uc.normalize_version(None) == ())
check("无数字返回空", uc.normalize_version("abc") == ())
check("四位版本", uc.normalize_version("1.2.3.4") == (1, 2, 3, 4))

print("\n=== 2. is_newer 比较 ===")
check("3.4.0 新于 3.3.0", uc.is_newer("v3.4.0", "3.3.0") is True)
check("3.3.0 不新于 3.3.0（相等）", uc.is_newer("3.3.0", "3.3.0") is False)
check("3.2.9 旧于 3.3.0", uc.is_newer("3.2.9", "3.3.0") is False)
check("4.0.0 新于 3.99.99", uc.is_newer("4.0.0", "3.99.99") is True)
check("3.3 视为 3.3.0，不新于 3.3.0", uc.is_newer("3.3", "3.3.0") is False)
check("数字段逐段比较（10 > 9）", uc.is_newer("3.10.0", "3.9.0") is True)
check("current 无法解析 -> False", uc.is_newer("3.4.0", "xyz") is False)
check("latest 无法解析 -> False", uc.is_newer("xyz", "3.3.0") is False)

print("\n=== 3. query_latest_release ===")
ok_resp = FakeResp({"tag_name": "v3.5.0",
                    "html_url": "https://github.com/x/y/releases/tag/v3.5.0",
                    "name": "3.5.0"})
got = uc.query_latest_release(http_get=lambda url, timeout: ok_resp)
check("正常返回三元组", got == ("v3.5.0",
      "https://github.com/x/y/releases/tag/v3.5.0", "3.5.0"), str(got))

err_resp = FakeResp({"message": "rate limit"}, status_code=403)
check("非 200 返回 None", uc.query_latest_release(http_get=lambda u, t: err_resp) is None)


def boom(url, timeout):
    raise RuntimeError("network down")


check("网络异常返回 None（不抛出）", uc.query_latest_release(http_get=boom) is None)

notdict = FakeResp(["unexpected"])
check("非 dict JSON 返回 None",
      uc.query_latest_release(http_get=lambda u, t: notdict) is None)

empty_tag = FakeResp({"tag_name": ""})
check("空 tag 返回 None", uc.query_latest_release(http_get=lambda u, t: empty_tag) is None)

print("\n=== 4. check_for_update 判定 ===")
uc.reset_cache()
r = uc.check_for_update("3.3.0", http_get=lambda u, t: FakeResp(
    {"tag_name": "v3.5.0", "html_url": "https://x/tag", "name": "n"}), force=True)
check("有新版本 -> True 且带 URL",
      r["update_available"] is True and r["latest_version"] == "v3.5.0"
      and r["release_url"] == "https://x/tag", str(r))

uc.reset_cache()
r = uc.check_for_update("3.5.0", http_get=lambda u, t: FakeResp(
    {"tag_name": "v3.5.0", "html_url": "https://x/tag", "name": "n"}), force=True)
check("版本相同 -> 无更新", r["update_available"] is False)

uc.reset_cache()
r = uc.check_for_update("3.3.0", http_get=lambda u, t: FakeResp(
    {"tag_name": "v3.2.0", "html_url": "https://x/tag", "name": "n"}), force=True)
check("远端更旧 -> 无更新（开发版）", r["update_available"] is False)

uc.reset_cache()
r = uc.check_for_update("3.3.0", http_get=boom, force=True)
check("检查失败静默 -> False 且不抛",
      r["update_available"] is False and r["current_version"] == "3.3.0")

print("\n=== 5. 进程内缓存（TTL 内不再请求网络）===")
uc.reset_cache()
calls = {"n": 0}


def counting_get(url, timeout):
    calls["n"] += 1
    return FakeResp({"tag_name": "v9.9.9", "html_url": "https://x", "name": "x"})


uc.check_for_update("1.0.0", http_get=counting_get, force=True)
uc.check_for_update("1.0.0", http_get=counting_get)  # 命中缓存
uc.check_for_update("1.0.0", http_get=counting_get)
check("TTL 内只请求一次", calls["n"] == 1, str(calls["n"]))
uc.reset_cache()
uc.check_for_update("1.0.0", http_get=counting_get)  # force 刷新
check("reset/force 后重新请求", calls["n"] == 2, str(calls["n"]))

uc.reset_cache()
print(f"\n通过 {passed} 条，失败 {failed} 条")
sys.exit(1 if failed else 0)
