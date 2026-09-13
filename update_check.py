# -*- coding: utf-8 -*-
"""新版本检查：查询 GitHub Releases 并与当前版本比较。

设计原则：
- 纯逻辑可单测：版本比较 parse/normalize 与网络请求分离，网络层可注入 http_get。
- 失败静默：桌面单机、内网或限流时绝不能弹窗报错打扰用户，只返回 {"update_available": False}。
- 轻量：用 GitHub REST 的 latest release（单次请求），短超时；进程内缓存避免频繁请求。
- 只做“提示”，不自动下载/更新（sidecar 约 400MB，更新执行交给用户去 Releases 页）。
"""
from __future__ import annotations

import re
import time

# 仓库：用户自己的公开仓库
GITHUB_REPO = "zshanwei2019/meeting-recorder"
LATEST_RELEASE_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{GITHUB_REPO}/releases/latest"

REQUEST_TIMEOUT_S = 6
# 进程内缓存有效期：2 小时（一次开机通常只查一次，留余量）
CACHE_TTL_S = 2 * 3600

_cache: dict = {"at": 0.0, "result": None}


def normalize_version(v) -> tuple:
    """把 'v3.3.0' / '3.3.0' / '3.3' 解析成可比较的非负整数元组。

    取开头的数字点分段；无法解析返回 ()，视为“没有有效版本”，比较时不算更新。
    """
    if not isinstance(v, str):
        return ()
    m = re.search(r"(\d+(?:\.\d+){0,3})", v.strip())
    if not m:
        return ()
    try:
        return tuple(int(x) for x in m.group(1).split("."))
    except ValueError:
        return ()


def is_newer(latest, current) -> bool:
    """latest 是否严格新于 current（按数字段逐段比较，缺段补 0）。

    预发布标记（-rc1、-beta）不纳入数字比较；GitHub latest release 本身就排除
    prerelease，所以这里只需比较数字版本。
    """
    a = normalize_version(latest)
    b = normalize_version(current)
    if not a or not b:
        return False
    n = max(len(a), len(b))
    a = a + (0,) * (n - len(a))
    b = b + (0,) * (n - len(b))
    return a > b


def query_latest_release(http_get=None, *, timeout=REQUEST_TIMEOUT_S):
    """请求 GitHub latest release，返回 (tag_name, html_url, name)；失败返回 None。

    http_get 可注入（测试用）：签名 http_get(url, timeout) -> 响应对象，
    需有 .status_code 和 .json()。默认用 requests。
    """
    try:
        if http_get is None:
            import requests
            headers = {"Accept": "application/vnd.github+json",
                       "User-Agent": "meeting-recorder-update-check"}
            resp = requests.get(LATEST_RELEASE_URL, headers=headers, timeout=timeout)
        else:
            # 位置参数调用，注入替身只需签名 http_get(url, timeout)，不强制形参名
            resp = http_get(LATEST_RELEASE_URL, timeout)
        status = getattr(resp, "status_code", 200)
        if status != 200:
            return None
        data = resp.json()
        if not isinstance(data, dict):
            return None
        tag = str(data.get("tag_name") or "").strip()
        if not tag:
            return None
        url = str(data.get("html_url") or RELEASES_PAGE)
        name = str(data.get("name") or tag)
        return tag, url, name
    except Exception:
        # 断网/限流/JSON 异常：更新检查永远不应影响主程序
        return None


def check_for_update(current_version, *, http_get=None, force=False):
    """检查是否有新版本。返回 dict：
    {update_available, latest_version, current_version, release_url, release_name}
    任何失败都返回 update_available=False。带进程内缓存。
    """
    now = time.time()
    if not force and _cache["result"] is not None and (now - _cache["at"]) < CACHE_TTL_S:
        return dict(_cache["result"])

    result = {
        "update_available": False,
        "latest_version": "",
        "current_version": current_version or "",
        "release_url": RELEASES_PAGE,
        "release_name": "",
    }
    info = query_latest_release(http_get)
    if info:
        tag, url, name = info
        result["latest_version"] = tag
        result["release_url"] = url
        result["release_name"] = name
        if current_version and is_newer(tag, current_version):
            result["update_available"] = True

    _cache["at"] = now
    _cache["result"] = result
    return dict(result)


def reset_cache():
    """测试辅助：清空进程内缓存。"""
    _cache["at"] = 0.0
    _cache["result"] = None
