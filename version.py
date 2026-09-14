#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""版本号单一事实来源。

取版本信息的优先级：
1. 打包时由 ``tools/gen_version.py`` 生成的 ``_build_info`` 模块——它是一个
   可 ``import`` 的 .py，PyInstaller 会自动跟随 import 把它打进 exe，
   不依赖任何 .spec 的 datas 配置，因此任一套打包流程都能带上。
2. frozen 环境下随数据文件捆绑的 ``_version.py``（PyInstaller 解压目录里）。
3. 开发运行时直接调用 ``git`` 读当前工作区的 commit 与日期。
4. 都拿不到时回退到 BASE_VERSION + "dev"。

UI 和后端都从这里取版本，不要再在各处硬编码 "3.1.0"。
"""
from __future__ import annotations

import subprocess
from pathlib import Path

# ─── 基础版本号（发版时手动 bump；仅此一处） ───
BASE_VERSION = "3.3.2"

_THIS_DIR = Path(__file__).resolve().parent


def _read_build_module() -> dict:
    """读取打包时生成、可被 import 的 _build_info 模块。"""
    info: dict = {}
    try:
        import _build_info  # type: ignore
        info["commit"] = getattr(_build_info, "COMMIT", None)
        info["date"] = getattr(_build_info, "COMMIT_DATE", None)
        info["build_date"] = getattr(_build_info, "BUILD_DATE", None)
        if getattr(_build_info, "DIRTY", False):
            info["dirty"] = True
    except Exception:
        pass
    return {k: v for k, v in info.items() if v}


def _read_bundled_file() -> dict:
    """frozen 环境下读随数据文件捆绑的 _version.py（兜底）。"""
    info: dict = {}
    candidates = [_THIS_DIR / "_version.py"]
    try:
        import sys as _sys
        if getattr(_sys, "frozen", False):
            meipass = getattr(_sys, "_MEIPASS", None)
            if meipass:
                candidates.insert(0, Path(meipass) / "_version.py")
    except Exception:
        pass
    ns: dict = {}
    for c in candidates:
        try:
            if c.is_file():
                exec(compile(c.read_text(encoding="utf-8"), str(c), "exec"), ns)
                info["commit"] = ns.get("COMMIT")
                info["date"] = ns.get("COMMIT_DATE")
                info["build_date"] = ns.get("BUILD_DATE")
                if ns.get("DIRTY"):
                    info["dirty"] = True
                break
        except Exception:
            continue
    return {k: v for k, v in info.items() if v}


def _read_git() -> dict:
    """开发环境：从 git 读取当前 commit 短哈希与提交日期。"""
    info: dict = {}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(_THIS_DIR), capture_output=True, text=True,
            timeout=5,
        )
        if commit.returncode == 0:
            info["commit"] = commit.stdout.strip()
        date = subprocess.run(
            ["git", "log", "-1", "--format=%cd", "--date=short"],
            cwd=str(_THIS_DIR), capture_output=True, text=True,
            timeout=5,
        )
        if date.returncode == 0:
            info["date"] = date.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(_THIS_DIR), capture_output=True, text=True,
            timeout=5,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            info["dirty"] = True
    except Exception:
        pass
    return info


def _build_info() -> dict:
    info = _read_build_module()
    if not info.get("commit"):
        info.update({k: v for k, v in _read_bundled_file().items() if k not in info})
    if not info.get("commit"):
        info.update({k: v for k, v in _read_git().items() if k not in info})
    return info


_info = _build_info()

#: 基础语义化版本，例如 "3.1.0"
APP_VERSION = BASE_VERSION
#: git 短哈希，拿不到为 None
VERSION_COMMIT = _info.get("commit")
#: 提交日期 YYYY-MM-DD（或构建日期）
VERSION_DATE = _info.get("date") or _info.get("build_date")
#: 工作区/构建是否含未提交改动
VERSION_DIRTY = bool(_info.get("dirty"))


def version_string() -> str:
    """给人看的完整版本串，例如 ``3.1.0 (a0e8d9e, 2026-08-28)``。"""
    parts = [BASE_VERSION]
    tail = []
    if VERSION_COMMIT:
        tail.append(VERSION_COMMIT)
    if VERSION_DATE:
        tail.append(VERSION_DATE)
    if tail:
        parts.append("(" + ", ".join(tail) + ")")
    elif not VERSION_COMMIT:
        parts.append("(dev)")
    if VERSION_DIRTY:
        parts.append("[dirty]")
    return " ".join(parts)


def version_dict() -> dict:
    """供 /api/version 返回。"""
    return {
        "version": APP_VERSION,
        "commit": VERSION_COMMIT,
        "date": VERSION_DATE,
        "dirty": VERSION_DIRTY,
        "display": version_string(),
    }


if __name__ == "__main__":
    print(version_string())
