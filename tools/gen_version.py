#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""打包前生成 _build_info.py，把 git commit / 日期固化进构建产物。

用法（PyInstaller 打包前调用，spec 已自动调用）：
    python tools/gen_version.py

会在仓库根目录写出 _build_info.py。它是一个普通可 import 的 Python 模块，
version.py 里 `import _build_info` 会让 PyInstaller 自动把它打进 exe
（无需在 .spec 的 datas 里登记），因此任一套打包流程都能带上版本信息。

同时生成 _version.py 作为 frozen 数据文件兜底（spec datas 里已登记）。
两个文件都已加入 .gitignore（构建产物，不入库）。

另外：把基础版本号（version.py 的 BASE_VERSION，单一事实来源）同步写进
Tauri 壳的 src-tauri/tauri.conf.json 的 "version" 字段。Windows 下壳 exe 的
文件版本资源与 NSIS 产品版本都由它驱动；in-app 的 commit/日期版本仍由
Python /api/version 提供，不塞进 Windows 数字版本。
"""
from __future__ import annotations

import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str:
    try:
        r = subprocess.run(
            ["git", *args], cwd=str(ROOT),
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else ""
    except Exception:
        return ""


def _base_version() -> str:
    """从 version.py 读取 BASE_VERSION（版本号单一来源）。"""
    try:
        text = (ROOT / "version.py").read_text(encoding="utf-8")
        m = re.search(r'^BASE_VERSION\s*=\s*["\']([^"\']+)["\']', text, re.M)
        if m:
            return m.group(1)
    except Exception:
        pass
    return "0.0.0"


def _sync_tauri_version(base: str) -> None:
    """把基础版本号写进 src-tauri/tauri.conf.json 的 version 字段。

    用 JSON 解析后回写，保留其它配置；文件不存在（未重建壳）时静默跳过。
    """
    conf = ROOT / "src-tauri" / "tauri.conf.json"
    if not conf.is_file():
        return
    try:
        data = json.loads(conf.read_text(encoding="utf-8-sig"))
        if data.get("version") == base:
            return
        data["version"] = base
        conf.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[gen_version] synced tauri.conf.json version -> {base}")
    except Exception as e:
        print(f"[gen_version] WARN: failed to sync tauri version: {e}")


def main() -> int:
    commit = _git("rev-parse", "--short", "HEAD") or "unknown"
    commit_date = _git("log", "-1", "--format=%cd", "--date=short")
    dirty = bool(_git("status", "--porcelain"))
    build_date = datetime.date.today().isoformat()
    base_version = _base_version()

    header = "# 由 tools/gen_version.py 在打包时自动生成，请勿手改、勿入库。\n"
    body = (
        f'COMMIT = {commit!r}\n'
        f'COMMIT_DATE = {commit_date!r}\n'
        f'BUILD_DATE = {build_date!r}\n'
        f'DIRTY = {dirty!r}\n'
    )

    (ROOT / "_build_info.py").write_text(header + body, encoding="utf-8")
    (ROOT / "_version.py").write_text(header + body, encoding="utf-8")
    # 同步 Tauri 壳版本（Windows exe 文件版本 / NSIS 产品版本由此驱动）。
    _sync_tauri_version(base_version)
    print(f"[gen_version] commit={commit} date={commit_date or build_date} dirty={dirty}")
    print("[gen_version] wrote _build_info.py / _version.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
