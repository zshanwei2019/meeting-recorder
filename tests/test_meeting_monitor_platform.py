# -*- coding: utf-8 -*-
"""MeetingMonitor 会议进程检测的平台分支（macOS/Linux 关键字 vs Windows 精确名）。

CI 精简环境可跑：匹配逻辑是纯函数 + sys.platform 打桩；真实 ps 枚举仅本机冒烟。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app   # noqa: E402

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        print(f"  OK   {name}")
        passed += 1
    else:
        print(f"  FAIL {name} {detail}")
        failed += 1


monitor = app.MeetingMonitor()
orig_platform = sys.platform


def with_platform(p):
    sys.platform = p


def restore_platform():
    sys.platform = orig_platform


print("=== 1. macOS 关键字子串匹配（comm 全路径）===")
try:
    with_platform("darwin")
    check("WeMeet.app 全路径命中腾讯会议",
          monitor._match_meeting_process(
              {"/applications/wemeet.app/Contents/macos/wemeet"}) == "腾讯会议")
    check("中文目录 .app 也命中（子串不依赖可执行名）",
          monitor._match_meeting_process(
              {"/applications/腾讯会议.app/contents/macos/wemeetapp"}) == "腾讯会议")
    check("VooV 命中",
          monitor._match_meeting_process(
              {"/applications/voovmeeting.app/contents/macos/voovmeeting"}) == "腾讯会议VooV")
    check("zoom.us 命中 Zoom",
          monitor._match_meeting_process(
              {"/applications/zoom.us.app/contents/macos/zoom.us"}) == "Zoom")
    check("无会议进程 -> None",
          monitor._match_meeting_process(
              {"/usr/sbin/cupsd", "/applications/safari.app/contents/macos/safari"}) is None)
    check("空集合 -> None", monitor._match_meeting_process(set()) is None)
    check("None 输入 -> None", monitor._match_meeting_process(None) is None)

    print("\n=== 2. Windows 精确名匹配（打桩 sys.platform）===")
    with_platform("win32")
    check("wemeet.exe 精确命中",
          monitor._match_meeting_process({"wemeet.exe", "svchost.exe"}) == "腾讯会议")
    check("大小写不敏感（集合已小写化，键也 lower）",
          monitor._match_meeting_process({"lark.exe"}) == "飞书")
    check("全路径不匹配（Windows 语义：只认精确文件名）",
          monitor._match_meeting_process({"c:\\program files\\wemeet\\wemeet.exe"}) is None)
    check("无会议进程 -> None",
          monitor._match_meeting_process({"svchost.exe"}) is None)
finally:
    restore_platform()

print("\n=== 3. 本机真实枚举冒烟（darwin/linux 才有 ps 分支）===")
if orig_platform == "darwin" or orig_platform.startswith("linux"):
    running = monitor._list_running_processes()
    check("ps 枚举返回非空集合", isinstance(running, set) and len(running) > 0,
          type(running))
    check("集合全为小写", running == {p.lower() for p in running})
    # 本机当前不应误判有会议（若用户真在开会议则此条人工忽略）
    print(f"  (info) 当前检测到会议软件: {monitor._match_meeting_process(running)}")
else:
    print("  (skip: Windows CI —— tasklist 分支由 Windows 侧覆盖)")

print(f"\n通过 {passed} 条，失败 {failed} 条")
sys.exit(1 if failed else 0)
