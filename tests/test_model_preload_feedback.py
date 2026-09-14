# -*- coding: utf-8 -*-
"""启动模型预加载（首次约 2.9GB 下载）状态与错误反馈回归测试。

背景：旧版启动后台预加载线程不传状态回调，首次下载失败只 print 到打包后不存在
的控制台（GUI 无控制台），前端毫无感知——README 明确写着“若下载失败不会有任何
弹窗或报错”。修复后：AppState 持久化 model_preload 状态，回调实时推 model_preload
事件，WS 连接时补发 loading/error；错误经 _friendly_model_error 翻译成人话。

纯函数/内存状态测试，不起服务、不联网、不加载 ASR、不碰音频。
运行：.venv\\Scripts\\python.exe tests\\test_model_preload_feedback.py
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


f = app._friendly_model_error

print("=== _friendly_model_error 错误翻译 ===")
m = f("HTTPSConnectionPool: Read timed out. (read timeout=60)")
check("超时 → 提示下载超时", "超时" in m, m)

m = f("Max retries exceeded with url: /xxx (Caused by NewConnectionError)")
check("连接失败 → 提示无法连接/检查网络", ("无法连接" in m and "网络" in m), m)

m = f("OSError: [Errno 28] No space left on device")
check("磁盘满 → 提示空间不足", "磁盘空间" in m, m)

m = f("PermissionError: [Errno 13] Access is denied")
check("权限 → 提示写入权限", "权限" in m, m)

m = f("some unpredicted English error XYZ")
check("其它英文 → 加中文前缀并保留原文",
      m.startswith("语音模型下载/加载失败") and "XYZ" in m, m)

m = f("文件转写模型加载失败: HTTPError 500")
check("已含中文 → 幂等放行，不重复加前缀",
      m == "文件转写模型加载失败: HTTPError 500", m)

m = f("   ")
check("空错误 → 兜底文案", "重启程序重试" in m, m)

print("\n=== AppState 预加载状态机 ===")
st = app.state
_saved = dict(st.model_preload)
try:
    # 回到干净起点
    st.model_preload.update({"status": "idle", "message": "", "started": None, "finished": None})

    check("初始 idle", st.preload_event().get("status") == "idle")

    # loading：记录起点、状态正确（无 ws 时 push_from_thread 仅告警，不抛异常）
    st.preload_callback("loading", "正在下载，请保持网络畅通…")
    ev = st.preload_event()
    check("loading 状态", ev.get("status") == "loading")
    check("loading 消息", ev.get("message") == "正在下载，请保持网络畅通…")
    check("loading 记录 started", ev.get("started") is not None)

    # error（英文原始异常）→ 翻译成人话
    st.preload_callback("error", "Read timed out.")
    ev = st.preload_event()
    check("error 状态", ev.get("status") == "error")
    check("error 消息已翻译（超时）", "超时" in ev.get("message", ""), ev)
    check("error 记录 finished", ev.get("finished") is not None)

    # ready
    st.preload_callback("ready", "语音模型就绪")
    check("ready 状态", st.preload_event().get("status") == "ready")

    # 未知 status 不改状态
    before = st.preload_event().get("status")
    st.preload_callback("something_weird", "x")
    check("未知 status 被忽略", st.preload_event().get("status") == before)

    # preload_event 返回的是快照副本，外部改不影响内部
    snap = st.preload_event()
    snap["status"] = "tampered"
    check("preload_event 返回副本", st.preload_event().get("status") != "tampered")
finally:
    st.model_preload.clear()
    st.model_preload.update(_saved)

print(f"\n通过 {passed} 条，失败 {failed} 条")
sys.exit(1 if failed else 0)
