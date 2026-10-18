# -*- coding: utf-8 -*-
"""实时转写音源必须跟随前端当前选择（回归 PR #49 后发现的漏保存 bug）。

背景：音源下拉框漏了 onchange=saveConfig()，改了不写配置；而实时转写
启动时只认已保存配置、不读下拉当前值 → 界面选「系统+麦克风」，实际仍按
纯系统声采集，系统没播放就是纯静音。

本测试走真实 create_app() 注册的 /ws endpoint + 受控假 ws，把 threading.Thread
换成只记录参数、不起线程的替身，断言 start_realtime 把前端 source 透传给了
_realtime_transcribe_task（完全确定性：不开设备/线程/模型/网络）。
另对 ui/index.html 做静态断言：下拉改动即保存、实时启动带 source。

运行：.venv\\Scripts\\python.exe tests\\test_realtime_source_passthrough.py
"""
import asyncio
import io
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app  # noqa: E402
import threading  # noqa: E402
from fastapi import WebSocketDisconnect  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    if cond:
        PASS.append(name); print("  OK  ", name)
    else:
        FAIL.append(name); print("  FAIL", name, detail)


class FakeWS:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []

    async def accept(self):
        pass

    async def receive_json(self):
        if self.messages:
            return self.messages.pop(0)
        raise WebSocketDisconnect(code=1000)

    async def send_json(self, data):
        self.sent.append(data)

    async def send_text(self, data):
        self.sent.append(data)


class CapturedThread:
    """替换 threading.Thread：只记录构造参数，start() 不做任何事。"""
    last = None
    instances = []

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self.target = target
        self.args = args
        self.kwargs = kwargs or {}
        self.daemon = daemon
        CapturedThread.last = self
        CapturedThread.instances.append(self)

    def start(self):
        pass

    def is_alive(self):
        return False


def get_ws_endpoint():
    fastapi_app = app.create_app()
    for r in fastapi_app.routes:
        if getattr(r, "path", None) == "/ws":
            return r.endpoint
    raise AssertionError("/ws 路由未找到")


def drive_start_realtime(payload_source, saved_source):
    """返回捕获到的后台线程 (target, args)。"""
    endpoint = get_ws_endpoint()
    st = app.state
    orig_thread = threading.Thread
    orig_rec_thread = st._realtime_thread
    orig_rt = st.is_realtime
    orig_saved = st.config.get("audio_source")
    CapturedThread.instances = []
    CapturedThread.last = None
    st.config["audio_source"] = saved_source
    st.is_realtime = False
    st._realtime_thread = None
    data = {"source": payload_source} if payload_source is not None else {}
    msg = {"action": "start_realtime", "data": data}
    fws = FakeWS([msg])
    real_thread = threading.Thread
    try:
        # app.py 以 `threading.Thread(...)` 方式引用，需替换模块属性
        app.threading.Thread = CapturedThread
        asyncio.run(endpoint(fws))
    finally:
        app.threading.Thread = real_thread
        st._realtime_thread = orig_rec_thread
        st.is_realtime = orig_rt
        if orig_saved is None:
            st.config.pop("audio_source", None)
        else:
            st.config["audio_source"] = orig_saved
    cap = CapturedThread.last
    assert cap is not None, "start_realtime 没有启动后台线程"
    return cap.target, cap.args


# 1) 已保存配置是 system（过期），前端当前选 both → 必须透传 both
target, args = drive_start_realtime("both", saved_source="system")
check("后台任务函数是 _realtime_transcribe_task", target is app._realtime_transcribe_task, target)
check("透传 engine=FunASR", len(args) >= 2 and args[1] == "FunASR", args)
check("透传 source=both（前端当前选择，不被过期配置覆盖）", len(args) >= 3 and args[2] == "both", args)

# 2) 前端选 mic → 透传 mic
target, args = drive_start_realtime("mic", saved_source="system")
check("前端选 mic → 透传 mic", args[2] == "mic", args)

# 3) 前端没带 source（旧客户端）→ 回退已保存配置
target, args = drive_start_realtime(None, saved_source="system")
check("未带 source → 回退已保存配置 system", args[2] == "system", args)

# 4) 静态断言：音源下拉框改动即保存
html = (ROOT / "ui" / "index.html").read_text(encoding="utf-8")
m = re.search(r'<select[^>]*id="audioSource"[^>]*>', html)
check("音源下拉框存在", m is not None)
if m:
    check("音源下拉框绑定 onchange=saveConfig()",
          'onchange="saveConfig()"' in m.group(0), m.group(0))

# 5) 静态断言：实时转写启动带当前音源
check("toggleRealtime 发送 start_realtime 时带 source",
      re.search(r"sendAction\(\s*'start_realtime'\s*,\s*\{\s*source:\s*rtSource\s*\}\s*\)",
                html, re.S) is not None,
      "未找到带 source 的 start_realtime 发送")
check("rt_source 取自 audioSource.value",
      "getElementById('audioSource').value" in html)

print(f"\n通过 {len(PASS)} 条，失败 {len(FAIL)} 条")
sys.exit(1 if FAIL else 0)
