"""stop_recording 状态失同步 + import 冗余的回归测试（直接 import app.py）。

覆盖 probe 实测确认的问题：
  1. `state.is_recording = False` 原先位于 `recorder.stop()` 之后的 try 体内，
     stop() 一抛异常就全部跳过，而 except 分支也不复位
     → is_recording 永久卡 True，前端按钮死在「录音中」，只能重启进程
  2. app.py 模块级 `import re` 重复两次，且 _save_transcript_docx 内
     另有一处被模块级覆盖的冗余局部 import

驱动方式：拿 create_app() 注册的真实 /ws endpoint，喂一个受控假 ws，
用 asyncio.run 跑 —— 走真实 handle_message，不复制业务逻辑。
完全确定性：不起 server、不开线程、不联网、不加载 ASR、不碰音频设备。

（注：最初用 fastapi TestClient，但 receive_json 在无事件时会永久阻塞 →
 改成假 ws，由 WebSocketDisconnect 精确控制收尾。）

运行：.venv\\Scripts\\python.exe tests\\test_stop_recording_state.py
"""
import ast
import asyncio
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app  # noqa: E402
# app.py 在 create_app() 内部才 import fastapi，故模块级取不到，直接从 fastapi 拿
from fastapi import WebSocketDisconnect  # noqa: E402

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


SRC = io.open(ROOT / "app.py", encoding="utf-8").read()
LINES = SRC.splitlines()
TREE = ast.parse(SRC)


# ─── 测试替身：不碰任何音频设备 ───

class RaisingRecorder:
    """stop() 抛异常 —— 复现「设备已失效」的真实场景。

    probe 实测 AudioRecorder.stop() 里 _stream.stop()/close() 不在 try 内，
    sounddevice 对已拔出/被占用的设备可能抛异常，故这不是臆造场景。
    """

    def __init__(self, exc=None):
        self.exc = exc or RuntimeError("PortAudio error: device unavailable")
        self.stop_called = 0

    def stop(self):
        self.stop_called += 1
        raise self.exc


class OkRecorder:
    """stop() 正常返回；默认返回 None（无音频数据）以免触发自动转写。"""

    def __init__(self, ret=None):
        self.ret = ret
        self.stop_called = 0

    def stop(self):
        self.stop_called += 1
        return self.ret


class FakeWS:
    """受控 WebSocket：消息发完即抛 WebSocketDisconnect，绝不阻塞。"""

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


def get_ws_endpoint():
    """从真实注册的路由里取 /ws 的 endpoint（不猜函数名）。"""
    fastapi_app = app.create_app()
    for r in fastapi_app.routes:
        if getattr(r, "path", None) == "/ws":
            return r.endpoint
    raise AssertionError("/ws 路由未找到")


def drive_stop_recording(recorder, pre_is_recording=True):
    """用真实 endpoint + 真实 handle_message 驱动 stop_recording 分支。"""
    endpoint = get_ws_endpoint()
    st = app.state
    orig_rec = st.recorder
    orig_auto = st.config.get("auto_transcribe", True)
    orig_ws = st.websocket
    # 关掉自动转写：否则 stop() 返回路径时会起线程加载 ASR 模型
    st.config["auto_transcribe"] = False
    st.recorder = recorder
    st.is_recording = pre_is_recording
    st.recording_start = 12345.0
    fws = FakeWS([{"action": "stop_recording"}])
    try:
        asyncio.run(endpoint(fws))
        return st.is_recording, st.recording_start, fws.sent
    finally:
        st.recorder = orig_rec
        st.config["auto_transcribe"] = orig_auto
        st.websocket = orig_ws
        st.is_recording = False
        st.recording_start = None


def logs_of(events):
    return " ".join(str(e.get("data", "")) for e in events
                    if isinstance(e, dict) and e.get("type") == "log")


print("=== 1. 旧 bug 复现：复位语句在 stop() 之后 → 抛异常即跳过 ===")


def old_stop_recording(recorder, flag):
    """忠实复刻旧结构（复位在 try 体内 stop() 之后，except 不复位）。"""
    st = {"is_recording": flag, "recording_start": 999.0}
    try:
        _ = recorder.stop()
        st["is_recording"] = False        # ← stop() 抛异常则永远到不了
        st["recording_start"] = None
    except Exception:
        pass                              # ← 旧代码只发日志，不复位
    return st


old = old_stop_recording(RaisingRecorder(), True)
check("旧写法：stop() 抛异常后 is_recording 仍为 True", old["is_recording"], True)
check("旧写法：recording_start 也未清理", old["recording_start"], 999.0)
print("       → 前端按钮永久停在「录音中」，只能重启进程")
old_ok = old_stop_recording(OkRecorder(), True)
check("旧写法：stop() 正常时复位无碍（故仅异常路径失同步）",
      old_ok["is_recording"], False)

print()
print("=== 2. 修复后：stop() 抛异常仍复位（真实 handle_message）===")
rec = RaisingRecorder()
is_rec, rec_start, events = drive_stop_recording(rec)
check("recorder.stop() 确实被调用", rec.stop_called, 1)
check("is_recording 已复位为 False", is_rec, False)
check("recording_start 已清空", rec_start, None)
kinds = [e.get("type") for e in events if isinstance(e, dict)]
check_true("前端收到 recording_changed 事件", "recording_changed" in kinds,
           f"事件序列 {kinds}")
changed = [e for e in events if isinstance(e, dict)
           and e.get("type") == "recording_changed"]
check("recording_changed 内容为 is_recording=False",
      changed[-1].get("data", {}).get("is_recording"), False)
check_true("状态被推回「就绪」",
           any(e.get("type") == "status" and e.get("data") == "就绪"
               for e in events if isinstance(e, dict)))
lg = logs_of(events)
check_true("失败原因通过 log 上报（不静默）", "停止录音失败" in lg)
check_true("异常文本包含具体原因", "device unavailable" in lg, "便于排查")

print()
print("=== 3. 修复后：stop() 正常时行为不变（无回归）===")
rec2 = OkRecorder(ret=None)
is_rec2, rec_start2, events2 = drive_stop_recording(rec2)
check("stop() 被调用", rec2.stop_called, 1)
check("is_recording 复位", is_rec2, False)
check("recording_start 清空", rec_start2, None)
lg2 = logs_of(events2)
check_true("无数据时提示「没有录到音频数据」", "没有录到音频数据" in lg2)
check_true("正常路径不出现失败日志", "停止录音失败" not in lg2)

print()
print("=== 4. 有录音文件时仍走保存日志（且不误报失败）===")
rec3 = OkRecorder(ret=r"C:\fake\recording_test.wav")
is_rec3, _, events3 = drive_stop_recording(rec3)
check("is_recording 复位", is_rec3, False)
lg3 = logs_of(events3)
check_true("提示录音已保存", "录音已保存" in lg3)
check_true("不误报失败", "停止录音失败" not in lg3)

print()
print("=== 5. 未在录音时点停止：不炸、状态仍为 False ===")
is_rec4, _, _ = drive_stop_recording(RaisingRecorder(), pre_is_recording=False)
check("仍安全复位", is_rec4, False)

print()
print("=== 6. 各类异常都能复位（不只 RuntimeError）===")
for exc in (OSError("stream closed"), ValueError("bad state"),
            AttributeError("NoneType has no attr stop")):
    r = RaisingRecorder(exc)
    got, _, ev = drive_stop_recording(r)
    check(f"{type(exc).__name__} 后仍复位", got, False)
    check_true(f"{type(exc).__name__} 有日志上报", "停止录音失败" in logs_of(ev))

print()
print("=== 7. AST 断言：两处调用点的复位必须在 finally 内 ===")


def owner_name(lineno):
    best = None
    for f in ast.walk(TREE):
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if f.lineno <= lineno <= f.end_lineno:
                if best is None or f.lineno > best.lineno:
                    best = f
    return best.name if best else "?"


call_lines = [i for i, l in enumerate(LINES, 1)
              if "recorder.stop()" in l and not l.strip().startswith("#")]
ui_sites = [i for i in call_lines
            if owner_name(i) in ("handle_message", "on_meeting_stop")]
check("维护 is_recording 的调用点数量", len(ui_sites), 2)
for site in ui_sites:
    fn = owner_name(site)
    tries = [n for n in ast.walk(TREE)
             if isinstance(n, ast.Try) and n.lineno <= site <= n.end_lineno
             and n.finalbody]
    check_true(f"{fn}() L{site} 的 try 带 finally", len(tries) > 0)
    if not tries:
        continue
    inner = min(tries, key=lambda t: t.end_lineno - t.lineno)
    fin = "\n".join(ast.unparse(s) for s in inner.finalbody)
    check_true(f"{fn}() finally 内复位 is_recording", "is_recording = False" in fin)
    check_true(f"{fn}() finally 内清空 recording_start", "recording_start = None" in fin)
    check_true(f"{fn}() finally 内推送 recording_changed", "recording_changed" in fin)

print()
print("=== 8. realtime 路径按设计不维护 is_recording（防误改）===")
rt = [i for i in call_lines if owner_name(i) == "_realtime_transcribe_task"]
check("realtime 内 stop() 调用点数", len(rt), 2)
for f in ast.walk(TREE):
    if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)) and \
            f.name == "_realtime_transcribe_task":
        seg = "\n".join(LINES[f.lineno - 1:f.end_lineno])
        check("realtime 全程不碰 is_recording（它用 is_realtime）",
              seg.count("is_recording"), 0)
        check_true("realtime 确实维护 is_realtime", "is_realtime = False" in seg)

print()
print("=== 9. import re 冗余已清理 ===")
mod_re = [n.lineno for n in TREE.body
          if isinstance(n, ast.Import) and any(a.name == "re" for a in n.names)]
check("模块级 import re 恰好一次", len(mod_re), 1)
check("全文 import re 语句总数为 1",
      len([i for i, l in enumerate(LINES, 1) if l.strip() == "import re"]), 1)
check_true("re 仍可正常使用（模块级已导入）", hasattr(app, "re"))
check_true("_save_transcript_docx 仍使用 re.split（证明删的是冗余不是必需）",
           any("re.split" in l for l in LINES))

print()
print("=== 10. 静态防回归 ===")
check("不存在「复位紧跟 stop() 且无 finally」的旧结构",
      SRC.count("filepath = state.recorder.stop()\n                state.is_recording = False"), 0)
check_true("stop_recording 分支存在", 'elif action == "stop_recording":' in SRC)
check_true("on_meeting_stop 存在", "def on_meeting_stop(" in SRC)

print()
print(f"通过 {len(PASS)} 条，失败 {len(FAIL)} 条")
if FAIL:
    for f in FAIL:
        print(f"  - {f}")
sys.exit(1 if FAIL else 0)
