# -*- coding: utf-8 -*-
"""文件转写“服务端权威状态”回归测试（修复会后处理重转按钮回弹）。

背景：前端原先用本地 180s 定时器猜重转是否结束；说话人分离可能跑十几分钟，
定时器一到就把“重转中”复位为“重转”，但后端其实还在转。改为服务端权威状态 +
GET /api/transcribe/status 轮询。本测试直接验证：
  1. AppState.transcribe_begin / set_stage / end 的状态机；
  2. create_app() 注册的真实 /api/transcribe/status 返回 active/wav_name/stage/elapsed_s；
  3. 阶段为空/非活动时不误改。

不起 server、不开线程、不加载 ASR、不碰音频、不联网（直接调路由 endpoint）。
运行：.venv\\Scripts\\python.exe tests\\test_transcribe_status.py
"""
import asyncio
import json
import sys
import time
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


def get_status_endpoint():
    fastapi_app = app.create_app()
    for r in fastapi_app.routes:
        if getattr(r, "path", None) == "/api/transcribe/status":
            return r.endpoint
    return None


def call_status():
    ep = get_status_endpoint()
    assert ep is not None, "未注册 /api/transcribe/status"
    resp = asyncio.run(ep())
    return json.loads(resp.body.decode("utf-8"))


st = app.state
_saved = {
    "is_transcribing": st.is_transcribing,
    "transcribing_wav": st.transcribing_wav,
    "transcribing_start": st.transcribing_start,
    "transcribing_stage": st.transcribing_stage,
}

try:
    st.transcribe_end()  # 确保从干净态开始

    print("=== 1. 空闲态 ===")
    idle = call_status()
    check("空闲 active=False", idle.get("active") is False, idle)
    check("空闲 wav_name=None", idle.get("wav_name") is None, idle)
    check("空闲 stage=空串", idle.get("stage") == "", idle)
    check("空闲 elapsed=0", idle.get("elapsed_s") == 0, idle)

    print("\n=== 2. begin 状态机 + endpoint 序列化 ===")
    st.transcribe_begin("rec_20260914.wav", "说话人分离中…")
    check("begin 后 is_transcribing=True", st.is_transcribing is True)
    check("begin 记录 wav_name", st.transcribing_wav == "rec_20260914.wav")
    check("begin 记录 stage", st.transcribing_stage == "说话人分离中…")
    check("begin 记录起点时间", isinstance(st.transcribing_start, float))
    # 模拟已经跑了 65 秒
    st.transcribing_start = time.time() - 65
    busy = call_status()
    check("忙 active=True", busy.get("active") is True, busy)
    check("忙 wav_name 回传", busy.get("wav_name") == "rec_20260914.wav", busy)
    check("忙 stage 回传", busy.get("stage") == "说话人分离中…", busy)
    check("忙 elapsed≈65s", 65 <= busy.get("elapsed_s", 0) <= 70, busy)

    print("\n=== 3. set_stage 更新规则 ===")
    st.transcribe_set_stage("声纹聚类中…")
    check("活动时 set_stage 生效", st.transcribing_stage == "声纹聚类中…")
    st.transcribe_set_stage("")
    check("空 stage 被忽略（不覆盖当前阶段）", st.transcribing_stage == "声纹聚类中…")
    st.transcribe_end()
    st.transcribe_set_stage("非活动时不应生效")
    check("非活动时 set_stage 无效", st.transcribing_stage == "")

    print("\n=== 4. end 复位 + endpoint 回到空闲态 ===")
    check("end 后 is_transcribing=False", st.is_transcribing is False)
    check("end 后 wav_name=None", st.transcribing_wav is None)
    check("end 后 start=None", st.transcribing_start is None)
    idle2 = call_status()
    check("end 后 endpoint active=False", idle2.get("active") is False, idle2)
    check("end 后 endpoint wav_name=None", idle2.get("wav_name") is None, idle2)

    print("\n=== 5. 并发互斥读取（is_transcribing 布尔可被 WS 重转拒绝逻辑使用）===")
    check("空闲时重转允许（is_transcribing False）",
          getattr(st, "is_transcribing", False) is False)
    st.transcribe_begin("a.wav")
    check("有任务时重转应被拒绝（is_transcribing True）",
          getattr(st, "is_transcribing", False) is True)
    st.transcribe_end()

finally:
    # 还原全局状态，避免污染同进程其它测试
    st.is_transcribing = _saved["is_transcribing"]
    st.transcribing_wav = _saved["transcribing_wav"]
    st.transcribing_start = _saved["transcribing_start"]
    st.transcribing_stage = _saved["transcribing_stage"]

print(f"\n通过 {passed} 条，失败 {failed} 条")
sys.exit(1 if failed else 0)
