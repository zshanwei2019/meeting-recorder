#!/usr/bin/env python
import webview

html = """<!DOCTYPE html>
<html>
<head><style>
* { margin:0; padding:0; box-sizing:border-box; }
html, body { background:#0f1117; color:#f1f5f9; font-family:"Segoe UI",sans-serif; height:100vh; width:100vw; }
.app { display:flex; height:100vh; background:#0f1117; }
.sidebar { width:272px; background:#161922; padding:24px 20px; border-right:1px solid rgba(148,163,184,0.08); }
.main { flex:1; background:#0f1117; padding:28px 32px; }
.logo-icon { width:40px; height:40px; background:linear-gradient(135deg,#6366f1,#a855f7); border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:18px; font-weight:700; color:white; }
.btn-record { width:100%; padding:12px 16px; background:linear-gradient(135deg,#dc2626,#ef4444); color:white; border:none; border-radius:10px; font-size:14px; font-weight:600; margin-top:16px; cursor:pointer; }
.btn-realtime { width:100%; padding:12px 16px; background:linear-gradient(135deg,#d97706,#f59e0b); color:white; border:none; border-radius:10px; font-size:14px; font-weight:600; margin-top:8px; cursor:pointer; }
.status-box { background:#1c1f2e; border-radius:10px; padding:14px 16px; margin-top:20px; border:1px solid rgba(148,163,184,0.08); }
.status-dot { width:8px; height:8px; border-radius:50%; background:#22c55e; display:inline-block; }
.transcript-box { flex:1; background:#161922; border-radius:14px; border:1px solid rgba(148,163,184,0.08); padding:24px; color:#f1f5f9; font-size:15px; }
h1 { font-size:22px; font-weight:700; color:#f1f5f9; }
</style></head>
<body>
<div class="app">
  <div class="sidebar">
    <div class="logo-icon">M</div>
    <div style="margin-top:8px; font-size:15px; font-weight:700; color:#f1f5f9;">会议录音转写助手</div>
    <div style="font-size:11px; color:#64748b;">v3.0.0</div>
    <div class="status-box">
      <span class="status-dot"></span>
      <span style="font-size:13px; font-weight:600; color:#f1f5f9; margin-left:8px;">就绪</span>
    </div>
    <button class="btn-record">● 开始录音</button>
    <button class="btn-realtime">● 实时转写</button>
  </div>
  <div class="main">
    <h1>转写结果</h1>
    <div class="transcript-box" style="margin-top:16px;">转写结果将显示在这里...</div>
  </div>
</div>
</body>
</html>"""

window = webview.create_window("会议录音转写助手 v3.0.0", html=html, width=1400, height=860, background_color='#0f1117')
webview.start(debug=False)
