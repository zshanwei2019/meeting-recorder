#!/usr/bin/env python
"""Test: minimal dark HTML with same structure as index.html"""
import webview

html = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
html, body { background:#0f1117; color:#f1f5f9; font-family:"Segoe UI",sans-serif; height:100vh; width:100vw; }
.app { display:flex; height:100vh; width:100vw; background:#0f1117; }
.sidebar { width:272px; background:#161922; padding:24px 20px; border-right:1px solid rgba(148,163,184,0.08); display:flex; flex-direction:column; }
.main { flex:1; background:#0f1117; padding:28px 32px; display:flex; flex-direction:column; }
.logo-icon { width:40px; height:40px; background:linear-gradient(135deg,#6366f1,#a855f7); border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:18px; font-weight:700; color:white; margin-bottom:20px; }
.status-box { background:#1c1f2e; border-radius:10px; padding:14px 16px; margin-bottom:20px; }
.status-dot { width:8px; height:8px; border-radius:50%; background:#22c55e; display:inline-block; }
.btn-record { width:100%; padding:12px 16px; background:linear-gradient(135deg,#dc2626,#ef4444); color:white; border:none; border-radius:10px; font-size:14px; font-weight:600; margin-bottom:8px; cursor:pointer; }
.btn-realtime { width:100%; padding:12px 16px; background:linear-gradient(135deg,#d97706,#f59e0b); color:white; border:none; border-radius:10px; font-size:14px; font-weight:600; margin-bottom:12px; cursor:pointer; }
.btn-secondary { flex:1; padding:9px 8px; background:#1c1f2e; border:1px solid rgba(148,163,184,0.12); border-radius:6px; color:#94a3b8; font-size:12px; font-weight:600; cursor:pointer; }
.btn-row { display:flex; gap:6px; margin-bottom:6px; }
.transcript-box { flex:1; background:#161922; border-radius:14px; border:1px solid rgba(148,163,184,0.08); padding:24px; color:#64748b; font-size:15px; }
.toolbar { display:flex; gap:8px; margin-top:12px; }
.toolbar-btn-primary { padding:8px 20px; background:#6366f1; color:white; border:none; border-radius:6px; font-size:13px; font-weight:600; cursor:pointer; }
.toolbar-btn-outline { padding:8px 20px; background:transparent; border:1px solid rgba(148,163,184,0.12); color:#94a3b8; border-radius:6px; font-size:13px; cursor:pointer; }
.search-box { background:#1c1f2e; border:1px solid rgba(148,163,184,0.12); border-radius:6px; padding:8px 12px; color:#f1f5f9; font-size:12px; width:160px; }
.word-count { font-size:11px; color:#64748b; }
.divider { height:1px; background:rgba(148,163,184,0.08); margin:12px 0; }
.section-label { font-size:11px; font-weight:700; color:#64748b; margin-bottom:8px; }
.source-select { width:100%; background:#1c1f2e; border:1px solid rgba(148,163,184,0.12); border-radius:6px; padding:9px 12px; color:#f1f5f9; font-size:13px; margin-bottom:16px; }
</style>
</head>
<body>
<div class="app">
  <div class="sidebar">
    <div class="logo-icon">M</div>
    <div style="font-size:15px;font-weight:700;color:#f1f5f9;margin-bottom:4px;">会议录音转写助手</div>
    <div style="font-size:11px;color:#64748b;margin-bottom:20px;">v3.0.0</div>
    <div class="status-box">
      <span class="status-dot"></span>
      <span style="font-size:13px;font-weight:600;color:#f1f5f9;margin-left:8px;">就绪</span>
    </div>
    <div class="section-label">音源</div>
    <select class="source-select">
      <option>系统音频</option>
      <option>麦克风</option>
    </select>
    <button class="btn-record">● 开始录音</button>
    <button class="btn-realtime">● 实时转写</button>
    <div class="btn-row">
      <button class="btn-secondary">转写文件</button>
      <button class="btn-secondary">AI 纪要</button>
    </div>
    <div class="btn-row">
      <button class="btn-secondary">保存</button>
      <button class="btn-secondary">复制</button>
    </div>
    <div class="divider"></div>
    <div style="font-size:12px;color:#818cf8;cursor:pointer;">▶ 设置</div>
  </div>
  <div class="main">
    <h1 style="font-size:22px;font-weight:700;color:#f1f5f9;margin-bottom:16px;">转写结果</h1>
    <div class="transcript-box">转写结果将显示在这里...</div>
    <div class="toolbar">
      <button class="toolbar-btn-primary">保存</button>
      <button class="toolbar-btn-outline">复制</button>
      <div style="flex:1;"></div>
      <input class="search-box" placeholder="搜索...">
      <span class="word-count">字数: 0</span>
    </div>
  </div>
</div>
</body>
</html>"""

window = webview.create_window("会议录音转写助手 v3.0.0", html=html, width=1400, height=860, background_color='#0f1117', text_select=True)
webview.start(debug=False)
