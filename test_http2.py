#!/usr/bin/env python
"""Test: HTTP server + url= with index.html"""
import webview, threading, time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os, socket

os.chdir(r"C:\Users\27204\Desktop\meeting-recorder\ui")

class QH(SimpleHTTPRequestHandler):
    def log_message(self, f, *a): pass

# Find free port
port = None
for p in range(18800, 18820):
    try:
        s = HTTPServer(('127.0.0.1', p), QH)
        port = p
        break
    except OSError:
        continue

t = threading.Thread(target=s.serve_forever, daemon=True)
t.start()

# Wait for ready
for _ in range(10):
    try:
        import urllib.request
        urllib.request.urlopen(f'http://127.0.0.1:{port}/index.html', timeout=1)
        print(f"Server ready on port {port}")
        break
    except:
        time.sleep(0.3)

window = webview.create_window(
    "会议录音转写助手 v3.0.0",
    url=f"http://127.0.0.1:{port}/index.html",
    width=1400, height=860,
    background_color='#0f1117',
    text_select=True,
)
webview.start(debug=False)
