#!/usr/bin/env python
"""Quick test: load HTML directly via html= parameter"""
import webview

# Read the HTML file
with open(r"C:\Users\27204\Desktop\meeting-recorder\ui\index.html", "r", encoding="utf-8") as f:
    html_content = f.read()

class Api:
    def get_config(self):
        return {"audio_source": "system", "engine": "FunASR"}
    def get_events(self, since_id):
        return []
    def set_config(self, config):
        pass

api = Api()
window = webview.create_window(
    "HTML Direct Test",
    html=html_content,
    js_api=api,
    width=1400, height=860,
    background_color='#0f1117',
    text_select=True,
)
webview.start(debug=False)
