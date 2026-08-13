#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""会议录音转写助手 - 启动入口（加载pyc核心模块）"""
import importlib.util
import sys
import os

pyc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "core.cpython-311.pyc")
spec = importlib.util.spec_from_file_location("meeting_recorder_core", pyc_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

if __name__ == "__main__":
    mod.run_gui()
