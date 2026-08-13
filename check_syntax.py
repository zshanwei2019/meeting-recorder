#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick syntax check for meeting_recorder.py"""
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except:
    pass

import py_compile
try:
    py_compile.compile('meeting_recorder.py', doraise=True)
    print("Syntax OK!")
except py_compile.PyCompileError as e:
    print(f"Syntax Error: {e}")
