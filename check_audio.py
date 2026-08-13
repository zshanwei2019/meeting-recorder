#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""检查录音文件的音频质量"""
import wave
import numpy as np
import sys
import os

recordings_dir = r"C:\Users\27204\MeetingRecordings"

files = sorted([f for f in os.listdir(recordings_dir) if f.endswith('.wav')], reverse=True)

for fname in files[:5]:
    fpath = os.path.join(recordings_dir, fname)
    with wave.open(fpath, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        duration = n_frames / framerate
        raw = wf.readframes(n_frames)
    
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if n_channels > 1:
        data = data.reshape(-1, n_channels).mean(axis=1)
    
    peak = np.abs(data).max()
    rms = np.sqrt(np.mean(data**2))
    
    # 计算静音比例
    silence_threshold = 0.001
    silent_samples = np.sum(np.abs(data) < silence_threshold)
    silence_ratio = silent_samples / len(data)
    
    print(f"\n{fname}:")
    print(f"  时长: {duration:.1f}s, 采样率: {framerate}Hz, 通道: {n_channels}")
    print(f"  Peak: {peak:.4f}, RMS: {rms:.6f}")
    print(f"  静音比例: {silence_ratio*100:.1f}%")
    
    if peak < 0.001:
        print(f"  >>> 完全静音！没录到声音")
    elif silence_ratio > 0.9:
        print(f"  >>> 90%以上是静音，基本没声音")
    elif peak > 0.5:
        print(f"  >>> 声音正常，音量充足")
    else:
        print(f"  >>> 有声音但音量偏小")
