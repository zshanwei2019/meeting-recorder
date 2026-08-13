#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试FunASR流式转写API"""
import sys
import os
import numpy as np
import time

# 设置ffmpeg路径
_ffdir = os.path.join(os.path.expanduser('~'), 'ffmpeg-bin')
if os.path.exists(_ffdir) and _ffdir not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _ffdir + os.pathsep + os.environ.get('PATH', '')

print("Loading FunASR streaming model...")
from funasr import AutoModel

model = AutoModel(
    model="paraformer-zh-streaming",
    model_revision="v2.0.4",
    disable_update=True
)
print("Model loaded!")

# 生成测试音频: 1秒静音
sample_rate = 16000
silence = np.zeros(sample_rate, dtype=np.float32)

# 流式推理测试
chunk_size = [0, 10, 5]  # 960ms per chunk
cache = {}

print("\nTest 1: Empty audio chunks...")
for i in range(3):
    chunk = np.zeros(int(sample_rate * 0.96), dtype=np.float32)  # 960ms
    is_final = (i == 2)
    try:
        result = model.generate(chunk, cache=cache, chunk_size=chunk_size, is_final=is_final)
        print(f"  Chunk {i}: result={result}")
    except Exception as e:
        print(f"  Chunk {i}: error={e}")

# 测试用真实音频
print("\nTest 2: Real audio file...")
test_wav = r"C:\Users\27204\MeetingRecordings\test_30s.wav"
if os.path.exists(test_wav):
    import wave
    with wave.open(test_wav, 'rb') as wf:
        sr = wf.getframerate()
        n = wf.getnframes()
        raw = wf.readframes(n)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if wf.getnchannels() > 1:
        data = data.reshape(-1, wf.getnchannels()).mean(axis=1)
    
    # 重采样到16kHz
    if sr != 16000:
        ratio = 16000 / sr
        n_new = int(len(data) * ratio)
        indices = np.linspace(0, len(data)-1, n_new)
        data = np.interp(indices, np.arange(len(data)), data)
    
    print(f"  Audio: {len(data)/16000:.1f}s, peak={np.abs(data).max():.4f}")
    
    # 分chunk流式推理
    chunk_samples = int(16000 * 0.96)  # 960ms = 15360 samples
    cache2 = {}
    full_text = ""
    
    n_chunks = len(data) // chunk_samples
    t0 = time.time()
    for i in range(n_chunks):
        chunk = data[i*chunk_samples : (i+1)*chunk_samples]
        is_final = (i == n_chunks - 1)
        try:
            result = model.generate(chunk, cache=cache2, chunk_size=chunk_size, is_final=is_final)
            if result and len(result) > 0:
                text = result[0].get('text', '') if isinstance(result[0], dict) else str(result[0])
                if text:
                    full_text += text
                    print(f"  [{i+1}/{n_chunks}] +{text}")
        except Exception as e:
            print(f"  [{i+1}/{n_chunks}] error: {e}")
    
    elapsed = time.time() - t0
    print(f"\n  Total: {elapsed:.1f}s for {len(data)/16000:.1f}s audio")
    print(f"  Text: {full_text[:200]}")
else:
    print(f"  File not found: {test_wav}")

print("\nDone!")
