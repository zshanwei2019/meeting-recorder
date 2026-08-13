#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试WASAPI Loopback - 精确静音填充"""
import sys
import time
import threading
import numpy as np

print("WASAPI Loopback 精确静音填充测试")

import pyaudiowpatch as pyaudio

p = pyaudio.PyAudio()
wasapi_info = p.get_default_wasapi_loopback()
dev_idx = wasapi_info['index']
rate = int(wasapi_info['defaultSampleRate'])
channels = int(wasapi_info['maxInputChannels'])
print(f"设备: {wasapi_info['name']}, {rate}Hz, {channels}ch")

frames = []
recording = [True]
last_cb_time = [time.time()]
chunk_samples = 4800  # 每次callback的样本数
chunk_duration = chunk_samples / rate  # 0.1秒

def callback(in_data, frame_count, time_info, status):
    audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    frames.append(audio.copy())
    last_cb_time[0] = time.time()
    return (in_data, pyaudio.paContinue)

stream = p.open(
    format=pyaudio.paInt16,
    channels=channels,
    rate=rate,
    input=True,
    input_device_index=dev_idx,
    stream_callback=callback,
    frames_per_buffer=chunk_samples,
    start=False
)
stream.start_stream()
print("callback流已启动")

# 精确静音填充：每0.05秒检查一次，如果超过0.15秒没callback就填充
def silence_filler():
    silence = np.zeros(chunk_samples, dtype=np.float32)
    while recording[0]:
        time.sleep(0.05)
        elapsed = time.time() - last_cb_time[0]
        if elapsed > 0.15:
            # 需要填充多少个chunk
            n_chunks = int(elapsed / chunk_duration)
            for _ in range(n_chunks):
                frames.append(silence.copy())
            last_cb_time[0] = time.time()

filler = threading.Thread(target=silence_filler, daemon=True)
filler.start()
print("静音填充线程已启动，录5秒...")

for i in range(5):
    time.sleep(1)
    total = sum(len(f) for f in frames)
    dur = total / rate
    peak = max(np.abs(f).max() for f in frames) if frames else 0
    print(f"  {i+1}s - 帧块={len(frames)}, 时长={dur:.1f}s, peak={peak:.4f}")

recording[0] = False
time.sleep(0.2)
stream.stop_stream()
stream.close()
p.terminate()

total = sum(len(f) for f in frames)
dur = total / rate
peak = max(np.abs(f).max() for f in frames) if frames else 0
print(f"\n结果: {dur:.1f}秒, peak={peak:.4f}")
if dur >= 4.5:
    print(">>> 成功！录到了完整时长")
else:
    print(f">>> 时长差{5-dur:.1f}秒")

print("测试完成")
