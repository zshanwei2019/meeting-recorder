#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试WASAPI Loopback - 高频静音填充"""
import sys
import time
import threading
import numpy as np

print("WASAPI Loopback 高频静音填充测试")

import pyaudiowpatch as pyaudio

p = pyaudio.PyAudio()
wasapi_info = p.get_default_wasapi_loopback()
dev_idx = wasapi_info['index']
rate = int(wasapi_info['defaultSampleRate'])
channels = int(wasapi_info['maxInputChannels'])
print(f"设备: {wasapi_info['name']}, {rate}Hz, {channels}ch")

frames = []
recording = [True]
# 用期望时间来追踪，而不是检测间隔
expected_time = [time.time()]

chunk_samples = 4800
chunk_duration = chunk_samples / rate  # 0.1秒

def callback(in_data, frame_count, time_info, status):
    audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    frames.append(audio.copy())
    expected_time[0] = time.time() + chunk_duration
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
start_time = time.time()
expected_time[0] = start_time + chunk_duration
print("callback流已启动")

# 高频填充线程：每0.02秒检查，如果当前时间超过期望时间就填充
def silence_filler():
    silence = np.zeros(chunk_samples, dtype=np.float32)
    while recording[0]:
        now = time.time()
        if now >= expected_time[0] + chunk_duration:
            # 需要填充
            n_chunks = int((now - expected_time[0]) / chunk_duration) + 1
            for _ in range(min(n_chunks, 5)):  # 一次最多填5个chunk
                frames.append(silence.copy())
                expected_time[0] += chunk_duration
        time.sleep(0.02)

filler = threading.Thread(target=silence_filler, daemon=True)
filler.start()
print("高频填充线程已启动，录5秒...")

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
actual_dur = time.time() - start_time
print(f"\n实际经过: {actual_dur:.1f}秒, 录音时长: {dur:.1f}秒, peak={peak:.4f}")
if dur >= actual_dur - 0.5:
    print(">>> 成功！时长基本匹配")
else:
    print(f">>> 差{actual_dur-dur:.1f}秒")

print("测试完成")
