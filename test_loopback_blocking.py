#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试WASAPI Loopback阻塞模式录音"""
import sys
import time
import threading
import numpy as np

print("=" * 50)
print("WASAPI Loopback 阻塞模式录音测试")
print("=" * 50)

import pyaudiowpatch as pyaudio

p = pyaudio.PyAudio()
wasapi_info = p.get_default_wasapi_loopback()
dev_idx = wasapi_info['index']
rate = int(wasapi_info['defaultSampleRate'])
channels = int(wasapi_info['maxInputChannels'])
print(f"设备: {wasapi_info['name']}, 采样率: {rate}Hz, 通道: {channels}")

# 阻塞模式打开
stream = p.open(
    format=pyaudio.paInt16,
    channels=channels,
    rate=rate,
    input=True,
    input_device_index=dev_idx,
    frames_per_buffer=4800
)
print("流已打开（阻塞模式）")

frames = []
recording = True

def rec_thread():
    chunk = 4800
    while recording:
        try:
            data = stream.read(chunk, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)
            frames.append(audio.copy())
        except Exception as e:
            if recording:
                print(f"读取异常: {e}")
            break

t = threading.Thread(target=rec_thread, daemon=True)
t.start()
print("录音线程已启动，录5秒...")

for i in range(5):
    time.sleep(1)
    total_samples = sum(len(f) for f in frames)
    dur = total_samples / rate
    peak = max(np.abs(f).max() for f in frames) if frames else 0
    print(f"  {i+1}s - 帧块数={len(frames)}, 时长={dur:.1f}s, peak={peak:.4f}")

# 停止
print("\n停止录音...")
recording = False
t.join(timeout=5)
if t.is_alive():
    print("  录音线程5秒未结束！")
else:
    print("  录音线程已结束")

stream.stop_stream()
stream.close()
print("  流已关闭")
p.terminate()

total_samples = sum(len(f) for f in frames)
dur = total_samples / rate
peak = max(np.abs(f).max() for f in frames) if frames else 0
print(f"\n结果: {len(frames)}帧块, {dur:.1f}秒, peak={peak:.4f}")
if dur >= 4.5:
    print(">>> 阻塞模式正常！录到了完整时长")
else:
    print(">>> 时长不够，有问题")

print("\n测试完成")
