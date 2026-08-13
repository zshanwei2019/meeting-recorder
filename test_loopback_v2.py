#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试WASAPI Loopback - 用stream.read加超时+静音填充"""
import sys
import time
import threading
import numpy as np

print("=" * 50)
print("WASAPI Loopback 超时+静音填充测试")
print("=" * 50)

import pyaudiowpatch as pyaudio

p = pyaudio.PyAudio()
wasapi_info = p.get_default_wasapi_loopback()
dev_idx = wasapi_info['index']
rate = int(wasapi_info['defaultSampleRate'])
channels = int(wasapi_info['maxInputChannels'])
print(f"设备: {wasapi_info['name']}, 采样率: {rate}Hz, 通道: {channels}")

# 用callback模式，但用事件驱动+静音填充
import struct

frames_data = []
silence_chunk = b'\x00\x00' * (4800 * channels)  # 静音数据

# 方案：用callback模式，但设置一个定时器来检测静音间隔
# WASAPI Loopback在没有声音时不触发callback
# 我们需要用另一种方式：用PyAudio的blocking read + 超时

# 实际上PyAudio的read()不支持超时参数
# 但我们可以用select或者用线程+超时

# 方案B：用sounddevice的WASAPI loopback
import sounddevice as sd

# 查找loopback设备
devices = sd.query_devices()
loopback_id = None
for i, dev in enumerate(devices):
    if 'loopback' in dev['name'].lower() or 'Loopback' in dev['name']:
        if dev['max_input_channels'] > 0:
            loopback_id = i
            print(f"sounddevice找到loopback: [{i}] {dev['name']}")
            break

if loopback_id is not None:
    # sounddevice的InputStream在WASAPI loopback上也会卡住
    # 但sounddevice有latency参数可以控制
    print("\n用sounddevice测试WASAPI Loopback...")
    
    frames = []
    recording = [True]
    
    def callback(indata, frame_count, time_info, status):
        mono = np.mean(indata, axis=1) if indata.shape[1] > 1 else indata[:, 0]
        frames.append(mono.copy())
    
    try:
        stream = sd.InputStream(
            device=loopback_id,
            channels=channels,
            samplerate=rate,
            callback=callback,
            blocksize=4800,
            latency='high'  # 高延迟模式，减少卡顿
        )
        stream.start()
        print("sounddevice流已启动")
        
        for i in range(5):
            time.sleep(1)
            total = sum(len(f) for f in frames)
            dur = total / rate
            peak = max(np.abs(f).max() for f in frames) if frames else 0
            print(f"  {i+1}s - 帧块={len(frames)}, 时长={dur:.1f}s, peak={peak:.4f}")
        
        stream.stop()
        stream.close()
        print("流已关闭")
    except Exception as e:
        print(f"sounddevice失败: {e}")
else:
    print("sounddevice没找到loopback设备")

# 方案C：用Windows Core Audio API直接录音
# 这是最可靠的方式，钉钉就是用这个
print("\n" + "=" * 50)
print("尝试方案C: Windows Core Audio (comtypes)")
print("=" * 50)

try:
    import comtypes
    from ctypes import cast, POINTER, c_float
    print("comtypes可用")
except ImportError:
    print("comtypes未安装，尝试安装...")
    import subprocess
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'comtypes'], check=True)
    import comtypes
    print("comtypes安装成功")

# 最终方案：用PyAudio callback模式 + 定时器检测静音
# 当callback不触发时，手动插入静音帧
print("\n" + "=" * 50)
print("方案D: PyAudio callback + 静音填充线程")
print("=" * 50)

p2 = pyaudio.PyAudio()
frames2 = []
recording2 = [True]
last_callback_time = [time.time()]

def callback2(in_data, frame_count, time_info, status):
    audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    frames2.append(audio.copy())
    last_callback_time[0] = time.time()
    return (in_data, pyaudio.paContinue)

stream2 = p2.open(
    format=pyaudio.paInt16,
    channels=channels,
    rate=rate,
    input=True,
    input_device_index=dev_idx,
    stream_callback=callback2,
    frames_per_buffer=4800,
    start=False
)
stream2.start_stream()
print("callback流已启动")

# 静音填充线程：当callback超过0.2秒没触发时，插入静音
def silence_filler():
    silence = np.zeros(4800, dtype=np.float32)
    while recording2[0]:
        time.sleep(0.1)
        if time.time() - last_callback_time[0] > 0.2:
            frames2.append(silence.copy())
            last_callback_time[0] = time.time()

filler = threading.Thread(target=silence_filler, daemon=True)
filler.start()
print("静音填充线程已启动")

for i in range(5):
    time.sleep(1)
    total = sum(len(f) for f in frames2)
    dur = total / rate
    peak = max(np.abs(f).max() for f in frames2) if frames2 else 0
    print(f"  {i+1}s - 帧块={len(frames2)}, 时长={dur:.1f}s, peak={peak:.4f}")

recording2[0] = False
stream2.stop_stream()
stream2.close()
p2.terminate()

total = sum(len(f) for f in frames2)
dur = total / rate
peak = max(np.abs(f).max() for f in frames2) if frames2 else 0
print(f"\n结果: {len(frames2)}帧块, {dur:.1f}秒, peak={peak:.4f}")
if dur >= 4.5:
    print(">>> 方案D成功！callback+静音填充可以录到完整时长")
else:
    print(">>> 时长仍不够")

p.terminate()
print("\n全部测试完成")
