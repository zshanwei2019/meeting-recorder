#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""测试WASAPI Loopback录音的启动和停止"""
import sys
import time
import numpy as np

print("=" * 50)
print("WASAPI Loopback 录音测试")
print("=" * 50)

# 1. 导入PyAudioWPatch
print("\n[1] 导入 PyAudioWPatch...")
try:
    import pyaudiowpatch as pyaudio
    print(f"    OK - PyAudioWPatch 版本: {pyaudio.__version__ if hasattr(pyaudio, '__version__') else 'unknown'}")
except ImportError as e:
    print(f"    失败: {e}")
    sys.exit(1)

# 2. 查找Loopback设备
print("\n[2] 查找 WASAPI Loopback 设备...")
p = pyaudio.PyAudio()
loopback_dev = None
try:
    wasapi_info = p.get_default_wasapi_loopback()
    loopback_dev = wasapi_info
    print(f"    找到: index={wasapi_info['index']}, name={wasapi_info['name']}")
    print(f"    采样率: {wasapi_info['defaultSampleRate']}Hz")
    print(f"    通道数: {wasapi_info['maxInputChannels']}")
except OSError as e:
    print(f"    get_wasapi_loopback() 失败: {e}")
    # 手动搜索
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if 'loopback' in dev['name'].lower() or 'Loopback' in dev['name']:
            loopback_dev = dev
            print(f"    手动找到: index={i}, name={dev['name']}")
            break

if not loopback_dev:
    print("    没找到Loopback设备！")
    p.terminate()
    sys.exit(1)

# 3. 打开流
print("\n[3] 打开 WASAPI Loopback 流...")
frames_data = []
dev_idx = loopback_dev['index']
rate = int(loopback_dev['defaultSampleRate'])
channels = int(loopback_dev['maxInputChannels'])

def callback(in_data, frame_count, time_info, status):
    audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    frames_data.append(audio.copy())
    peak = np.abs(audio).max()
    if len(frames_data) % 50 == 0:  # 每50帧打印一次
        print(f"    [callback] frames={len(frames_data)}, peak={peak:.4f}")
    return (in_data, pyaudio.paContinue)

try:
    stream = p.open(
        format=pyaudio.paInt16,
        channels=channels,
        rate=rate,
        input=True,
        input_device_index=dev_idx,
        stream_callback=callback,
        frames_per_buffer=4800,
        start=False
    )
    print("    流已创建 (start=False)")
except Exception as e:
    print(f"    打开流失败: {e}")
    p.terminate()
    sys.exit(1)

# 4. 启动流
print("\n[4] 启动流...")
try:
    stream.start_stream()
    print(f"    流已启动, is_active={stream.is_active()}")
except Exception as e:
    print(f"    启动失败: {e}")
    p.terminate()
    sys.exit(1)

# 5. 录音5秒
print("\n[5] 录音5秒（请播放音乐/视频）...")
for i in range(5):
    time.sleep(1)
    print(f"    {i+1}s - frames={len(frames_data)}, is_active={stream.is_active()}")

# 6. 停止流
print("\n[6] 停止流...")
import threading
stop_result = {'done': False, 'error': None}

def do_stop():
    try:
        stream.stop_stream()
        stop_result['done'] = True
        print("    stop_stream() 完成")
    except Exception as e:
        stop_result['error'] = e
        print(f"    stop_stream() 异常: {e}")

t = threading.Thread(target=do_stop, daemon=True)
t.start()
t.join(timeout=5)

if t.is_alive():
    print("    !!! stop_stream() 卡住了！5秒超时 !!!")
    print("    这就是GUI一直显示'录音中'的原因")
else:
    if stop_result['done']:
        print("    stop_stream() 正常完成")
    elif stop_result['error']:
        print(f"    stop_stream() 出错: {stop_result['error']}")

# 7. 关闭流
print("\n[7] 关闭流...")
try:
    if not t.is_alive():
        stream.close()
        print("    close() 完成")
    else:
        print("    跳过close()（stop_stream卡住）")
except Exception as e:
    print(f"    close() 异常: {e}")

p.terminate()
print("\n    PyAudio terminate() 完成")

# 8. 检查录音数据
print("\n[8] 录音数据检查...")
if frames_data:
    audio = np.concatenate(frames_data)
    peak = np.abs(audio).max()
    duration = len(audio) / rate
    print(f"    总帧数: {len(frames_data)}")
    print(f"    音频长度: {duration:.1f}秒")
    print(f"    峰值: {peak:.4f}")
    if peak > 0.01:
        print("    >>> 录到了声音！")
    else:
        print("    >>> 静音，没录到声音")
else:
    print("    没有录音数据！")

print("\n" + "=" * 50)
print("测试完成")
