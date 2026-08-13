#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""端到端录音测试 - WASAPI Loopback callback+静音填充"""
import sys, os, time, threading, wave
import numpy as np

# 设置ffmpeg
_ffdir = os.path.join(os.path.expanduser('~'), 'ffmpeg-bin')
if os.path.exists(_ffdir) and _ffdir not in os.environ.get('PATH', ''):
    os.environ['PATH'] = _ffdir + os.pathsep + os.environ.get('PATH', '')

import pyaudiowpatch as pyaudio

SAMPLE_RATE = 16000

print("=" * 50)
print("WASAPI Loopback E2E test")

# Find loopback device
p = pyaudio.PyAudio()
loopback_idx = None
loopback_info = None
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev.get('isLoopbackDevice', False) and dev['maxInputChannels'] > 0:
        loopback_idx = i
        loopback_info = dev
        break
p.terminate()

if not loopback_idx:
    print("No loopback device found!")
    sys.exit(1)

print(f"Device: {loopback_info['name']}")
print(f"Rate: {loopback_info['defaultSampleRate']}, Channels: {loopback_info['maxInputChannels']}")

# Record
p = pyaudio.PyAudio()
rate = int(loopback_info['defaultSampleRate'])
channels = int(loopback_info['maxInputChannels'])
chunk_samples = 4800
chunk_duration = chunk_samples / rate
frames = []
recording = [True]
expected_time = [time.time() + chunk_duration]
callback_count = [0]
filler_count = [0]

def callback(in_data, frame_count, time_info, status):
    audio = np.frombuffer(in_data, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    frames.append(audio.copy())
    callback_count[0] += 1
    expected_time[0] = time.time() + chunk_duration
    return (in_data, pyaudio.paContinue)

stream = p.open(
    format=pyaudio.paInt16,
    channels=channels,
    rate=rate,
    input=True,
    input_device_index=loopback_idx,
    stream_callback=callback,
    frames_per_buffer=chunk_samples,
    start=False
)
stream.start_stream()

# Silence filler thread
def filler():
    silence = np.zeros(chunk_samples, dtype=np.float32)
    while recording[0]:
        now = time.time()
        if now >= expected_time[0] + chunk_duration:
            n = int((now - expected_time[0]) / chunk_duration) + 1
            for _ in range(min(n, 5)):
                frames.append(silence.copy())
                expected_time[0] += chunk_duration
                filler_count[0] += 1
        time.sleep(0.02)

ft = threading.Thread(target=filler, daemon=True)
ft.start()

print("\nRecording 8 seconds... PLAY SOMETHING NOW!")
for i in range(8):
    time.sleep(1)
    total_frames = len(frames)
    duration = total_frames * chunk_samples / rate
    peak = max(np.abs(f).max() for f in frames[-10:]) if frames else 0
    print(f"  {i+1}s - callbacks={callback_count[0]}, filler={filler_count[0]}, "
          f"chunks={total_frames}, dur={duration:.1f}s, recent_peak={peak:.4f}")

recording[0] = False
ft.join(timeout=3)
stream.stop_stream()
stream.close()
p.terminate()

# Save
audio_data = np.concatenate(frames)
peak = np.abs(audio_data).max()
duration = len(audio_data) / rate

# Resample to 16kHz
if rate != SAMPLE_RATE:
    ratio = SAMPLE_RATE / rate
    n_samples = int(len(audio_data) * ratio)
    indices = np.linspace(0, len(audio_data) - 1, n_samples)
    audio_data = np.interp(indices, np.arange(len(audio_data)), audio_data)

audio_data = np.clip(audio_data, -1.0, 1.0)
audio_int16 = (audio_data * 32767).astype(np.int16)

outpath = r"C:\Users\27204\MeetingRecordings\test_e2e.wav"
os.makedirs(os.path.dirname(outpath), exist_ok=True)
with wave.open(outpath, 'wb') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(SAMPLE_RATE)
    wf.writeframes(audio_int16.tobytes())

print(f"\nResult: {duration:.1f}s, peak={peak:.4f}")
print(f"Saved: {outpath}")
if peak > 0.01:
    print(">>> SUCCESS: Audio captured!")
else:
    print(">>> FAIL: No audio captured (was anything playing?)")
