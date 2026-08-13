#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test FunASR speaker diarization on real recording"""
import sys, os, json
os.environ['PYTHONIOENCODING'] = 'utf-8'
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except:
    pass

# Set ffmpeg
ffdir = os.path.join(os.path.expanduser("~"), "ffmpeg-bin")
if os.path.exists(ffdir):
    os.environ['PATH'] = ffdir + os.pathsep + os.environ.get('PATH', '')

from funasr import AutoModel

print("Loading FunASR with speaker diarization...")
m = AutoModel(
    model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
    disable_update=True,
)
print("Model loaded!")

# Test on the 11-minute recording
test_file = os.path.join(os.path.expanduser("~"), "MeetingRecordings", "meeting_20260719_204032.wav")
print(f"\nTesting on: {test_file}")
result = m.generate(input=test_file, batch_size_s=300)

if result:
    r = result[0]
    print(f"Result keys: {list(r.keys())}")
    print(f"Text (first 300): {r.get('text', '')[:300]}")
    
    # Check sentence_info for speaker labels
    sentence_info = r.get('sentence_info', [])
    print(f"\nSentence info count: {len(sentence_info)}")
    if sentence_info:
        for i, si in enumerate(sentence_info[:5]):
            spk = si.get('spk', 'N/A')
            text = si.get('text', '')
            start = si.get('start', 0)
            end = si.get('end', 0)
            print(f"  [{i}] Speaker={spk}, Start={start}ms, End={end}ms: {text[:80]}")
    else:
        print("No sentence_info - checking timestamp...")
        ts = r.get('timestamp', [])
        print(f"Timestamp count: {len(ts)}")
        
    # Check if there's raw result with speaker info
    print(f"\nAll top-level keys: {list(r.keys())}")
else:
    print("No result!")
