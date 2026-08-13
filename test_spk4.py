#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test FunASR with non-large paraformer for better timestamp support"""
import sys, os, json
os.environ['PYTHONIOENCODING'] = 'utf-8'
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except:
    pass

ffdir = os.path.join(os.path.expanduser("~"), "ffmpeg-bin")
if os.path.exists(ffdir):
    os.environ['PATH'] = ffdir + os.pathsep + os.environ.get('PATH', '')

from funasr import AutoModel

print("Testing with paraformer (non-large) + spk_model...")
m = AutoModel(
    model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
    disable_update=True,
)

test_file = os.path.join(os.path.expanduser("~"), "MeetingRecordings", "meeting_20260719_204032.wav")
print(f"Testing on: {test_file}")

# Try with return_raw_text=True 
result = m.generate(input=test_file, batch_size_s=300)

if result:
    r = result[0]
    print(f"\nKeys: {list(r.keys())}")
    sentence_info = r.get('sentence_info', [])
    print(f"Sentence count: {len(sentence_info)}")
    
    # Check if any sentence has text
    has_text = sum(1 for si in sentence_info if si.get('text', ''))
    print(f"Sentences with text: {has_text}")
    
    # Try to match text with sentence_info by position
    full_text = r.get('text', '')
    print(f"Full text length: {len(full_text)}")
    print(f"Full text (first 300): {full_text[:300]}")
    
    # The text might be in timestamp field
    ts = r.get('timestamp', [])
    print(f"Timestamp entries: {len(ts)}")
    if ts:
        print(f"First timestamp: {ts[0]}")
