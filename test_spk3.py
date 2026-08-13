#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test FunASR speaker diarization - check sentence_info text"""
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

print("Loading FunASR with speaker diarization...")
m = AutoModel(
    model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
    spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
    disable_update=True,
)

test_file = os.path.join(os.path.expanduser("~"), "MeetingRecordings", "meeting_20260719_204032.wav")
print(f"Testing on: {test_file}")
result = m.generate(input=test_file, batch_size_s=300)

if result:
    r = result[0]
    sentence_info = r.get('sentence_info', [])
    print(f"\nTotal sentences: {len(sentence_info)}")
    
    # Group by speaker
    speakers = {}
    for si in sentence_info:
        spk = si.get('spk', -1)
        text = si.get('text', '')
        start = si.get('start', 0)
        end = si.get('end', 0)
        if spk not in speakers:
            speakers[spk] = []
        speakers[spk].append({'text': text, 'start': start, 'end': end})
    
    print(f"\nSpeakers found: {len(speakers)}")
    for spk_id in sorted(speakers.keys()):
        sents = speakers[spk_id]
        total_text = ''.join(s['text'] for s in sents)
        total_dur = sum(s['end'] - s['start'] for s in sents) / 1000
        print(f"\n  Speaker {spk_id}: {len(sents)} sentences, {total_dur:.1f}s total")
        print(f"    Text (first 100): {total_text[:100]}")
    
    # Show first 10 sentences with speaker labels
    print("\n\n=== First 10 sentences with speaker labels ===")
    for i, si in enumerate(sentence_info[:10]):
        spk = si.get('spk', -1)
        text = si.get('text', '')
        start = si.get('start', 0) / 1000
        end = si.get('end', 0) / 1000
        print(f"[{start:06.1f}-{end:06.1f}] Speaker{spk}: {text[:60]}")
