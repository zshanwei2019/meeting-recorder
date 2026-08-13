#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Test FunASR speaker diarization model loading"""
import sys
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except:
    pass

print("Testing FunASR speaker model import...")
try:
    from funasr import AutoModel
    print("AutoModel imported OK")
    
    # Test loading with spk_model
    print("Loading model with speaker diarization...")
    m = AutoModel(
        model="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
        punc_model="iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
    )
    print("Model with speaker diarization loaded OK!")
    
    # Test on a sample file
    import glob
    rec_dir = os.path.join(os.path.expanduser("~"), "MeetingRecordings")
    wavs = glob.glob(os.path.join(rec_dir, "*.wav"))
    if wavs:
        test_file = wavs[-1]
        print(f"Testing on: {test_file}")
        result = m.generate(input=test_file, batch_size_s=300)
        if result:
            print(f"Result keys: {result[0].keys()}")
            # Check for speaker info
            if 'sentence_info' in result[0]:
                for si in result[0]['sentence_info'][:3]:
                    spk = si.get('spk', 'N/A')
                    text = si.get('text', '')
                    print(f"  Speaker={spk}: {text[:50]}")
            text = result[0].get('text', '')
            print(f"Full text (first 200): {text[:200]}")
    else:
        print("No wav files found for testing")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
