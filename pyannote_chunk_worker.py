"""Subprocess worker: run pyannote diarization + speaker embeddings on an audio chunk.

Usage: python pyannote_chunk_worker.py <request_json> <output_json>

request_json contains: wav_path, start_s, end_s, num_speakers, model_name, hf_token,
                       embed_model (optional), embed_step (optional)
output_json: {"tracks": [{"start","end","speaker","emb":[256 floats or null]}], ...}
             start/end are LOCAL to the chunk (parent adds the chunk offset).

Speaker embeddings (wespeaker resnet34, 256-d) are attached to every segment so the
PARENT can do global voiceprint clustering across chunks — this replaces the old
overlap-region label stitching that fragmented one person into many labels.

Progress is written to stderr (unbuffered) so the parent can monitor.
"""
import os, sys, json, gc, time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["TQDM_DISABLE"] = "1"  # suppress progress bars to avoid pipe issues

def log(msg):
    sys.stderr.write(f"[worker] {msg}\n")
    sys.stderr.flush()

def _segment_embeddings(emb_feat, seg_s, seg_e):
    """Average the sliding-window embedding frames whose centers fall in [s,e].

    emb_feat: pyannote SlidingWindowFeature with .data (N,D) and .sliding_window.
    Returns L2-normalized vector (list[float]) or None if no frame available.
    """
    import numpy as np
    data = np.asarray(emb_feat.data, dtype="float32")
    if data.ndim != 2 or data.shape[0] == 0:
        return None
    sw = emb_feat.sliding_window
    n = data.shape[0]
    centers = [sw.start + i * sw.step + sw.duration / 2.0 for i in range(n)]
    idx = [i for i, c in enumerate(centers) if seg_s <= c <= seg_e]
    if not idx:
        # segment shorter than one step / between frames: use nearest center
        mid = (seg_s + seg_e) / 2.0
        j = min(range(n), key=lambda i: abs(centers[i] - mid))
        idx = [j]
    v = data[idx].mean(axis=0)
    nrm = float(np.linalg.norm(v))
    if nrm < 1e-8:
        return None
    v = v / nrm
    return v.tolist()

def main():
    req_path = sys.argv[1]
    out_path = sys.argv[2]

    with open(req_path, "r", encoding="utf-8-sig") as f:
        req = json.load(f)

    wav_path = req["wav_path"]
    start_s = float(req.get("start_s", 0))
    end_s = float(req.get("end_s", 0))
    num_speakers = req.get("num_speakers")
    model_name = req.get("model_name", "pyannote/speaker-diarization-3.1")
    hf_token = req.get("hf_token", "")
    embed_model = req.get("embed_model", "pyannote/wespeaker-voxceleb-resnet34-LM")
    embed_step = float(req.get("embed_step", 1.0))  # coarser than default 0.5s to cut CPU

    import soundfile as sf
    import torch

    t0 = time.time()
    log(f"reading audio {start_s:.0f}-{end_s:.0f}s...")
    info = sf.info(wav_path)
    sr = info.samplerate
    sample_start = int(start_s * sr)
    sample_end = int(end_s * sr) if end_s > 0 else info.frames

    waveform_np, _ = sf.read(wav_path, dtype="float32", always_2d=True,
                             start=sample_start, frames=sample_end - sample_start)
    waveform = torch.from_numpy(waveform_np.T).contiguous()
    del waveform_np
    audio_in = {"waveform": waveform, "sample_rate": int(sr)}
    chunk_dur = waveform.shape[1] / sr
    log(f"audio loaded: {chunk_dur:.1f}s, took {time.time()-t0:.1f}s")

    log("loading pyannote pipeline...")
    t1 = time.time()
    from pyannote.audio import Pipeline
    kwargs = {}
    if hf_token:
        kwargs["token"] = hf_token
    pipeline = Pipeline.from_pretrained(model_name, **kwargs)
    pipeline.to(torch.device("cpu"))
    log(f"pipeline loaded in {time.time()-t1:.1f}s")

    call_kwargs = {}
    if num_speakers:
        call_kwargs["num_speakers"] = int(num_speakers)

    log("running diarization...")
    t2 = time.time()
    result = pipeline.apply(audio_in, **call_kwargs)
    log(f"diarization done in {time.time()-t2:.1f}s")

    annotation = getattr(result, "speaker_diarization", result)
    raw_tracks = []
    for segment, _, speaker in annotation.itertracks(yield_label=True):
        raw_tracks.append((float(segment.start), float(segment.end), str(speaker)))

    # ── speaker embeddings (best-effort; failure → emb=null, parent can fall back) ──
    emb_feat = None
    if embed_model:
        try:
            log("loading embedding model...")
            te = time.time()
            from pyannote.audio import Model
            from pyannote.audio.core.inference import Inference
            emb_model = Model.from_pretrained(embed_model, token=hf_token if hf_token else None)
            inference = Inference(emb_model, window="sliding", step=embed_step,
                                  device=torch.device("cpu"))
            log(f"embedding model loaded in {time.time()-te:.1f}s")
            t3 = time.time()
            emb_feat = inference(audio_in)
            log(f"embeddings extracted in {time.time()-t3:.1f}s "
                f"({emb_feat.data.shape[0]} frames)")
            del emb_model, inference
        except Exception as e:
            log(f"embedding extraction failed (will continue without emb): {e}")
            emb_feat = None

    tracks = []
    for s, e, spk in raw_tracks:
        emb = None
        if emb_feat is not None:
            try:
                emb = _segment_embeddings(emb_feat, s, e)
            except Exception as ex:
                log(f"seg emb failed {s:.1f}-{e:.1f}: {ex}")
                emb = None
        tracks.append({"start": s, "end": e, "speaker": spk, "emb": emb})

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"tracks": tracks}, f, ensure_ascii=False)

    n_with = sum(1 for t in tracks if t["emb"] is not None)
    log(f"wrote {len(tracks)} tracks ({n_with} with embeddings) to {out_path}")
    log(f"total time: {time.time()-t0:.1f}s")

    del waveform, audio_in, pipeline, result, annotation, raw_tracks
    if emb_feat is not None:
        del emb_feat
    gc.collect()

if __name__ == "__main__":
    main()
