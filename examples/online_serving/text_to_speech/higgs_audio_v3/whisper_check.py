#!/usr/bin/env python3
"""Round-trip check: transcribe the TTS output with Whisper and score it
against the text that was synthesised.

    python3 whisper_check.py <manifest.json> [model_dir]

manifest.json: [{"wav": "...", "text": "...", "lang": "en"}, ...]

A TTS server can return a well-formed WAV that is silence, noise, or the wrong
words. Comparing the ASR transcript back to the prompt is what actually shows
the audio carries the intended speech.
"""
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from transformers import pipeline

TARGET_SR = 16000


def load_wav(path: str):
    """Decode with soundfile and resample to 16 kHz.

    Deliberately avoids torchcodec: the PyPI torchcodec wheel links against
    libnvrtc.so.13 (CUDA 13) and cannot load inside a CUDA 12.8 image.
    """
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        n = int(round(len(audio) * TARGET_SR / sr))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n),
            np.arange(len(audio)),
            audio,
        ).astype("float32")
    return {"array": audio, "sampling_rate": TARGET_SR}

manifest = json.loads(Path(sys.argv[1]).read_text())
model_dir = sys.argv[2] if len(sys.argv) > 2 else "/models/whisper-large-v3-turbo"

device = 0 if torch.cuda.is_available() else -1
dtype = torch.float16 if device == 0 else torch.float32
print(f"whisper: {model_dir}  device={'cuda' if device==0 else 'cpu'}\n")

asr = pipeline(
    "automatic-speech-recognition",
    model=model_dir,
    torch_dtype=dtype,
    device=device,
)


def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


def wer(ref: str, hyp: str) -> float:
    r, h = norm(ref).split(), norm(hyp).split()
    if not r:
        return 1.0
    # Levenshtein over words
    prev = list(range(len(h) + 1))
    for i, rw in enumerate(r, 1):
        cur = [i]
        for j, hw in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rw != hw)))
        prev = cur
    return prev[-1] / len(r)


rows = []
for item in manifest:
    wav, ref = item["wav"], item["text"]
    kw = {"language": item["lang"]} if item.get("lang") else {}
    out = asr(load_wav(wav), generate_kwargs=kw)
    hyp = out["text"].strip()
    w = wer(ref, hyp)
    sim = SequenceMatcher(None, norm(ref), norm(hyp)).ratio()
    rows.append((Path(wav).name, w, sim, ref, hyp))

print(f"{'file':<22}{'WER':>7}{'sim':>7}")
print("-" * 60)
bad = 0
for name, w, sim, ref, hyp in rows:
    flag = "" if w <= 0.35 else "   <-- MISMATCH"
    if w > 0.35:
        bad += 1
    print(f"{name:<22}{w:>7.2f}{sim:>7.2f}{flag}")
    print(f"   ref: {ref}")
    print(f"   asr: {hyp}")

print(f"\n{len(rows)-bad}/{len(rows)} clips transcribe back to the prompt (WER<=0.35)")
sys.exit(1 if bad else 0)
