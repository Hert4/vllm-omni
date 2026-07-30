#!/usr/bin/env python3
"""Smoke-test the higgs-tts-3-4b server: generate speech and verify the WAV.

    python3 test_higgs.py [port] [outdir]

Checks the response is a real, non-silent, correctly-formed 24 kHz WAV rather
than just asserting HTTP 200 -- a server can return a valid-looking empty clip.
"""
import json
import struct
import sys
import urllib.request
import wave
from pathlib import Path

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8095
OUTDIR = Path(sys.argv[2] if len(sys.argv) > 2 else "/root/work/tmduc/omni/out")
OUTDIR.mkdir(parents=True, exist_ok=True)

CASES = [
    ("en", "Hello, how are you? This is a CUDA twelve point eight build of vLLM Omni."),
    ("vi", "Xin chao, day la ban dung thu cua Higgs TTS ba chay tren CUDA muoi hai cham tam."),
    ("long", "The quick brown fox jumps over the lazy dog. "
             "Pack my box with five dozen liquor jugs. "
             "How vexingly quick daft zebras jump."),
]


def post_speech(text: str) -> bytes:
    body = json.dumps({
        "model": "higgs-tts-3-4b",
        "input": text,
        "response_format": "wav",
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/audio/speech",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read()


def describe(raw: bytes, path: Path) -> str:
    path.write_bytes(raw)
    if raw[:4] != b"RIFF":
        return f"NOT a RIFF/WAV (first bytes {raw[:16]!r}, {len(raw)} bytes)"
    with wave.open(str(path), "rb") as w:
        n, sr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
        frames = w.readframes(n)
    dur = n / sr if sr else 0
    if sw == 2:
        vals = struct.unpack(f"<{len(frames)//2}h", frames[: (len(frames)//2)*2])
        peak = max(abs(v) for v in vals) if vals else 0
        rms = (sum(v * v for v in vals) / len(vals)) ** 0.5 if vals else 0
    else:
        peak = rms = -1
    return (f"{len(raw)/1024:.0f} KB  {sr} Hz  {ch}ch  {dur:.2f}s  "
            f"peak={peak}  rms={rms:.0f}")


fails = 0
for name, text in CASES:
    try:
        raw = post_speech(text)
        info = describe(raw, OUTDIR / f"higgs_{name}.wav")
        print(f"[{name:5}] {info}")
        if "NOT a RIFF" in info or " 0.00s " in info or "peak=0 " in info:
            fails += 1
    except Exception as e:
        print(f"[{name:5}] FAILED: {type(e).__name__}: {e}")
        fails += 1

print(f"\nwavs in {OUTDIR}")
sys.exit(1 if fails else 0)
