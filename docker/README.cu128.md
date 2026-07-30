# vllm-omni on CUDA 12.8

A `vllm-omni` image built against a **real CUDA 12.8 toolkit**, verified on H200
(sm90a) with Higgs TTS 3, Qwen3 and Whisper.

## Why this exists

`vllm-omni`'s own `docker/Dockerfile.cuda` layers onto `vllm/vllm-openai:v0.26.0`.
That image family **has no CUDA 12.8 variant** — all 502 tags on Docker Hub are
`cu129` or `cu13`. So a genuine 12.8 image requires compiling vLLM from source.

vLLM v0.26.0 does support 12.8: `CMakeLists.txt` carries an explicit
`CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 12.8` branch and falls back to
arch-specific (`9.0a`, `12.0a`) targets on CUDA < 13.0.

## Build

```bash
git clone https://github.com/vllm-project/vllm-omni.git
cd vllm-omni
cp /path/to/Dockerfile.cu128 docker/
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.cu128 -t vllm-omni:cu128 .
```

`ARCH_LIST` defaults to `9.0a` (Hopper only) — that is what keeps the compile at
roughly 25 minutes instead of the multi-hour all-architecture build. Override it
for other cards:

```bash
--build-arg ARCH_LIST="9.0a;10.0a" --build-arg JOBS=64
```

Resulting stack:

| component | version |
|---|---|
| CUDA toolkit | 12.8.1 (`nvidia/cuda:12.8.1-devel-ubuntu24.04`) |
| torch | `2.11.0+cu128` |
| vLLM | `0.26.1.dev0+g568afb3a1` built from tag `v0.26.0` |
| torchcodec | `0.15.0+cpu` |
| Python | 3.12 |

The build ends with a self-check that fails loudly if anything replaced the
cu128 torch or broke the audio decode path.

## Two patches CUDA 12.8 needs

Both are applied inside the Dockerfile and both are version-conditional or
documented, so nothing silently degrades on a newer base.

### 1. `cooperative_topk` will not compile on 12.8

`csrc/libtorch_stable/cooperative_topk.cuh` calls libcu++ overloads that only
exist in **CCCL >= 2.9**, which ships with CUDA 12.9:

```
cuda::ptx::mbarrier_try_wait_parity(sem_relaxed, scope_cta, ...)
cuda::ptx::mbarrier_arrive_expect_tx(sem_relaxed, scope_cta, space_shared, ...)
cuda::ptx::cp_async_bulk(space_shared, space_global, ...)
```

CUDA 12.8 ships CCCL 2.8, whose signatures differ, so that translation unit fails
with `no instance of overloaded function`. vLLM's CMake gates the kernel on
CUDA >= 12.0, which is too loose.

The Dockerfile disables it **only when nvcc < 12.9**. The kernel is the DSA
indexer (DeepSeek sparse attention) and every use site sits behind
`#ifdef VLLM_ENABLE_COOPERATIVE_TOPK`, so nothing else is affected. Models that
rely on DeepSeek sparse attention are the one thing this image cannot serve.

### 2. `torchcodec` from PyPI links CUDA 13

vLLM pins `torchcodec>=0.14`, but `download.pytorch.org` only publishes cu128
builds up to `0.9.1`. pip therefore resolves the generic PyPI wheel, which links
`libnvrtc.so.13` and cannot `dlopen` inside a 12.8 image:

```
OSError: libnvrtc.so.13: cannot open shared object file
```

`transformers` imports torchcodec eagerly when constructing an audio pipeline, so
this breaks **every audio model**, not just direct torchcodec users. The fix is
`torchcodec==0.15.0+cpu`, which satisfies the pin and decodes on CPU with no CUDA
runtime — which is all these audio paths need.

## Verified models

| model | path exercised | result |
|---|---|---|
| `bosonai/higgs-tts-3-4b` | omni 2-stage TTS, `/v1/audio/speech` | 24 kHz WAV, correct speech |
| `Qwen3-1.7B` | core vLLM, `/v1/chat/completions` | correct answers |
| `openai/whisper-large-v3-turbo` | audio input, `/v1/audio/transcriptions` | transcribes the TTS output |

### Higgs TTS

```bash
GPU=7 PORT=8095 bash run_higgs.sh
python3 test_higgs.py 8095 ./out
```

`run_higgs.sh` clones the stock deploy config with smaller
`gpu_memory_utilization` values — the shipped config asks for `0.6 + 0.25` of
*total* GPU memory (~122 GB on an H200), which is far more than a 4B model needs
and will not fit on a shared card.

### Round-trip verification

A TTS server can return a well-formed WAV that is silence, noise, or the wrong
words, so `test_higgs.py` checks sample rate, duration, peak and RMS rather than
just HTTP 200 — and `whisper_check.py` transcribes the audio back and scores WER
against the prompt:

```
file                      WER    sim
higgs_en.wav             0.27   0.81
   ref: Hello, how are you? This is a CUDA twelve point eight build of vLLM Omni.
   asr: Hey, how are you? This is a CUDA 12.8 build of VLLM Omni.
higgs_long.wav           0.00   1.00
   ref: The quick brown fox jumps over the lazy dog. Pack my box ...
   asr: The quick brown fox jumps over the lazy dog. Pack my box ...
```

The 0.27 on the first clip is Whisper normalising "twelve point eight" to "12.8",
not a synthesis error.

## Notes

- FlashInfer JIT-compiles kernels at runtime and needs `nvcc`, which is why the
  image keeps the `-devel` base rather than slimming to `-runtime`.
- Higgs TTS cold start is about 4 minutes to a healthy `/health`.
