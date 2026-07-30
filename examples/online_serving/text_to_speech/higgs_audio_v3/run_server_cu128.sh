#!/bin/bash
# Serve bosonai/higgs-tts-3-4b on the cu128 vllm-omni image.
#
#   GPU=3 PORT=8095 bash run_higgs.sh
#
# The stock deploy config asks for 0.6 + 0.25 of *total* GPU memory (~122 GB on
# an H200). This box is shared and every card already has a tenant, so we clone
# the config with much smaller fractions -- a 4B model needs nowhere near that.
set -euo pipefail

GPU="${GPU:-3}"
PORT="${PORT:-8095}"
IMAGE="${IMAGE:-vllm-omni:cu128}"
NAME="${NAME:-higgs-cu128}"
MODEL_DIR="${MODEL_DIR:-/root/work/models/higgs-tts-3-4b}"
REPO_DIR="${REPO_DIR:-/root/work/tmduc/omni/vllm-omni}"
S0_UTIL="${S0_UTIL:-0.22}"
S1_UTIL="${S1_UTIL:-0.08}"
WORK=/root/work/tmduc/omni

free=$(nvidia-smi --id="$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
total=$(nvidia-smi --id="$GPU" --query-gpu=memory.total --format=csv,noheader,nounits)
need=$(awk "BEGIN{printf \"%d\", ($S0_UTIL+$S1_UTIL)*$total}")
echo "GPU $GPU: ${free} MiB free, this run wants ~${need} MiB"
if [ "$free" -lt "$need" ]; then
    echo "not enough free memory on GPU $GPU -- pick another card" >&2
    exit 1
fi

# Clone the deploy config with reduced memory fractions.
CFG="$WORK/higgs_deploy_small.yaml"
python3 - "$REPO_DIR/vllm_omni/deploy/higgs_multimodal_qwen3.yaml" "$CFG" \
         "$S0_UTIL" "$S1_UTIL" <<'PY'
import re, sys
src, dst, s0, s1 = sys.argv[1:5]
text = open(src).read()
vals = iter([s0, s1])
text = re.sub(r'(gpu_memory_utilization:\s*)([0-9.]+)',
              lambda m: m.group(1) + next(vals), text)
open(dst, 'w').write(text)
print("gpu_memory_utilization now:", re.findall(r'gpu_memory_utilization:\s*[0-9.]+', text))
PY

docker rm -f "$NAME" >/dev/null 2>&1 || true

docker run -d --name "$NAME" \
    --gpus "\"device=${GPU}\"" \
    --ipc=host \
    -p "${PORT}:${PORT}" \
    -v "${MODEL_DIR}:/models/higgs-tts-3-4b:ro" \
    -v "${CFG}:/tmp/higgs_deploy.yaml:ro" \
    -v "${WORK}/out:/out" \
    -e VLLM_USE_DEEP_GEMM=0 \
    -e VLLM_MOE_USE_DEEP_GEMM=0 \
    -e HF_HUB_OFFLINE=1 \
    -w /opt/vllm-omni \
    "$IMAGE" \
    bash -lc "vllm-omni serve /models/higgs-tts-3-4b \
        --deploy-config /tmp/higgs_deploy.yaml \
        --served-model-name higgs-tts-3-4b \
        --host 0.0.0.0 --port ${PORT} \
        --trust-remote-code --omni"

echo "container ${NAME} on GPU ${GPU}, port ${PORT}"
