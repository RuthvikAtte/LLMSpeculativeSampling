#!/bin/bash
# Speculative decoding benchmark on 4x A100 GPUs
# Target: Qwen/Qwen3-VL-32B-Thinking
# Draft:  Qwen/Qwen3-VL-2B-Thinking   (num_speculative_tokens=5)
#
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=a100-specdecode-bm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=240g
#SBATCH --time=03:00:00
#SBATCH --output=/u/ratte/LLMSpeculativeSampling/logs/slurm_%j.out
#SBATCH --error=/u/ratte/LLMSpeculativeSampling/logs/slurm_%j.err
#SBATCH --account=bgum-delta-gpu
#SBATCH --partition=gpuA100x4

set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_CACHE_DIR="${HF_HOME:-/scratch/bgum/ratte/hf_cache}"
RESULTS_DIR="${RESULTS_DIR:-$WORK_DIR/results_a100_$(date +%Y%m%d_%H%M%S)}"
export RESULTS_DIR

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load cuda/12.3 2>/dev/null || module load cuda/12.2 2>/dev/null || \
    module load cuda 2>/dev/null || echo "WARNING: no cuda module found"

source /sw/external/python/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate specsampling 2>/dev/null || true

cd "$WORK_DIR"
mkdir -p "$RESULTS_DIR" logs

export HF_HOME="$MODEL_CACHE_DIR"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=fork

VLLM_PORT=8000
export VLLM_SERVER_BASE="http://localhost:${VLLM_PORT}"

nvidia-smi

echo ""
echo "=== Installing / verifying vLLM ==="
pip install --upgrade "vllm>=0.19.1" --quiet
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

echo ""
echo "=== Checking model weights ==="
python3 - <<'PY_EOF'
from huggingface_hub import try_to_load_from_cache
import sys
models = [
    ("Qwen/Qwen3-VL-32B-Thinking", "config.json"),
    ("Qwen/Qwen3-VL-2B-Thinking",  "config.json"),
]
missing = []
for repo, fname in models:
    result = try_to_load_from_cache(repo_id=repo, filename=fname)
    if result is None or result == -1:
        missing.append(repo)
if missing:
    print(f"ERROR: weights not cached for: {missing}")
    sys.exit(1)
print("Model weights found in cache.")
PY_EOF

SERVER_PID=""

start_server() {
    local label="$1"; shift
    local log="$RESULTS_DIR/vllm_server_${label}.log"
    echo ""
    echo "=== Starting vLLM server: $label ==="
    vllm serve "$@" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT" \
        --dtype bfloat16 \
        --trust-remote-code \
        --tensor-parallel-size 4 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 32768 \
        > "$log" 2>&1 &
    SERVER_PID=$!
    echo "Server PID=$SERVER_PID | log: $log"
    echo "Waiting for /health..."
    until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null; do
        sleep 5
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: vLLM server died — last 60 lines of $log:"
            tail -60 "$log" || true
            exit 1
        fi
    done
    echo "Server ready."
}

stop_server() {
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping server (PID=$SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}

trap stop_server EXIT

SPEC_OUT="$RESULTS_DIR/spec_decode_results.json"

if [ -f "$SPEC_OUT" ]; then
    echo "=== Spec decode result already exists — skipping ==="
else
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  PHASE 1 — Speculative Decode (k=5 draft tokens) ║"
    echo "╚══════════════════════════════════════════════════╝"
    start_server "specdecode" Qwen/Qwen3-VL-32B-Thinking \
        --speculative-config '{"model":"Qwen/Qwen3-VL-2B-Thinking","num_speculative_tokens":5,"draft_tensor_parallel_size":4, "method":"draft_model"}'

    python3 run_mathvision_benchmark.py run \
        --mode   spec \
        --server "http://localhost:${VLLM_PORT}" \
        --out    "$SPEC_OUT" \
        --dataset mathvision_vl.json


    stop_server
    sleep 5
fi

BASELINE_OUT="$RESULTS_DIR/baseline_results.json"

if [ -f "$BASELINE_OUT" ]; then
    echo "=== Baseline result already exists — skipping ==="
else
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  PHASE 2 — Baseline (32B standalone, no draft)   ║"
    echo "╚══════════════════════════════════════════════════╝"
    start_server "baseline" Qwen/Qwen3-VL-32B-Thinking

    python3 run_mathvision_benchmark.py run \
        --mode   baseline \
        --server "http://localhost:${VLLM_PORT}" \
        --out    "$BASELINE_OUT" \
        --dataset mathvision_vl.json

    stop_server
    sleep 5
fi

echo "╔══════════════════════════════════════════════════╗"
echo "║  FINAL COMPARISON                                 ║"
echo "╚══════════════════════════════════════════════════╝"
if [ -f "$SPEC_OUT" ] && [ -f "$BASELINE_OUT" ]; then
    python3 run_mathvision_benchmark.py compare "$SPEC_OUT" "$BASELINE_OUT" > "$RESULTS_DIR/comparison.txt"
    cat "$RESULTS_DIR/comparison.txt"
fi

echo "Results directory: $RESULTS_DIR"
