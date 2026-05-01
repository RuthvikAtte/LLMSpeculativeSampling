#!/bin/bash
# Speculative decoding benchmark on a single H200 GPU (141 GB VRAM)
# Target: Qwen/Qwen3-VL-32B-Thinking
# Draft:  Qwen/Qwen3-VL-2B-Thinking   (num_speculative_tokens=5)
#
# Adjust the three SBATCH lines marked "EDIT" for your cluster.
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=h200-specdecode-bm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=240g
#SBATCH --time=03:00:00
#SBATCH --output=/home/ruthvik/LLMSpeculativeSampling/logs/slurm_%j.out
#SBATCH --error=/home/ruthvik/LLMSpeculativeSampling/logs/slurm_%j.err
# EDIT: set your account and H200 partition:
# #SBATCH --account=YOUR_ACCOUNT
# #SBATCH --partition=YOUR_H200_PARTITION   # e.g. gpuH200, gpu_h200, etc.
# #SBATCH --constraint=h200                 # if your cluster uses constraints

set -euo pipefail

WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
RESULTS_DIR="${RESULTS_DIR:-$WORK_DIR/results_h200_$(date +%Y%m%d_%H%M%S)}"
export RESULTS_DIR

# ── Environment ───────────────────────────────────────────────────────────────
# EDIT: activate the environment that has (or will have) vLLM + openai
# For the Delta/SLURM cluster with conda:
module purge
module load cuda/12.3 2>/dev/null || module load cuda/12.2 2>/dev/null || \
    module load cuda 2>/dev/null || echo "WARNING: no cuda module found"

source /sw/external/python/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate specsampling 2>/dev/null || true
# If using a venv instead:  source /path/to/venv/bin/activate

cd "$WORK_DIR"
mkdir -p "$RESULTS_DIR" logs

export HF_HOME="$MODEL_CACHE_DIR"
# Keep offline if weights are pre-cached; comment out to allow downloads:
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

VLLM_PORT=8000
export VLLM_SERVER_BASE="http://localhost:${VLLM_PORT}"

nvidia-smi

# ── Install / upgrade vLLM ────────────────────────────────────────────────────
# H200 requires vLLM >= 0.19.1 (draft_tensor_parallel_size constraint fix).
# NOTE: On H200 with TP=1, draft_tensor_parallel_size defaults to 1 and matches.
#       The crash seen on A40x4 (TP=4 vs draft_TP=1) does NOT apply here.
echo ""
echo "=== Installing / verifying vLLM ==="
pip install --upgrade "vllm>=0.19.1" --quiet
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"

# ── Pre-flight: verify model weights are cached ───────────────────────────────
echo ""
echo "=== Checking model weights ==="
python3 - <<'EOF'
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
    print("Download on a node with internet access:")
    for m in missing:
        print(f"  huggingface-cli download {m}")
    sys.exit(1)
print("Model weights found in cache.")
EOF

echo "Pre-flight checks passed."

# ── Server helpers ────────────────────────────────────────────────────────────
SERVER_PID=""

start_server() {
    local label="$1"; shift
    local log="$RESULTS_DIR/vllm_server_${label}.log"
    echo ""
    echo "=== Starting vLLM server: $label ==="
    # Single H200 (141 GB): 32B-BF16 ≈ 64 GB + 2B-BF16 ≈ 4 GB leaves ~73 GB for KV cache.
    # TP=1 for both target and draft — no tensor-parallel mismatch.
    vllm serve "$@" \
        --host 0.0.0.0 \
        --port "$VLLM_PORT" \
        --dtype bfloat16 \
        --trust-remote-code \
        --tensor-parallel-size 1 \
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

# ── Phase 1: Spec Decode ──────────────────────────────────────────────────────
SPEC_OUT="$RESULTS_DIR/spec_decode_results.json"

if [ -f "$SPEC_OUT" ]; then
    echo "=== Spec decode result already exists — skipping ==="
else
    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  PHASE 1 — Speculative Decode (k=5 draft tokens) ║"
    echo "╚══════════════════════════════════════════════════╝"
    # On a single H200 (TP=1), draft_tensor_parallel_size defaults to 1 automatically,
    # matching tensor_parallel_size=1.  No explicit draft_tensor_parallel_size needed.
    start_server "specdecode" Qwen/Qwen3-VL-32B-Thinking \
        --speculative-config \
        '{"model":"Qwen/Qwen3-VL-2B-Thinking","num_speculative_tokens":5}'

    python3 run_h200_benchmark.py run \
        --mode   spec \
        --server "http://localhost:${VLLM_PORT}" \
        --out    "$SPEC_OUT" \
        --image  imgs/sps.jpg

    stop_server
fi

# ── Phase 2: Baseline ─────────────────────────────────────────────────────────
BASELINE_OUT="$RESULTS_DIR/baseline_results.json"

if [ -f "$BASELINE_OUT" ]; then
    echo "=== Baseline result already exists — skipping ==="
else
    echo ""
    echo "╔══════════════════════════════════════════════════╗"
    echo "║  PHASE 2 — Baseline (32B standalone, no draft)   ║"
    echo "╚══════════════════════════════════════════════════╝"
    start_server "baseline" Qwen/Qwen3-VL-32B-Thinking

    python3 run_h200_benchmark.py run \
        --mode   baseline \
        --server "http://localhost:${VLLM_PORT}" \
        --out    "$BASELINE_OUT" \
        --image  imgs/sps.jpg

    stop_server
fi

# ── Comparison ────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  FINAL COMPARISON                                 ║"
echo "╚══════════════════════════════════════════════════╝"
python3 run_h200_benchmark.py compare "$SPEC_OUT" "$BASELINE_OUT"

echo ""
echo "Results directory: $RESULTS_DIR"
echo "  Spec decode : $SPEC_OUT"
echo "  Baseline    : $BASELINE_OUT"
