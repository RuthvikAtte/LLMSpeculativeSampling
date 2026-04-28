#!/bin/bash
#SBATCH --job-name=specdecode-qwen
#SBATCH --account=bgum-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=4
#SBATCH --mem=200g
#SBATCH --time=05:00:00
#SBATCH --output=/u/ratte/LLMSpeculativeSampling/logs/slurm_%j.out
#SBATCH --error=/u/ratte/LLMSpeculativeSampling/logs/slurm_%j.err
#SBATCH --mail-user=ruthvikatte24@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -euo pipefail

WORK_DIR="/u/ratte/LLMSpeculativeSampling"
MODEL_CACHE_DIR="/scratch/bgum/ratte/specreason_models"
# Allow resuming a previous run by passing RESULTS_DIR from outside, e.g.:
#   RESULTS_DIR=/path/to/existing sbatch job_experiment.sh
RESULTS_DIR="${RESULTS_DIR:-$WORK_DIR/results_$(date +%Y%m%d_%H%M%S)}"
export RESULTS_DIR

# ── Environment ───────────────────────────────────────────────────────────────
module purge
module load cuda/12.3 2>/dev/null || module load cuda/12.2 2>/dev/null || module load cuda 2>/dev/null || {
    echo "WARNING: No cuda module found — proceeding without it."
}

source /sw/external/python/anaconda3/etc/profile.d/conda.sh
conda activate specsampling

cd "$WORK_DIR"
mkdir -p "$RESULTS_DIR" logs

export HF_HOME="$MODEL_CACHE_DIR"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=fork

VLLM_SERVER_PORT=8000
export VLLM_SERVER_URL="http://localhost:${VLLM_SERVER_PORT}/v1"
export VLLM_SERVER_BASE="http://localhost:${VLLM_SERVER_PORT}"

nvidia-smi

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo "Checking model weights..."
python3 - <<'EOF'
from huggingface_hub import try_to_load_from_cache
import sys
models = [
    ("Qwen/Qwen3-32B",  "config.json"),
    ("Qwen/Qwen3-1.7B", "config.json"),
]
missing = []
for repo, filename in models:
    result = try_to_load_from_cache(repo_id=repo, filename=filename)
    if result is None or result == -1:
        missing.append(repo)
if missing:
    print(f"ERROR: weights not cached for: {missing}")
    print("On a login node with internet access, run:")
    for m in missing:
        print(f"  HF_HOME=/scratch/bgum/ratte/specreason_models huggingface-cli download {m}")
    sys.exit(1)
print("Model weights found.")
EOF

echo "All pre-flight checks passed."

# ── Server management ─────────────────────────────────────────────────────────
SERVER_PID=""

start_server() {
    local label="$1"; shift
    local log="$RESULTS_DIR/vllm_server_${label}.log"
    echo "=== Starting vLLM server: $label ==="
    vllm serve "$@" \
        --host 0.0.0.0 \
        --port "$VLLM_SERVER_PORT" \
        --dtype bfloat16 \
        --trust-remote-code \
        --tensor-parallel-size 4 \
        > "$log" 2>&1 &
    SERVER_PID=$!
    echo "Server PID=$SERVER_PID — waiting for health..."
    until curl -sf "http://localhost:${VLLM_SERVER_PORT}/health" > /dev/null; do
        sleep 5
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: vLLM server died during startup. Check $log"
            exit 1
        fi
    done
    echo "Server ready."
}

stop_server() {
    if [ -n "$SERVER_PID" ]; then
        echo "Stopping vLLM server (PID=$SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
        SERVER_PID=""
    fi
}

# Ensure server is always stopped on exit (covers errors and normal completion)
trap stop_server EXIT

# ── Run experiments ───────────────────────────────────────────────────────────

# Script 1: 32B standalone baseline
if [ -f "$RESULTS_DIR/run_32b_standalone_results.json" ]; then
    echo "=== Script 1: 32B standalone — already done, skipping ==="
else
    echo "=== Script 1: 32B standalone (baseline) ==="
    start_server "32b" Qwen/Qwen3-32B --gpu-memory-utilization 0.90
    python3 run_32b_standalone.py
    stop_server
fi

# Script 2: 1.7B standalone (draft model ceiling)
if [ -f "$RESULTS_DIR/run_2b_standalone_results.json" ]; then
    echo "=== Script 2: 1.7B standalone — already done, skipping ==="
else
    echo "=== Script 2: 1.7B standalone (draft ceiling) ==="
    start_server "1p7b" Qwen/Qwen3-1.7B --gpu-memory-utilization 0.50
    python3 run_2b_standalone.py
    stop_server
fi

# Script 3: Speculative decoding (32B target + 1.7B draft)
if [ -f "$RESULTS_DIR/run_specdecode_results.json" ]; then
    echo "=== Script 3: Speculative decoding — already done, skipping ==="
else
    echo "=== Script 3: Speculative decoding (32B + 1.7B draft) ==="
    start_server "specdecode" Qwen/Qwen3-32B \
        --gpu-memory-utilization 0.70 \
        --speculative-config "{\"method\":\"draft_model\",\"model\":\"Qwen/Qwen3-1.7B\",\"num_speculative_tokens\":5,\"draft_tensor_parallel_size\":4}"
    python3 run_specdecode.py
    stop_server
fi

echo "Done. Results in $RESULTS_DIR"
