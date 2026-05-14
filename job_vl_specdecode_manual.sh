#!/bin/bash
#SBATCH --job-name=vl-specdecode-manual
#SBATCH --account=bgum-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
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
MODEL_CACHE_DIR="/scratch/bgum/ratte/hf_cache"
RESULTS_DIR="${RESULTS_DIR:-$WORK_DIR/results_vl_specdecode_manual_$(date +%Y%m%d_%H%M%S)}"
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

# ── Resolve node hostnames ────────────────────────────────────────────────────
NODE_ARRAY=($(scontrol show hostname "$SLURM_JOB_NODELIST"))
NODE1="${NODE_ARRAY[0]}"
NODE2="${NODE_ARRAY[1]}"
PORT_32B=8000
PORT_2B=8001

export VLLM_32B_URL="http://${NODE1}:${PORT_32B}/v1"
export VLLM_2B_URL="http://${NODE2}:${PORT_2B}/v1"

echo "Node layout:"
echo "  Node1 (32B target): $NODE1  →  $VLLM_32B_URL"
echo "  Node2 (2B draft):   $NODE2  →  $VLLM_2B_URL"

nvidia-smi

# ── Pre-flight checks ─────────────────────────────────────────────────────────
[ -f "$WORK_DIR/mathvision_vl.json" ] || {
    echo "ERROR: mathvision_vl.json not found."
    echo "Run on a login node first:  HF_HOME=$MODEL_CACHE_DIR python3 prepare_mathvision_vl.py"
    exit 1
}

echo "Checking model weights..."
python3 - <<'EOF'
from huggingface_hub import try_to_load_from_cache
import sys
models = [
    ("Qwen/Qwen3-VL-32B-Thinking", "config.json"),
    ("Qwen/Qwen3-VL-2B-Thinking",  "config.json"),
]
missing = []
for repo, filename in models:
    result = try_to_load_from_cache(repo_id=repo, filename=filename)
    if result is None or result == -1:
        missing.append(repo)
if missing:
    print(f"ERROR: weights not cached for: {missing}")
    for m in missing:
        print(f"  HF_HOME=/scratch/bgum/ratte/hf_cache huggingface-cli download {m}")
    sys.exit(1)
print("Model weights found.")
EOF

echo "All pre-flight checks passed."

# ── Server management ─────────────────────────────────────────────────────────
SERVER_32B_PID=""
SERVER_2B_PID=""

start_server_on_node() {
    local node="$1"
    local port="$2"
    local label="$3"
    local model="$4"
    local tp="$5"
    local mem_util="$6"
    local max_model_len="${7:-}"   # optional; empty = use model default
    local log="$RESULTS_DIR/vllm_server_${label}.log"
    local max_len_flag=""
    [ -n "$max_model_len" ] && max_len_flag="--max-model-len $max_model_len"

    echo "=== Starting $label on $node:$port (TP=$tp max_model_len=${max_model_len:-default}) ==="
    srun --nodes=1 --ntasks=1 --nodelist="$node" \
        bash -c "
            source /sw/external/python/anaconda3/etc/profile.d/conda.sh
            conda activate specsampling
            export HF_HOME=$MODEL_CACHE_DIR
            export HF_HUB_OFFLINE=1
            export VLLM_WORKER_MULTIPROC_METHOD=fork
            vllm serve $model \
                --host 0.0.0.0 \
                --port $port \
                --dtype bfloat16 \
                --trust-remote-code \
                --tensor-parallel-size $tp \
                --gpu-memory-utilization $mem_util \
                $max_len_flag
        " > "$log" 2>&1 &
}

stop_servers() {
    echo "Stopping vLLM servers..."
    [ -n "$SERVER_32B_PID" ] && { kill "$SERVER_32B_PID" 2>/dev/null || true; wait "$SERVER_32B_PID" 2>/dev/null || true; }
    [ -n "$SERVER_2B_PID"  ] && { kill "$SERVER_2B_PID"  2>/dev/null || true; wait "$SERVER_2B_PID"  2>/dev/null || true; }
}

wait_for_server() {
    local url="$1"
    local label="$2"
    local log="$3"
    local pid="$4"
    echo "Waiting for $label at $url ..."
    until curl -sf "${url%/v1}/health" > /dev/null; do
        sleep 5
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "ERROR: $label server died — see $log"
            exit 1
        fi
    done
    echo "$label ready."
}

trap stop_servers EXIT

# ── Run manual spec decode ────────────────────────────────────────────────────

if [ -f "$RESULTS_DIR/run_vl_specdecode_manual_results.json" ]; then
    echo "=== Already done, skipping ==="
else
    echo "=== Manual token-level spec decode (32B target + 2B draft, 2 nodes) ==="

    # 32B target on node 1 (TP=4)
    start_server_on_node "$NODE1" "$PORT_32B" "32b_target" \
        "Qwen/Qwen3-VL-32B-Thinking" 4 "0.90"
    SERVER_32B_PID=$!

    # 2B draft on node 2 (TP=1 — single GPU is enough)
    # max-model-len=131072: default 262144 needs 28 GiB KV cache but only 16 GiB available at gpu_util=0.50
    start_server_on_node "$NODE2" "$PORT_2B" "2b_draft" \
        "Qwen/Qwen3-VL-2B-Thinking" 1 "0.55" "131072"
    SERVER_2B_PID=$!

    wait_for_server "$VLLM_32B_URL" "32B target" \
        "$RESULTS_DIR/vllm_server_32b_target.log" "$SERVER_32B_PID"
    wait_for_server "$VLLM_2B_URL"  "2B draft" \
        "$RESULTS_DIR/vllm_server_2b_draft.log"   "$SERVER_2B_PID"

    python3 run_vl_specdecode_manual.py
    stop_servers
fi

echo "Done. Results in $RESULTS_DIR"
