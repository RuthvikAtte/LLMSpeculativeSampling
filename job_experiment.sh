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
# If not set, a new timestamped directory is created.
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
export VLLM_LOGGING_LEVEL=DEBUG
export VLLM_WORKER_MULTIPROC_METHOD=fork

nvidia-smi

# ── Pre-flight checks ─────────────────────────────────────────────────────────

# 1. Verify model weights are cached
echo "Checking model weights..."
python - <<'EOF'
from huggingface_hub import try_to_load_from_cache
import sys
models = [
    ("Qwen/Qwen3-32B", "config.json"),
    ("Qwen/Qwen3-2B",  "config.json"),
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

# ── Run experiment ────────────────────────────────────────────────────────────
run_if_missing() {
    local result_file="$1"
    local label="$2"
    shift 2
    if [ -f "$RESULTS_DIR/$result_file" ]; then
        echo "=== $label — already done, skipping ==="
    else
        echo "=== $label ==="
        "$@"
    fi
}

run_if_missing run_32b_standalone_results.json  "Script 1: 32B standalone (baseline)"   python3 run_32b_standalone.py
run_if_missing run_2b_standalone_results.json   "Script 2: 2B standalone (draft ceiling)" python3 run_2b_standalone.py
run_if_missing run_specdecode_results.json      "Script 3: Speculative decoding (32B+2B)" python3 run_specdecode.py

echo "Done. Results in $RESULTS_DIR"
