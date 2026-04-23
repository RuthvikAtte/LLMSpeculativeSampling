# Speculative Decoding Experiment — Setup & Run Guide

Benchmarks Qwen3-32B (target) against Qwen3-2B (draft) on MathVision_MINI
using vLLM's native draft-model speculative decoding.

---

## What this experiment does

Three scripts run sequentially, each writing a JSON to `results/`:

| Script | Purpose | Result file |
|---|---|---|
| `run_32b_standalone.py` | Target model alone (baseline) | `results/run_32b_standalone_results.json` |
| `run_2b_standalone.py` | Draft model alone (speed ceiling) | `results/run_2b_standalone_results.json` |
| `run_specdecode.py` | Speculative decoding (32B + 2B) | `results/run_specdecode_results.json` |

Each script runs all 6 MathVision_MINI questions one at a time (batch=1)
and records latency, tokens generated, tok/s, and — for the spec decode
script — acceptance rate α, mean acceptance length, per-position acceptance
rates, and the theoretical speedup predicted by the paper.

---

## 1. Environment setup (Delta @ UIUC)

### 1.1 Connect and request a GPU node

```bash
ssh <netid>@login.delta.ncsa.illinois.edu

# Interactive session — 4x A100 40GB, 2 hours
srun --account=<your_account> \
     --partition=gpuA100x4 \
     --nodes=1 \
     --ntasks-per-node=1 \
     --gpus-per-node=4 \
     --cpus-per-task=32 \
     --mem=240G \
     --time=02:00:00 \
     --pty bash
```

### 1.2 Load modules and activate your conda environment

```bash
module load anaconda3_gpu
conda activate <your-env>       # must have Python ≥ 3.10
```

### 1.3 Install vLLM

```bash
pip install "vllm>=0.19.1"
```

> **Version note:** vLLM < 0.19 does not support `method="draft_model"` in
> the V1 engine and will raise `NotImplementedError` on `run_specdecode.py`.
> The two standalone scripts work on any recent vLLM version.

### 1.4 Set HuggingFace cache to scratch (models are large)

```bash
export HF_HOME=/scratch/<your_account>/$USER/hf_cache
export HF_HUB_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
```

Add these to your `~/.bashrc` or your SLURM job script so they persist.

### 1.5 Clone the repo and enter it

```bash
git clone https://github.com/RuthvikAtte/LLMSpeculativeSampling.git
cd LLMSpeculativeSampling
```

---

## 2. Pre-download the models (do this once)

The models are large; download them to scratch before the timed run so
network latency doesn't skew your results.

```bash
python3 - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3-32B")
snapshot_download("Qwen/Qwen3-2B")
EOF
```

This takes ~10–20 minutes on Delta's scratch filesystem.
Qwen3-32B is ~65 GB; Qwen3-2B is ~5 GB.

---

## 3. Run the experiment

Run the scripts in order. Each one prints per-question results to stdout
and saves a JSON to `results/`.

```bash
mkdir -p results

# Baseline: target model only (~30–60 min for 6 questions at max_tokens=40960)
python3 run_32b_standalone.py

# Baseline: draft model only (~5–10 min)
python3 run_2b_standalone.py

# Speculative decoding: 32B + 2B (~20–40 min, depends on α)
python3 run_specdecode.py
```

### Expected stdout from `run_specdecode.py`

```
Loading target model: Qwen/Qwen3-32B
Draft model:          Qwen/Qwen3-2B  (num_spec_tokens=5)
Loaded 6 questions from mathvision_mini.json

[Q0] latency=41.32s  tokens=8192  tok/s=198.2  α=0.7241  MAL=4.620  theory_speedup=2.89x
     per-position: p0=0.891  p1=0.821  p2=0.751  p3=0.682  p4=0.601
[Q1] ...

--- Summary Table ---
  Q   Lat(s)    Toks    Tok/s        α    MAL   Theory
--------------------------------------------------------
  0    41.32    8192    198.2   0.7241  4.620   2.89x
  ...
ALL   ...
```

---

## 4. SLURM batch job (recommended for long runs)

Save as `job_experiment.sh` and submit with `sbatch job_experiment.sh`.

```bash
#!/bin/bash
#SBATCH --job-name=specdecode-qwen
#SBATCH --account=<your_account>
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load anaconda3_gpu
conda activate <your-env>

export HF_HOME=/scratch/<your_account>/$USER/hf_cache
export HF_HUB_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false

cd $SLURM_SUBMIT_DIR
mkdir -p results logs

echo "=== Script 1: 32B standalone ===" && python3 run_32b_standalone.py
echo "=== Script 2: 2B standalone ===" && python3 run_2b_standalone.py
echo "=== Script 3: Speculative decoding ===" && python3 run_specdecode.py

echo "Done. Results in results/"
```

```bash
mkdir -p logs
sbatch job_experiment.sh
squeue -u $USER          # check job status
```

---

## 5. Reading the results

### 5.1 File structure

Every result file has the same top-level shape:

```json
{
  "model": "Qwen/Qwen3-32B",
  "total_questions": 6,
  "total_latency_s": 248.1,
  "average_tokens_per_second": 187.4,
  "per_question": [
    {
      "question_idx": 0,
      "question": "A right triangle...",
      "reference_answer": "5",
      "generated_text": "<full chain-of-thought + \\boxed{5}>",
      "latency_s": 41.32,
      "tokens_generated": 8192,
      "tokens_per_second": 198.2
    },
    ...
  ]
}
```

`run_specdecode_results.json` adds these fields to each question:

```json
{
  "acceptance_rate": 0.7241,
  "mean_acceptance_length": 4.620,
  "theoretical_speedup": 2.89,
  "per_pos_acceptance_rate": [0.891, 0.821, 0.751, 0.682, 0.601]
}
```

### 5.2 What each metric means

| Metric | Definition | Good value |
|---|---|---|
| `acceptance_rate` (α) | Fraction of draft tokens accepted by target | > 0.6 |
| `mean_acceptance_length` (MAL) | Avg tokens produced per target forward pass, incl. bonus | > 3 |
| `theoretical_speedup` | `(1 - α^(γ+1)) / (1 - α)` from Theorem 3.8 of the paper | > 2x |
| `per_pos_acceptance_rate[i]` | Fraction of rounds where draft token at position i was accepted | Falls with i |

**Theoretical speedup** is the upper bound assuming the draft model costs
nothing (c → 0 in the paper's notation). The observed wall-clock speedup
(`run_32b_standalone` latency / `run_specdecode` latency) will be lower
because the 2B draft is not free.

### 5.3 Quick comparison script

```python
import json

r32 = json.load(open("results/run_32b_standalone_results.json"))
rsd = json.load(open("results/run_specdecode_results.json"))

for q32, qsd in zip(r32["per_question"], rsd["per_question"]):
    ratio = q32["latency_s"] / qsd["latency_s"]
    print(f"Q{q32['question_idx']}  observed speedup={ratio:.2f}x  "
          f"α={qsd['acceptance_rate']:.3f}  theory={qsd['theoretical_speedup']:.2f}x")
```

### 5.4 Tuning `num_speculative_tokens` (γ)

| α range | What it means | Suggested γ |
|---|---|---|
| > 0.80 | Draft closely tracks target | Increase to 8–10 |
| 0.55 – 0.80 | Good alignment | γ=5 is appropriate |
| < 0.55 | Draft diverges quickly | Reduce to 3, or try a larger draft |

Edit `NUM_SPEC_TOKENS` at the top of `run_specdecode.py` and re-run.
The `per_pos_acceptance_rate` list tells you exactly where acceptance drops
off: if `per_pos[3]` is already below 0.3, there is little benefit to
proposing a 5th token.

---

## 6. Evaluating answer correctness

Each result JSON stores `generated_text` (full chain-of-thought) and
`reference_answer` (from the dataset). To check correctness:

```python
import json, re

def extract_boxed(text):
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    return m.group(1).strip() if m else None

data = json.load(open("results/run_specdecode_results.json"))
for q in data["per_question"]:
    pred = extract_boxed(q["generated_text"])
    correct = pred == q["reference_answer"]
    print(f"Q{q['question_idx']}  ref={q['reference_answer']}  pred={pred}  {'✓' if correct else '✗'}")
```

---

## 7. Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `NotImplementedError: Draft model speculative decoding is not supported` | vLLM < 0.19 | `pip install "vllm>=0.19.1"` |
| `CUDA out of memory` on 32B | Not enough VRAM | Ensure TP=4 (`tensor_parallel_size=4`) and no other GPU jobs on the node |
| `acceptance_rate: null` in JSON | `disable_log_stats` is True | Already handled in `run_specdecode.py`; verify vLLM ≥ 0.19.1 |
| Model download fails | HF_HOME on slow filesystem | Point `HF_HOME` to `/scratch/...` (§1.4) |
| NCCL timeout during model load | Slow checkpoint sharding | `export NCCL_TIMEOUT=1800` before running |
| `per_pos_acceptance_rate` all zeros | Spec decode not actually running | Check that `speculative_config` is present in logs at startup |
