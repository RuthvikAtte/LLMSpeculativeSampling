# vLLM Speculative Decoding — Perlmutter Setup

Three scripts benchmark Qwen3 inference on MathVision_MINI using vLLM's
native draft-model speculative decoding.

| Script | Description |
|---|---|
| `run_32b_standalone.py` | Target model (Qwen3-32B) only, no spec-decode |
| `run_2b_standalone.py` | Draft model (Qwen3-2B) only, no spec-decode |
| `run_specdecode.py` | Spec-decode: 32B target + 2B draft, k=5 |

Results are written to `results/<script_name>_results.json`.

---

## Prerequisites

```bash
module load conda
conda activate <your-env>
pip install vllm==0.11.0
```

The dataset file `mathvision_mini.json` must be present in the repo root
(already included).  To use a different dataset, replace the file; each
entry must have `"question"` and `"answer"` keys.

---

## SLURM job scripts

### run_32b_standalone.py

```bash
#!/bin/bash
#SBATCH --job-name=qwen32b-standalone
#SBATCH --account=<your_account>
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out

module load conda
conda activate <your-env>

cd $SLURM_SUBMIT_DIR
python run_32b_standalone.py
```

### run_2b_standalone.py

```bash
#!/bin/bash
#SBATCH --job-name=qwen2b-standalone
#SBATCH --account=<your_account>
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

module load conda
conda activate <your-env>

cd $SLURM_SUBMIT_DIR
python run_2b_standalone.py
```

### run_specdecode.py

```bash
#!/bin/bash
#SBATCH --job-name=qwen-specdecode
#SBATCH --account=<your_account>
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x_%j.out

module load conda
conda activate <your-env>

cd $SLURM_SUBMIT_DIR
python run_specdecode.py
```

Submit with:
```bash
sbatch job_32b.sh
sbatch job_2b.sh
sbatch job_specdecode.sh
```

---

## Environment variables (recommended)

```bash
export HF_HOME=/pscratch/sd/<user>/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
export TOKENIZERS_PARALLELISM=false
# Optional: avoid NCCL timeout on slow model loads
export NCCL_TIMEOUT=1800
```

Set these in your SLURM script before calling `python`.

---

## Key vLLM parameters

### Tensor parallelism
All three scripts use `tensor_parallel_size=4` for the target (32B) model.
The speculative script additionally sets `speculative_draft_tensor_parallel_size=1`
for the 2B draft model — the draft runs on a single GPU rank, which is
sufficient for a 2B model and avoids coordination overhead.

### Speculative config (`run_specdecode.py`)
```python
speculative_config={
    "method": "draft_model",
    "model": "Qwen/Qwen3-2B",
    "num_speculative_tokens": 5,
    "speculative_draft_tensor_parallel_size": 1,
    "draft_dtype": "bfloat16",
}
```
- `method="draft_model"` activates vLLM's native draft-model speculative path.
- `num_speculative_tokens=5` — the draft proposes 5 tokens per step; the
  target verifies them in one forward pass.
- Increase `num_speculative_tokens` to trade more draft-compute for potentially
  higher throughput when the acceptance rate (alpha) is high.

### Acceptance rate (alpha)
`run_specdecode.py` attempts to read per-request acceptance rate from
`output.metrics`.  vLLM exposes this field under slightly different attribute
names across versions; the script tries several known names and falls back to
`null` gracefully.  Aggregate alpha is also printed at the end.

---

## Tuning guidance

| alpha range | Interpretation | Action |
|---|---|---|
| > 0.8 | Draft is well-aligned | Increase `num_speculative_tokens` (try 8–10) |
| 0.5 – 0.8 | Moderate alignment | k=5 is reasonable |
| < 0.5 | Draft diverges often | Reduce k or switch draft model |

---

## Expected output

Each script prints a summary table like:

```
--- Summary Table ---
   Q   Latency (s)    Tokens     Tok/s
----------------------------------------
   0        42.31      8192     193.6
   1        38.07      7104     186.6
   ...
 ALL       240.15     45000     187.4
```

`run_specdecode.py` adds an `Alpha` column.
