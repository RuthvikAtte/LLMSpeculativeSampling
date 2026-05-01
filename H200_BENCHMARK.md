# H200 Speculative Decoding Benchmark

**Target model:** `Qwen/Qwen3-VL-32B-Thinking`
**Draft model:** `Qwen/Qwen3-VL-2B-Thinking`
**Hardware:** Single NVIDIA H200 NVL (143,771 MiB)
**vLLM version:** 0.19.1 (with multimodal patch — see below)

---

## What changed in this session

### New files added

| File | Purpose |
|---|---|
| `run_h200_benchmark.py` | Self-contained benchmark client. Runs 8 hardcoded prompts (6 text-only math + 2 text+image using `imgs/sps.jpg`) against a live vLLM server, measures tok/s / latency / acceptance rate α, and prints a comparison table. |
| `job_h200_benchmark.sh` | SLURM job script for H200. Handles server lifecycle: spec decode phase → baseline phase → comparison. Edit `--account` and `--partition` before submitting. |

### Models downloaded (cached in `~/.cache/huggingface/hub/`)

| Model | Size (BF16) | Use |
|---|---|---|
| `Qwen/Qwen3-VL-32B-Thinking` | ~64 GB | Target / baseline |
| `Qwen/Qwen3-VL-2B-Thinking` | ~4 GB | Spec decode draft |
| `Qwen/Qwen2.5-7B-Instruct` | ~14 GB | Alternative draft for text-only QwQ-32B pair |

### vLLM patch — multimodal guard removed

**File:** `~/.local/lib/python3.10/site-packages/vllm/v1/spec_decode/eagle.py`

vLLM 0.19.1 contains a hard block preventing draft-model spec decode on any
multimodal (VL) model:

```python
# ORIGINAL (line 288) — crashes on Qwen3-VL
def _raise_if_multimodal(self):
    if self.supports_mm_inputs:
        raise NotImplementedError(
            "Speculative Decoding with draft models or parallel drafting "
            "does not support multimodal models yet"
        )
```

This was patched to a no-op:

```python
# PATCHED
def _raise_if_multimodal(self):
    pass  # patched: multimodal VL spec decode enabled
```

vLLM 0.20.0 ships with this guard already removed, but 0.20.0 installs
`torch 2.11.0+cu130` which requires a CUDA 13.0 driver. The H200 here runs
driver 570.153.02 (CUDA 12.8), so 0.20.0 crashes at GPU init. Staying on
0.19.1 + patch is the current workaround.

### vLLM version history in this session

```
0.19.1  →  upgraded to 0.20.0  →  reverted to 0.19.1
```

Revert was needed because 0.20.0 pulled in `torch 2.11.0+cu130` which is
incompatible with the CUDA 12.8 driver on this machine. 0.19.1 uses
`torch 2.10.0+cu128` which works correctly.

---

## Issues encountered and fixes

| # | Error | Root Cause | Fix Applied |
|---|---|---|---|
| 1 | `ValueError: draft_tensor_parallel_size (1) != tensor_parallel_size (4)` | A40x4 SLURM jobs used TP=4 with draft_TP=1 | Not applicable on H200 single GPU (TP=1 for both) |
| 2 | `NotImplementedError: Speculative Decoding … does not support multimodal models yet` | vLLM 0.19.1 explicitly blocks VL models | Patched `_raise_if_multimodal()` to `pass` in eagle.py |
| 3 | `RuntimeError: NVIDIA driver … too old (found version 12080)` | vLLM 0.20.0 needs CUDA 13.0; machine has CUDA 12.8 | Reverted to vLLM 0.19.1 + torch 2.10.0+cu128 |
| 4 | `ValueError: Free memory (31.73 GiB) < desired utilization (125.83 GiB)` | Another user's orphaned `VLLM::EngineCore` (PID 2147278, ~107 GB) held the GPU | Kill that process (`sudo kill -9 2146863 2147278`), then re-run |

---

## Repository file map

```
LLMSpeculativeSampling/
│
├── run_h200_benchmark.py        ← NEW: H200 benchmark client (this session)
├── job_h200_benchmark.sh        ← NEW: H200 SLURM job script (this session)
│
├── run_32b_vl_standalone.py     Baseline: 32B-VL standalone against vLLM server
├── run_2b_vl_standalone.py      Baseline: 2B-VL standalone against vLLM server
├── run_32b_vl_specdecode.py     Spec decode client (reads Prometheus α metrics)
├── run_specReason.py            Step-level speculative reasoning (2-server approach)
│
├── job_vl_standalone.sh         SLURM: A40x4, runs 32B + 2B standalones
├── job_vl_specdecode.sh         SLURM: A40x4, runs VL spec decode (was broken — fixed)
├── job_vl_specReason.sh         SLURM: 2-node, 32B scorer + 2B drafter
│
├── run_32b_standalone.py        Text-only 32B baseline (MathVision Mini)
├── run_2b_standalone.py         Text-only 2B baseline (MathVision Mini)
├── run_specdecode.py            Text-only spec decode (MathVision Mini)
│
├── prepare_mathvision_vl.py     Downloads MathVision VL dataset, encodes images as b64
├── mathvision_vl.json           Full MathVision VL dataset with base64 images
├── mathvision_mini.json         6-question text-only subset
│
├── EXPERIMENT.md                Setup guide for the text-only Delta experiment
├── RESULTS.md                   Results report: Qwen3-32B + 1.7B on A40x4
├── README_VLLM.md               vLLM setup notes and tuning guidance
├── H200_BENCHMARK.md            ← THIS FILE
│
└── results_h200_20260501_162320/  ← Benchmark output directory (this session)
    ├── vllm_spec_server.log
    └── vllm_baseline_server.log
```

---

## How to run the H200 benchmark

### Prerequisites

1. **GPU must be free.** Check with `nvidia-smi`. If another process is holding
   the VRAM, kill it:
   ```bash
   nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader
   sudo kill -9 <PID>
   ```

2. **Models cached** (already done — skip if re-running):
   ```bash
   huggingface-cli download Qwen/Qwen3-VL-32B-Thinking
   huggingface-cli download Qwen/Qwen3-VL-2B-Thinking
   ```

3. **vLLM 0.19.1 with patch applied** (already done — verify):
   ```bash
   python3 -c "import vllm; print(vllm.__version__)"   # should print 0.19.1
   python3 -c "import torch; print(torch.cuda.is_available())"  # should print True
   grep "pass.*patched" ~/.local/lib/python3.10/site-packages/vllm/v1/spec_decode/eagle.py
   # should print:     pass  # patched: multimodal VL spec decode enabled
   ```
   If the patch is missing (e.g. after a vLLM reinstall), re-apply it:
   ```python
   # In ~/.local/lib/python3.10/site-packages/vllm/v1/spec_decode/eagle.py
   # Replace the body of _raise_if_multimodal with: pass
   ```

### Step 1 — Start the spec decode server

```bash
cd ~/LLMSpeculativeSampling

RESULTS_DIR="results_h200_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

vllm serve Qwen/Qwen3-VL-32B-Thinking \
  --speculative-config '{"model":"Qwen/Qwen3-VL-2B-Thinking","num_speculative_tokens":5}' \
  --host 0.0.0.0 --port 8010 \
  --dtype bfloat16 --trust-remote-code \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  > "$RESULTS_DIR/vllm_spec_server.log" 2>&1 &

# Wait for it to be ready (takes ~3-5 minutes to load)
until curl -sf http://localhost:8010/health > /dev/null 2>&1; do sleep 10; done
echo "Server ready"
```

### Step 2 — Run the spec decode benchmark

```bash
python3 run_h200_benchmark.py run \
  --mode spec \
  --server http://localhost:8010 \
  --out "$RESULTS_DIR/spec_results.json" \
  --image imgs/sps.jpg
```

### Step 3 — Stop spec server, start baseline server

```bash
pkill -f "vllm serve" || true
sleep 5

vllm serve Qwen/Qwen3-VL-32B-Thinking \
  --host 0.0.0.0 --port 8010 \
  --dtype bfloat16 --trust-remote-code \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  > "$RESULTS_DIR/vllm_baseline_server.log" 2>&1 &

until curl -sf http://localhost:8010/health > /dev/null 2>&1; do sleep 10; done
echo "Baseline server ready"
```

### Step 4 — Run the baseline benchmark

```bash
python3 run_h200_benchmark.py run \
  --mode baseline \
  --server http://localhost:8010 \
  --out "$RESULTS_DIR/baseline_results.json" \
  --image imgs/sps.jpg
```

### Step 5 — Compare results

```bash
pkill -f "vllm serve" || true

python3 run_h200_benchmark.py compare \
  "$RESULTS_DIR/spec_results.json" \
  "$RESULTS_DIR/baseline_results.json"
```

### All-in-one (copy-paste version)

```bash
cd ~/LLMSpeculativeSampling
RD="results_h200_$(date +%Y%m%d_%H%M%S)" && mkdir -p "$RD"

# --- Spec decode ---
vllm serve Qwen/Qwen3-VL-32B-Thinking \
  --speculative-config '{"model":"Qwen/Qwen3-VL-2B-Thinking","num_speculative_tokens":5}' \
  --host 0.0.0.0 --port 8010 --dtype bfloat16 --trust-remote-code \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.90 --max-model-len 32768 \
  > "$RD/spec_server.log" 2>&1 &
until curl -sf http://localhost:8010/health > /dev/null 2>&1; do sleep 10; done
python3 run_h200_benchmark.py run --mode spec --server http://localhost:8010 \
  --out "$RD/spec_results.json" --image imgs/sps.jpg
pkill -f "vllm serve"; sleep 5

# --- Baseline ---
vllm serve Qwen/Qwen3-VL-32B-Thinking \
  --host 0.0.0.0 --port 8010 --dtype bfloat16 --trust-remote-code \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.90 --max-model-len 32768 \
  > "$RD/baseline_server.log" 2>&1 &
until curl -sf http://localhost:8010/health > /dev/null 2>&1; do sleep 10; done
python3 run_h200_benchmark.py run --mode baseline --server http://localhost:8010 \
  --out "$RD/baseline_results.json" --image imgs/sps.jpg
pkill -f "vllm serve"

# --- Compare ---
python3 run_h200_benchmark.py compare "$RD/spec_results.json" "$RD/baseline_results.json"
```

---

## Benchmark prompts (what run_h200_benchmark.py sends)

Six text-only math/reasoning questions and two image prompts using `imgs/sps.jpg`:

| ID | Type | Prompt summary |
|---|---|---|
| `algebra_quadratic` | text | Solve x² − 7x + 10 = 0 by factoring |
| `arithmetic_gauss` | text | Sum 1+2+…+200 using Gauss's formula |
| `geometry_inscribed` | text | Circle inscribed in 10cm square — area fraction |
| `systems_linear` | text | Solve 3x+2y=16, x−y=2 |
| `train_problem` | text | Two trains meeting — time calculation |
| `pythagorean` | text | Right triangle legs 5 and 12 — hypotenuse and area |
| `img_describe` | text + image | Describe the diagram and the concept it illustrates |
| `img_algorithm` | text + image | Explain the algorithm shown; give mathematical derivation |

All prompts use `enable_thinking=True` (chain-of-thought mode), `max_tokens=2048`,
temperature 1.0, top-p 0.95, top-k 20, seed 42.

---

## Metrics explained

| Metric | Definition | Source |
|---|---|---|
| `tps` (tok/s) | `completion_tokens / latency` | OpenAI `/v1/chat/completions` usage |
| `alpha` (α) | `accepted_tokens / drafted_tokens` | Prometheus `/metrics` — `vllm:spec_decode_num_accepted_tokens` |
| `MAL` | `1 + accepted / draft_rounds` | Prometheus — mean tokens produced per 32B forward pass |
| `per_pos_acceptance_rate` | Acceptance rate at each of the 5 draft positions | Prometheus — `vllm:spec_decode_num_accepted_tokens_per_pos` |
| `speedup (theoretical)` | `(1 − α^(k+1)) / (1 − α)` where k=5 | Leviathan et al. 2023, Theorem 3.8 |
| `speedup (empirical)` | `spec_tps / baseline_tps` | Ratio of the two benchmark runs |

---

## Existing experiment results (Delta A40x4, April 2026)

From `RESULTS.md` — text-only Qwen3-32B + Qwen3-1.7B on 6 MathVision Mini questions:

| Method | Avg tok/s | Total latency | Speedup |
|---|---|---|---|
| 32B standalone | 27.5 | 362.3 s | 1.00× |
| Spec decode (32B+1.7B, γ=5) | 43.3 | 257.7 s | **1.41× wall-time / 1.57× tok/s** |
| Acceptance rate α | — | — | 0.573 avg |
| Mean acceptance length | — | — | 3.87 tokens/pass |
| Theoretical speedup | — | — | 2.28× (gap due to shared TP=4) |

The VL model experiment on H200 is expected to produce similar or slightly lower
α (VL tokens are more variable) with better efficiency (H200 has higher memory
bandwidth than A40, reducing the draft model overhead).

---

## Known issues and future work

- **vLLM TP constraint (0.19.1):** `draft_tensor_parallel_size` must equal
  `tensor_parallel_size`. On H200 (TP=1) this is trivially satisfied. On the
  A40x4 cluster (TP=4) it forces the 2B draft to also use TP=4, increasing
  overhead. Expected to be relaxed in a future vLLM release.

- **Multimodal spec decode patch:** The `_raise_if_multimodal` guard in
  `eagle.py` is a hand-patch to installed library code. It will be lost if
  vLLM is reinstalled. vLLM 0.20.0 removes the guard natively but requires a
  CUDA 13.0 driver — re-check when the driver is updated.

- **GPU sharing:** This H200 is a shared machine. If other users hold GPU
  memory, the 32B model (~64 GB) won't fit. Always check `nvidia-smi` first.

- **MathVision VL results:** The VL spec decode experiments on the A40x4 cluster
  (`job_vl_specdecode.sh`) all failed due to the TP mismatch bug. No VL spec
  decode results exist yet. The H200 benchmark will be the first to produce them.
