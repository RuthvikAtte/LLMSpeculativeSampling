# Experiment Parameters Reference

This document covers every configurable parameter in the speculative decoding
experiment — what each one does, why it was set to its current value, and what
happens if you change it.

---

## 1. Models

| Parameter | Value | Where set |
|---|---|---|
| Target model | `Qwen/Qwen3-32B` | top of each `run_*.py` |
| Draft model | `Qwen/Qwen3-1.7B` | top of `run_specdecode.py` |

**Target model** (`Qwen/Qwen3-32B`) is the large, authoritative model whose
output quality we care about. It is the latency bottleneck — each forward pass
takes ~200–400 ms at TP=4 on A40s.

**Draft model** (`Qwen/Qwen3-2B`) proposes token continuations cheaply. It must
share the same vocabulary as the target so the speculative decoding algorithm
can compare logits directly. Qwen3-2B and Qwen3-32B are both trained on the
same tokenizer, satisfying this requirement.

> For the new VL experiment the models are `Qwen/Qwen3-VL-32B-Thinking` (target)
> and `Qwen/Qwen3-VL-2B-Thinking` (draft), cached at
> `/scratch/bgum/ratte/hf_cache`.

---

## 2. vLLM Engine Parameters

These are passed to `LLM(...)` at startup.

| Parameter | Value | Scripts |
|---|---|---|
| `tensor_parallel_size` | `4` | all three |
| `dtype` | `"bfloat16"` | all three |
| `trust_remote_code` | `True` | all three |
| `disable_log_stats` | `False` | `run_specdecode.py` only |
| `draft_tensor_parallel_size` | `1` | `run_specdecode.py` only |

**`tensor_parallel_size=4`** — splits the target model weights across 4 GPUs.
Qwen3-32B is ~64 GB in BF16; 4× A40s (48 GB each) gives 192 GB total, so TP=4
is the minimum that fits. Reducing this to TP=2 would require A100 80 GB cards.

**`dtype="bfloat16"`** — uses BF16 instead of FP16 or FP32. BF16 has the same
dynamic range as FP32 (8-bit exponent) and is natively accelerated on A100/A40
Tensor Cores. It avoids the overflow risk of FP16 on very deep reasoning chains.

**`trust_remote_code=True`** — allows the model's own `modeling_*.py` to run.
Required for Qwen models that ship custom attention/RoPE implementations.

**`disable_log_stats=False`** (`run_specdecode.py` only) — re-enables vLLM's
internal statistics pipeline. By default offline `LLM` disables it to reduce
overhead, but the alpha tracker needs `SpecDecodingLogging.observe()` to be
called each engine step. Overhead is negligible for batch=1 generation.

**`draft_tensor_parallel_size=1`** — the 2B draft model fits on a single GPU
(~4 GB weights), so it does not need to be sharded. Keeping it on TP=1 avoids
synchronisation overhead between draft steps.

---

## 3. Speculative Decoding Config

Set inside `speculative_config={...}` in `run_specdecode.py`.

| Parameter | Value | Meaning |
|---|---|---|
| `method` | `"draft_model"` | vLLM V1 draft-model spec decode |
| `model` | `"Qwen/Qwen3-2B"` | the draft model |
| `num_speculative_tokens` (γ) | `5` | draft tokens proposed per round |
| `draft_tensor_parallel_size` | `1` | sharding for the draft model |

**`method="draft_model"`** selects vLLM's native speculative decoding path
(introduced in vLLM V1, requires ≥ 0.19.1). The alternative `"ngram"` method
uses n-gram matching instead of a separate model.

**`num_speculative_tokens=5` (γ)** — each round the draft model proposes γ=5
tokens and the target verifies all of them in one forward pass. The theoretical
speedup is `(1 − α^(γ+1)) / (1 − α)`. At α ≈ 0.72 and γ=5 this gives ≈ 2.9×.
Tuning guide:

| Observed α | Recommendation |
|---|---|
| > 0.80 | increase γ to 8–10 |
| 0.55 – 0.80 | γ=5 is appropriate |
| < 0.55 | reduce γ to 3, or use a larger draft model |

---

## 4. Sampling Parameters

Identical across all three scripts (`SAMPLING_PARAMS`).

| Parameter | Value | Effect |
|---|---|---|
| `temperature` | `1.0` | no logit scaling; raw softmax distribution |
| `top_p` | `0.95` | nucleus sampling — keeps the top 95% probability mass |
| `top_k` | `20` | hard cap at 20 candidates before top_p is applied |
| `presence_penalty` | `0.0` | disabled — no per-token presence penalty |
| `repetition_penalty` | `1.0` | disabled — 1.0 means no adjustment |
| `max_tokens` | `40960` | maximum tokens the model may generate per question |
| `seed` | `42` | fixed RNG seed for reproducibility |

**`temperature=1.0`** preserves the model's trained probability distribution
exactly. Values < 1 sharpen it (more greedy), values > 1 flatten it (more
random). For speculative decoding, the acceptance criterion in the Leviathan
et al. algorithm is defined with respect to the target's unmodified distribution,
so keeping temperature=1 makes α most meaningful.

**`top_p=0.95` + `top_k=20`** — both filters are applied in sequence (top_k
first, then top_p). They prevent low-probability tail tokens without changing
the relative probabilities of the kept tokens. This combination is standard for
long chain-of-thought generation.

**`max_tokens=40960`** — 40 K tokens allows a full extended reasoning chain.
Qwen3-32B's context window is 128 K; 40 K is a practical budget that covers
virtually all MathVision problems while keeping per-question latency under ~60 s
at TP=4.

**`seed=42`** — makes every run deterministic so latency differences between
the three scripts reflect only the decoding method, not sampling noise.

---

## 5. Prompt Format

```python
SYSTEM_PREFIX = "Please reason step by step, and put your final answer within \\boxed{}."

messages = [
    {"role": "system", "content": SYSTEM_PREFIX},
    {"role": "user",   "content": question},
]
prompt = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
)
```

The chat template is applied with `enable_thinking=True`, which leaves the
assistant turn open so Qwen3 generates a `<think>...</think>` block before its
final answer. Setting `enable_thinking=False` would instead pre-fill an empty
`<think></think>` block, instructing the model to skip reasoning entirely.

The system message tells the model to use chain-of-thought and wrap its final
answer in `\boxed{...}`, enabling automated answer extraction with a regex.

---

## 6. Dataset

| Property | Value |
|---|---|
| File | `mathvision_mini.json` |
| Questions | 6 |
| Format | `[{"question": "...", "answer": "..."}]` |
| Task type | Mathematical reasoning |
| Batch size | 1 (questions generated one at a time) |

Batch size 1 is intentional — it isolates per-question latency so the
speculative decoding speedup can be measured cleanly. Multi-question batching
would amortise memory bandwidth differently and make α harder to interpret.

---

## 7. Output Metrics

| Metric | Definition | Applies to |
|---|---|---|
| `latency_s` | Wall-clock seconds for `llm.generate()` | all |
| `tokens_generated` | Number of output token IDs | all |
| `tokens_per_second` | `tokens / latency` | all |
| `acceptance_rate` (α) | `accepted_tokens / drafted_tokens` | spec decode only |
| `mean_acceptance_length` (MAL) | `1 + accepted / rounds` | spec decode only |
| `theoretical_speedup` | `(1 − α^(γ+1)) / (1 − α)` | spec decode only |
| `per_pos_acceptance_rate[i]` | fraction of rounds where position i was accepted | spec decode only |

**MAL** includes the guaranteed bonus token vLLM emits at the end of every
accepted run, hence the `+1`. It equals the expected number of tokens produced
per target forward pass.

**Theoretical speedup** assumes the draft model costs nothing (c → 0 in the
paper). The observed speedup (`run_32b_standalone` latency / `run_specdecode`
latency) will be lower because the 2B model is not free.

---

## 8. SLURM Job Parameters (`job_experiment.sh`)

| Parameter | Value | Reason |
|---|---|---|
| `--account` | `bgum-delta-gpu` | Delta GPU allocation |
| `--partition` | `gpuA40x4` | 4× A40 nodes (48 GB VRAM each) |
| `--gpus-per-node` | `4` | required for TP=4 |
| `--cpus-per-task` | `16` | tokeniser + vLLM async threads |
| `--mem` | `200g` | model weights + KV cache in CPU RAM |
| `--time` | `08:00:00` | conservative budget; 3 scripts ≈ 2–5 h |
| `HF_HOME` | `/scratch/bgum/ratte/hf_cache` | scratch avoids `/u/` quota |
| `HF_HUB_OFFLINE=1` | set | prevents download attempts on compute node |
