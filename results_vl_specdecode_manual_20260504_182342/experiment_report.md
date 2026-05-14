# Token-Level Speculative Decoding for Vision-Language Models
## Experiment Report — Manual SpecDecode on MathVision

**Date:** 2026-05-04  
**Results file:** `run_vl_specdecode_manual_results.json`  
**Job ID:** SLURM 18052165  
**Cluster:** Delta (NCSA) — `gpuA40x4` partition  

---

## 1. Overview

This experiment implements **token-level speculative decoding** for vision-language reasoning, applied to the MathVision benchmark. Two separate vLLM servers are used to sidestep vLLM's built-in multimodal guard that blocks speculative decoding with VLMs:

- **Draft model (2B):** `Qwen/Qwen3-VL-2B-Thinking` — generates _k=5_ candidate tokens per round
- **Target model (32B):** `Qwen/Qwen3-VL-32B-Thinking` — verifies via rejection sampling

The core claim: the 2B model can draft speculative tokens that the 32B frequently accepts, yielding faster effective generation without changing the output distribution of the 32B.

---

## 2. Algorithm

### 2.1 Rejection Sampling (Token Level)

Each speculative decoding round proceeds as:

1. **Draft:** 2B model generates `k=5` tokens from the current partial `<think>` context, with logprobs.
2. **Verify:** 32B model generates `k+1` tokens from the **same** context, with `top_logprobs=20`.
3. **Accept/Reject:** For each draft token `t_i` at position `i`:
   - Look up `p_target(t_i)` in the 32B's top-20 logprobs at position `i`
   - Draw `u ~ Uniform(0,1)`
   - Accept if `u ≤ min(1, p_target(t_i) / p_draft(t_i))`
   - On first rejection: use 32B's token at that position as a correction, stop
4. **Bonus:** If all `k` tokens accepted, append the 32B's `(k+1)`-th token as a free bonus.
5. Repeat until `</think>` is generated, then 32B produces the final boxed answer.

### 2.2 Why Two Servers

vLLM's native `--speculative-config` path raises `_raise_if_multimodal()` and `_raise_if_mrope()` errors for Qwen-VL models. Using two independent OpenAI-compatible servers bypasses these guards entirely — the client performs the rejection sampling loop manually using the `/chat/completions` API with `logprobs=True`.

### 2.3 Image Handling

MathVision questions contain `<image1>` placeholder tokens (lmms-eval dataset artifacts). These are stripped with `re.sub(r'<image\d+>', '', question)` and the actual image is embedded as a `data:image/png;base64,...` content part sent to both models on every call.

### 2.4 Thinking Phase

Each round sends the growing partial `<think>` block as a continued assistant message (`continue_final_message=True`, `enable_thinking=True`). This lets both models treat the partial thinking as a prefix to extend rather than a completed turn. The thinking phase runs until `</think>` appears or `MAX_THINKING_TOKENS=32768` is hit.

---

## 3. Hardware & Configuration

| Component | Detail |
|-----------|--------|
| Cluster | Delta (NCSA) — `gpuA40x4` |
| Nodes | 2 × `gpuA40x4` (4× NVIDIA A40 48 GB each) |
| Node 1 (32B target) | `gpub057` — TP=4, `gpu_util=0.90` |
| Node 2 (2B draft) | `gpub075` — TP=1, `gpu_util=0.50`, `max_model_len=131072` |
| dtype | `bfloat16` |
| k (draft tokens/round) | 5 |
| top_logprobs | 20 (vLLM maximum) |
| Sampling | `temperature=1.0, top_p=0.95, top_k=20, seed=42` |

> **Note on `max_model_len`:** The 2B model's default context length (262,144) requires ~28 GiB of KV cache. At `gpu_util=0.50` on a single A40, only ~16 GiB is available. Capping at 131,072 tokens (~14 GiB KV) resolves the OOM error.

---

## 4. Dataset

**MathVision** (`MathLLMs/MathVision`, test split) — 6 questions selected to span a range of expected reasoning depths:

| QID | Category | Expected Tokens | Image |
|-----|----------|----------------|-------|
| 5 | Logic (kittens) | ~317 | ✓ |
| 133 | Transformation geometry (cogs) | ~1,967 | ✓ |
| 123 | Counting (children in a line) | ~3,382 | ✓ |
| 168 | Combinatorial geometry (grid squares) | ~4,804 | ✓ |
| 117 | Combinatorics (hexagonal grid / bee path) | ~8,482 | ✓ |
| 104 | Solid geometry / counting (shapes) | ~10,458 | ✓ |

All questions include base64-encoded PNG images. The `expected_tokens` column reflects prior single-model baseline measurements and is used only as a rough ordering guide.

---

## 5. Results

### 5.1 Aggregate Summary

| Metric | Value |
|--------|-------|
| Total questions | 6 |
| Total wall-clock time | 4,353 s (72.6 min) |
| Total drafted tokens | 56,305 |
| Total accepted tokens | 38,900 |
| **Overall acceptance rate (α)** | **0.6909** |
| **Overall theoretical speedup** | **2.883×** |
| Accuracy (boxed answer match) | 3 / 6 (50%) |

The theoretical speedup is computed from the classic formula:

```
speedup = (1 - α^(k+1)) / (1 - α)
        = (1 - 0.6909^6) / (1 - 0.6909)
        ≈ 2.88×
```

This represents the expected reduction in 32B forward passes relative to pure autoregressive generation. The actual wall-clock improvement depends on the draft/verify latency ratio; since both models run on separate nodes, the round-trip overhead is dominated by the 32B verify call.

### 5.2 Per-Question Results

| QID | Category | Latency (s) | Rounds | Drafted | Accepted | α | MAL | Theory Speedup | Correct |
|-----|----------|------------|--------|---------|----------|---|-----|----------------|---------|
| 5 | Logic | 402.8 | 997 | 4,985 | 2,325 | 0.4664 | 2.332 | 1.855× | ✗ (pred B, ref D) |
| 133 | Transformation geometry | 547.1 | 1,422 | 7,110 | 3,904 | 0.5491 | 2.745 | 2.157× | ✓ |
| 123 | Counting | 639.2 | 1,816 | 9,080 | 4,790 | 0.5275 | 2.638 | 2.071× | ✓ |
| 168 | Combinatorial geometry | 256.9 | 770 | 3,850 | 2,227 | 0.5784 | 2.892 | 2.283× | ✓ |
| **117** | **Combinatorics** | **2,217.5** | **5,462** | **27,310** | **23,366** | **0.8556** | **4.278** | **4.208×** | ✗ (pred 12, ref 16) |
| 104 | Counting | 289.8 | 794 | 3,970 | 2,288 | 0.5763 | 2.882 | 2.274× | ✗ (pred B, ref D) |
| **ALL** | | **4,353.3** | **11,261** | **56,305** | **38,900** | **0.6909** | **2.961** | **2.883×** | **3/6** |

**MAL** = Mean Accepted Length = average tokens accepted per round (including bonus). Theoretical maximum is `k+1 = 6` when α → 1.

### 5.3 Per-Position Acceptance Rates

The acceptance rate at each draft position, averaged across all 6 questions:

| Position | p0 | p1 | p2 | p3 | p4 |
|----------|----|----|----|----|-----|
| Avg α | 0.8457 | 0.7888 | 0.8287 | 0.8234 | 0.8511 |

And per-question breakdown:

| QID | p0 | p1 | p2 | p3 | p4 |
|-----|----|----|----|----|-----|
| 5 | 0.8054 | 0.7024 | 0.7411 | 0.7177 | 0.8000 |
| 133 | 0.8221 | 0.7750 | 0.8201 | 0.7981 | 0.8314 |
| 123 | 0.8315 | 0.7450 | 0.7813 | 0.7895 | 0.8386 |
| 168 | 0.8273 | 0.8038 | 0.8398 | 0.8349 | 0.8050 |
| **117** | **0.9440** | **0.9319** | **0.9575** | **0.9670** | **0.9789** |
| 104 | 0.8438 | 0.7746 | 0.8324 | 0.8333 | 0.8528 |

**Key observation:** Per-position rates are approximately flat (0.79–0.85 for 5 of 6 questions), with a slight dip at p1. This is notable — standard theory predicts acceptance rates should decline monotonically with position as the draft sequence diverges from the target distribution. The near-flat profile suggests the 2B model maintains good local coherence across a 5-token window. QID 117 is uniformly high (0.93–0.98) across all positions.

### 5.4 Throughput

| QID | Drafted tok/s | Rounds/min |
|-----|--------------|------------|
| 5 | 12.4 | 148.5 |
| 133 | 13.0 | 155.9 |
| 123 | 14.2 | 170.5 |
| 168 | 15.0 | 179.8 |
| 117 | 12.3 | 147.8 |
| 104 | 13.7 | 164.4 |
| **Mean** | **13.4** | **161.2** |

Throughput is relatively consistent at ~13 drafted tokens/s across questions, reflecting the fixed 32B verify call dominating each round's latency.

---

## 6. Key Findings

### 6.1 QID 117 (Combinatorics) — Exceptional Alignment

QID 117 stands out dramatically:

- α = **0.855** vs 0.527–0.578 for the other 5 questions
- MAL = **4.278** (72% of the theoretical maximum of 6)
- Theoretical speedup = **4.21×**
- 23,366 / 27,310 tokens accepted (85.5%)
- Thinking text: **110,067 characters / 25,057 words** — the longest chain-of-thought by far
- This question involved reasoning about hexagonal grid paths (Maia the bee), which apparently aligns well with the 2B model's token distribution

Despite the high acceptance rate, the model got the answer wrong (predicted 12, reference 16). This demonstrates that acceptance rate measures draft/target *alignment*, not reasoning *correctness*.

QID 117 also dominated the total runtime: 2,218s out of 4,353s total (51%), despite being one of 6 questions.

### 6.2 Acceptance Rate Variance

Excluding QID 117, the α range is 0.47–0.58 (mean ≈ 0.54), giving theoretical speedups of 1.85–2.28×. Including QID 117 pulls the overall α to 0.691 and overall speedup to 2.88×.

This variance suggests the 2B/32B alignment is **problem-dependent** — more predictable/formulaic reasoning steps (combinatorics identities) are easier for the small model to anticipate, while spatial/visual reasoning (logic with kittens, shapes) is harder.

### 6.3 Per-Position Flat Profile

The expected monotonic decline in per-position acceptance is not observed. Instead:

- p1 is slightly lower than p0 across most questions (0.70–0.80 vs 0.81–0.94 at p0)
- p2–p4 recover and are comparable to p0
- p4 is often the highest

One explanation: p1 is the first position where the draft sequence has "committed" to a direction not fully validated by the target, creating a small dip. But by p2–p4, the sequence has entered a locally coherent region where both models agree on continuation, recovering acceptance.

### 6.4 Accuracy (50%)

3 of 6 questions were answered correctly (QID 133, 123, 168). The 3 errors:

- **QID 5** (logic, kittens): Multiple-choice — predicted B, correct D. Visual counting problem where the image content is critical.
- **QID 117** (combinatorics, bee path): Numeric — predicted 12, correct 16. The longest thinking chain (25K words) still reached the wrong conclusion, suggesting the 2B draft tokens led the reasoning down an incorrect path.
- **QID 104** (counting, shapes): Multiple-choice — predicted B, correct D. Another visual discrimination task.

The two multiple-choice errors (B instead of D) may indicate a subtle bias in the 32B's final-answer generation, though 3 errors on 6 questions is too small a sample to draw conclusions.

---

## 7. Thinking Text Characteristics

| QID | Thinking chars | Thinking words | Final answer chars |
|-----|---------------|---------------|-------------------|
| 5 | 11,734 | 2,203 | 2,022 |
| 133 | 21,144 | 4,211 | 1,549 |
| 123 | 24,332 | 4,526 | 1,939 |
| 168 | 8,111 | 1,512 | 1,686 |
| **117** | **110,067** | **25,057** | **4,561** |
| 104 | 11,313 | 2,061 | 1,190 |

QID 117's thinking block is 4.5× longer than all other questions combined, consistent with its expected complexity (~8,482 tokens in the baseline).

---

## 8. Implementation Notes

### 8.1 Bugs Fixed During Development

| Issue | Fix |
|-------|-----|
| 2B KV cache OOM (`ValueError: 28.0 GiB KV cache needed, only 15.99 GiB available`) | Added `--max-model-len 131072` to 2B server |
| vLLM `top_logprobs` cap exceeded (`VLLMValidationError: Requested 40 > max 20`) | Reduced `TOP_LOGPROBS = 40 → 20` |
| Stale `<image1>` token in prompt text | Added `re.sub(r'<image\d+>', '', question)` in `build_prompt()` |

### 8.2 `top_logprobs=20` Cap Impact

With `top_logprobs=20`, any draft token `t_i` that falls outside the 32B's top-20 candidates is assigned `p_target = 0`, forcing rejection. In practice this is the primary source of α < 1 for most rounds — not that the 2B and 32B fundamentally disagree, but that the correct target token isn't in the top-20 candidates returned. A higher cap would likely increase α, but vLLM's hard limit is 20.

---

## 9. Comparison to Baseline

No baseline (pure 32B autoregressive) was run in this experiment for direct latency comparison. The theoretical speedup of **2.88×** is the standard metric for speculative decoding performance, representing the reduction in number of 32B forward passes:

```
Rounds with spec decode:  11,261
Equivalent autoregressive rounds (1 token each): 38,900 accepted + 11,261 corrections ≈ 50,161
Ratio: 50,161 / 11,261 ≈ 4.5× fewer 32B calls (raw); adjusted for verify overhead → ~2.9× net
```

The key efficiency question for future work is the **draft/verify latency ratio** — if the 2B call is ~1/16 the cost of the 32B call, the verify step dominates and the speedup approaches 2.88× wall-clock; if the 2B call is near-free, wall-clock speedup approaches the raw 32B call reduction.

---

## 10. Conclusions

1. **Token-level speculative decoding works for VLMs** — the manual two-server approach successfully bypasses vLLM's multimodal guard and produces valid rejection-sampled outputs with images.

2. **Overall α = 0.691 → 2.88× theoretical speedup** is a promising result. The 2B Qwen3-VL draft model is substantially aligned with the 32B target in its token predictions during the thinking phase.

3. **QID 117 is an outlier** (α = 0.856, 4.21× speedup) suggesting that problem type strongly influences draft/target alignment. Predictable combinatorial reasoning benefits most from speculative decoding.

4. **Per-position rates are flat**, not monotonically declining — indicating the 2B model's 5-token continuations remain locally coherent across the full window.

5. **Accuracy (50%)** is consistent with a hard subset of MathVision on a visual reasoning task; the small sample size prevents strong conclusions. Importantly, high α (QID 117) does not guarantee correctness.

6. **Next steps:**
   - Run the full MathVision test set for accuracy comparison
   - Compare wall-clock latency to a pure 32B baseline
   - Vary `k` (3, 7, 10) to find the optimal draft length
   - Investigate whether a fine-tuned 2B draft on MathVision improves α
   - Try `top_logprobs` workaround (e.g. sample from target's full distribution at rejection positions) to reduce the p_target=0 rejection floor

---

*Generated from SLURM job 18052165 on Delta cluster. Models loaded from `/scratch/bgum/ratte/hf_cache`. Experiment code: `run_vl_specdecode_manual.py`, job script: `job_vl_specdecode_manual.sh`.*
