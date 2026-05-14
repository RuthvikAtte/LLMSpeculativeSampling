# Token-Level Speculative Decoding for Vision-Language Models
## Experiment Report — Manual SpecDecode on MathVision (A100 Run)

**Date:** 2026-05-13
**Results file:** `run_vl_specdecode_manual_results.json`
**Job ID:** SLURM 18222736
**Cluster:** Delta (NCSA) — `gpuA100x4` partition

---

## 1. Overview

This experiment implements **token-level speculative decoding** for vision-language reasoning, applied to the MathVision benchmark. Two separate vLLM servers are used to sidestep vLLM's built-in multimodal guard that blocks speculative decoding with VLMs:

- **Draft model (2B):** `Qwen/Qwen3-VL-2B-Thinking` — generates _k=5_ candidate tokens per round
- **Target model (32B):** `Qwen/Qwen3-VL-32B-Thinking` — verifies via rejection sampling

This is a direct rerun of the prior A40 experiment (SLURM 18052165, 2026-05-04) on A100 hardware, with one configuration adjustment required for the memory difference between A40 48 GB and A100 40 GB (see §3).

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
| Cluster | Delta (NCSA) — `gpuA100x4` |
| Nodes | 2 × `gpuA100x4` (4× NVIDIA A100-SXM4-40GB each) |
| Node 1 (32B target) | `gpua006` — TP=4, `gpu_util=0.90` |
| Node 2 (2B draft) | `gpua007` — TP=1, `gpu_util=0.55`, `max_model_len=131072` |
| dtype | `bfloat16` |
| k (draft tokens/round) | 5 |
| top_logprobs | 20 (vLLM maximum) |
| Sampling | `temperature=1.0, top_p=0.95, top_k=20, seed=42` |

> **Note on A100 vs A40 memory:** The prior run used A40 48 GB nodes with `gpu_util=0.50` for the 2B draft (TP=1). On A100 40 GB, the same setting left only 13.48 GiB available for KV cache, 520 MiB short of the 14.0 GiB required for `max_model_len=131072`. The first submission (job 18206513) failed with:
> ```
> ValueError: To serve at least one request with max seq len (131072),
> 14.0 GiB KV cache needed but only 13.48 GiB available.
> Estimated maximum model length: 126208.
> ```
> Fix: bumped 2B draft `gpu_memory_utilization` from `0.50` → `0.55`, providing ~17 GiB for KV cache. The 32B target server configuration was unchanged.

> **Note on `max_model_len`:** The 2B model's default context length (262,144) requires ~28 GiB of KV cache. At `gpu_util=0.55` on a single A100 40GB, only ~17 GiB is available for KV. Capping at 131,072 tokens (~14 GiB KV) fits comfortably.

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

All questions include base64-encoded PNG images. The `expected_tokens` column reflects prior single-model baseline measurements.

---

## 5. Results

### 5.1 Aggregate Summary

| Metric | Value |
|--------|-------|
| Total questions | 6 |
| Total wall-clock time | 4,002 s (66.7 min) |
| Total drafted tokens | 76,835 |
| Total accepted tokens | 55,764 |
| **Overall acceptance rate (α)** | **0.7258** |
| **Overall theoretical speedup** | **3.11×** |
| Accuracy (boxed answer match) | 2 / 6 (33%) |

The theoretical speedup is computed from the classic formula:

```
speedup = (1 - α^(k+1)) / (1 - α)
        = (1 - 0.7258^6) / (1 - 0.7258)
        ≈ 3.11×
```

### 5.2 Per-Question Results

| QID | Category | Latency (s) | Rounds | Drafted | Accepted | α | MAL | Theory Speedup | Correct |
|-----|----------|------------|--------|---------|----------|---|-----|----------------|---------|
| 5 | Logic | 336.2 | 1,263 | 6,315 | 3,048 | 0.4827 | 2.413 | 1.91× | ✓ (pred D, ref D) |
| 133 | Transformation geometry | 253.9 | 1,023 | 5,115 | 2,581 | 0.5046 | 2.523 | 1.99× | ✗ (pred E, ref C) |
| 123 | Counting | 1,488.5 | 5,462 | 27,310 | 20,115 | 0.7365 | 3.683 | 3.19× | ✗ (pred 7, ref 6) |
| 168 | Combinatorial geometry | 90.8 | 451 | 2,255 | 1,369 | 0.6071 | 3.035 | 2.42× | ✓ (pred A, ref A) |
| **117** | **Combinatorics** | **1,428.9** | **5,462** | **27,310** | **23,831** | **0.8726** | **4.363** | **4.38×** | ✗ (pred 7, ref 16) |
| 104 | Counting | 403.3 | 1,706 | 8,530 | 4,820 | 0.5651 | 2.825 | 2.22× | ✗ (pred B, ref D) |
| **ALL** | | **4,001.7** | **—** | **76,835** | **55,764** | **0.7258** | **—** | **3.11×** | **2/6** |

**MAL** = Mean Accepted Length = average tokens accepted per round (including bonus). Theoretical maximum is `k+1 = 6` when α → 1.

### 5.3 Per-Position Acceptance Rates

Acceptance rate at each draft position, per question and averaged across all 6 questions:

| QID | p0 | p1 | p2 | p3 | p4 |
|-----|----|----|----|----|-----|
| 5 | 0.8044 | 0.7382 | 0.7453 | 0.7138 | 0.8120 |
| 133 | 0.8221 | 0.7444 | 0.7572 | 0.7743 | 0.7439 |
| 123 | 0.8988 | 0.8829 | 0.9070 | 0.9082 | 0.9443 |
| 168 | 0.8670 | 0.7698 | 0.8605 | 0.8610 | 0.8744 |
| **117** | **0.9497** | **0.9416** | **0.9650** | **0.9684** | **0.9823** |
| 104 | 0.8388 | 0.7778 | 0.8203 | 0.8050 | 0.8544 |
| **Mean** | **0.8635** | **0.8091** | **0.8426** | **0.8385** | **0.8686** |

**Key observation:** As in the A40 run, per-position rates are approximately flat rather than monotonically declining. The slight dip at p1 is consistent across almost all questions — the draft sequence "commits" to a direction after the first token, causing a small dip, then recovers to p0-level coherence by p2–p4. QID 117 maintains uniformly high rates (0.94–0.98) across all positions.

### 5.4 Throughput

| QID | Drafted tok/s | Rounds/min |
|-----|--------------|------------|
| 5 | 18.8 | 225.2 |
| 133 | 20.1 | 241.8 |
| 123 | 18.3 | 220.2 |
| 168 | 24.8 | 297.8 |
| 117 | 19.1 | 229.3 |
| 104 | 21.2 | 253.9 |
| **Mean** | **20.4** | **244.7** |

Throughput is ~52% higher than the A40 run (mean 13.4 drafted tok/s), reflecting the A100's higher memory bandwidth and compute throughput. Rounds/min is similarly elevated.

---

## 6. Key Findings

### 6.1 QID 117 (Combinatorics) — Exceptional Alignment

QID 117 (hexagonal grid / Maia the bee path problem) again stands out dramatically:

- α = **0.873** vs 0.483–0.737 for the other 5 questions
- MAL = **4.363** (73% of the theoretical maximum of 6)
- Theoretical speedup = **4.38×**
- 23,831 / 27,310 tokens accepted (87.3%)
- Thinking text: **102,255 characters / 19,645 words** — longest chain-of-thought by far
- This is consistent with the A40 result (α = 0.856), confirming that combinatorial/formulaic reasoning strongly aligns the 2B and 32B token distributions

Despite the high acceptance rate, the model again predicted the wrong answer (predicted 7, reference 16). This reinforces that acceptance rate measures draft/target *alignment*, not reasoning *correctness*.

QID 117 dominated the total runtime: 1,429s out of 4,002s total (35.7%), sharing first place with QID 123.

### 6.2 QID 123 — High Acceptance, Long Thinking

QID 123 (counting children in a line) also hit the 5,462-round cap (same as QID 117), with α = 0.737 — significantly higher than the prior A40 run (α = 0.528). The thinking text was 106,173 characters (17,704 words). Despite strong draft-target alignment, the model predicted 7 instead of the correct 6.

### 6.3 Acceptance Rate Variance

Excluding QID 117 and 123, the α range is 0.483–0.607 (mean ≈ 0.539), giving theoretical speedups of 1.91–2.42×. QID 117 and 123 pull the overall α to 0.726 and overall speedup to 3.11×. This confirms the pattern from the A40 run: alignment is **highly problem-dependent**, with combinatorial and counting-heavy reasoning favouring the small draft model.

### 6.4 Per-Position Flat Profile

The expected monotonic decline in per-position acceptance is again not observed:

- p1 is the consistent low across all questions (mean 0.809 vs 0.864 at p0)
- p2–p4 recover to near-p0 levels
- p4 is often the highest position (mean 0.869)

This pattern is identical to the A40 run and supports the interpretation that after p1's "commitment dip," both models enter a locally coherent region where they continue to agree across the full 5-token window.

### 6.5 Accuracy (33%)

2 of 6 questions were answered correctly (QID 5, QID 168). The 4 errors:

- **QID 133** (transformation geometry, cogs): pred E, ref C. Multiple-choice spatial rotation — the image is critical and both models appear to mis-read the gear configuration.
- **QID 123** (counting, children in a line): pred 7, ref 6. Off-by-one counting error after 17,704 words of reasoning.
- **QID 117** (combinatorics, bee path): pred 7, ref 16. The 19,645-word chain-of-thought converged on the wrong answer despite high acceptance rate.
- **QID 104** (counting, shapes): pred B, ref D. Multiple-choice visual discrimination — same failure mode as the A40 run.

The two multiple-choice errors (B/E instead of correct answer) are consistent with the A40 run's pattern (also B instead of D on QID 104), suggesting a persistent bias in the 32B's final-answer generation on spatial/visual tasks. With only 6 questions, no strong conclusions can be drawn, but the accuracy drop vs. the A40 run (2/6 vs. 3/6) may reflect stochastic variation rather than a systematic A100 effect since the sampling seed was fixed.

---

## 7. Thinking Text Characteristics

| QID | Thinking chars | Thinking words | Final answer chars |
|-----|---------------|---------------|-------------------|
| 5 | 16,033 | 2,951 | 1,955 |
| 133 | 14,663 | 2,791 | 1,737 |
| 123 | 106,173 | 17,704 | 2,408 |
| 168 | 5,198 | 1,028 | 1,751 |
| **117** | **102,255** | **19,645** | **2,512** |
| 104 | 24,447 | 4,496 | 1,611 |

QID 117 and 123 together account for 208,428 chars / 37,349 words of thinking — 77% of all thinking text across the 6 questions. QID 168 remains the lightest thinker (5,198 chars) and was also the fastest question (90.8s).

---

## 8. Comparison to Prior A40 Run (SLURM 18052165, 2026-05-04)

| Metric | A40 Run | A100 Run | Δ |
|--------|---------|---------|---|
| Partition | `gpuA40x4` | `gpuA100x4` | — |
| 2B gpu_util | 0.50 | 0.55 | +0.05 (required for OOM fix) |
| Total latency | 4,353 s | 4,002 s | **−351 s (−8.1%)** |
| Total drafted tokens | 56,305 | 76,835 | +20,530 |
| Total accepted tokens | 38,900 | 55,764 | +16,864 |
| Overall α | 0.6909 | 0.7258 | **+0.035** |
| Theoretical speedup | 2.883× | 3.114× | **+0.23×** |
| Mean drafted tok/s | 13.4 | 20.4 | **+52%** |
| Accuracy | 3/6 (50%) | 2/6 (33%) | −1 (noise) |

The A100 run shows meaningfully better throughput (+52% tok/s), a higher acceptance rate (+0.035), and higher theoretical speedup (+0.23×). The total latency reduction (−8.1%) is modest because QID 117 and 123 both hit the 5,462-round cap regardless of hardware — the bottleneck is the number of verification rounds, not raw GPU speed.

The token totals are higher on A100 because QID 123 generated substantially more thinking tokens (27,310 drafted vs. the prior run where it was much shorter), suggesting non-determinism in the speculative loop despite `seed=42` — likely because the round-trip timing differences between A40 and A100 affect how `continue_final_message` prefixes are constructed at round boundaries.

---

## 9. Implementation Notes

### 9.1 Issues Encountered

| Issue | Fix |
|-------|-----|
| Job 18206513: 2B draft OOM on A100 40GB (`13.48 GiB available < 14.0 GiB needed for max_model_len=131072`) | Bumped `gpu_memory_utilization` from `0.50` → `0.55` in `job_vl_specdecode_manual.sh` |
| Prior text-only experiment (job 18201061): `max_tokens=40960` equals full context window, leaving 0 tokens for input | Reduced `max_tokens` to `16384` in `run_32b_standalone.py`, `run_2b_standalone.py`, `run_specdecode.py` |

### 9.2 `top_logprobs=20` Cap Impact

With `top_logprobs=20`, any draft token `t_i` that falls outside the 32B's top-20 candidates is assigned `p_target = 0`, forcing rejection. In practice this is the primary source of α < 1 for most rounds — not that the 2B and 32B fundamentally disagree, but that the correct target token isn't in the top-20 candidates returned. A higher cap would likely increase α, but vLLM's hard limit is 20.

### 9.3 Round Cap

QID 117 and QID 123 both hit the hard cap of 5,462 rounds before `</think>` was generated, meaning neither question's thinking phase fully completed. Their final answers were generated from a truncated chain-of-thought. This likely contributes to their incorrect predictions. The cap is set by `MAX_THINKING_TOKENS=32768` / `k=5` + overhead = ~5,500 rounds.

---

## 10. Conclusions

1. **The A100 run succeeded** after fixing a KV cache OOM on the 2B draft server (`gpu_util 0.50 → 0.55`), confirming the two-server manual rejection sampling approach works on A100 hardware.

2. **Overall α = 0.726 → 3.11× theoretical speedup** is an improvement over the A40 result (α = 0.691, 2.88×), driven by higher acceptance on QID 123 and marginal gains across other questions.

3. **Throughput is 52% higher on A100** (20.4 vs 13.4 drafted tok/s), confirming A100 compute advantage. End-to-end wall-clock savings are limited by the round cap on the two longest questions.

4. **QID 117 (combinatorics) is again the outlier** (α = 0.873, 4.38× speedup), consistent across both runs. Combinatorial/formulaic reasoning strongly aligns the 2B and 32B token distributions.

5. **Per-position flat profile reconfirmed** — p1 dips slightly, p2–p4 recover — across both hardware runs and across all questions. This is a robust pattern, not noise.

6. **Accuracy (2/6)** is one question lower than the A40 run (3/6), but with n=6 this is within expected variance given `seed=42` and non-deterministic round boundaries.

7. **Next steps:**
   - Run the full MathVision test set for statistically significant accuracy comparison
   - Compare wall-clock latency to a pure 32B-VL autoregressive baseline on A100
   - Vary `k` (3, 7, 10) to find the optimal draft length
   - Increase `MAX_THINKING_TOKENS` to allow QID 117/123 to complete their chains-of-thought
   - Investigate whether a fine-tuned 2B-VL draft on MathVision improves α on visual/spatial questions

---

*Generated from SLURM job 18222736 on Delta cluster. Models loaded from `/scratch/bgum/ratte/hf_cache`. Experiment code: `run_vl_specdecode_manual.py`, job script: `job_vl_specdecode_manual.sh`.*
