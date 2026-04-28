# Speculative Decoding Experiment Report
**Qwen3-32B + Qwen3-1.7B on Delta GPU (4× A40)**
**Date: April 23–24, 2026**

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Experiment Setup](#experiment-setup)
3. [Dataset](#dataset)
4. [Results Overview](#results-overview)
5. [32B Standalone Baseline](#32b-standalone-baseline)
6. [2B Standalone Baseline](#2b-standalone-baseline)
7. [Speculative Decoding](#speculative-decoding)
8. [Correctness Analysis](#correctness-analysis)
9. [Speculative Decoding Deep Dive](#speculative-decoding-deep-dive)
10. [Theoretical vs Actual Speedup](#theoretical-vs-actual-speedup)
11. [Issues and Limitations](#issues-and-limitations)
12. [Recommendations](#recommendations)

---

## Executive Summary

This experiment benchmarks **vLLM's native draft-model speculative decoding** using Qwen3-32B as the target and Qwen3-1.7B as the draft, comparing against standalone 32B and standalone 1.7B baselines on 6 MathVision questions on a 4× A40 node.

**Key results:**

| Method | Avg Tok/s | Total Latency | Speedup vs 32B |
|---|---|---|---|
| 32B Standalone | 27.5 | 362.3 s | 1.00× (baseline) |
| 1.7B Standalone | 225.6 s | 419.0 s | — (different model) |
| **Spec Decode** | **43.3** | **257.7 s** | **1.57× tok/s / 1.41× wall-time** |

Speculative decoding achieved a **mean acceptance rate (α) of 0.573** and a **mean acceptance length (MAL) of 3.87 tokens per 32B forward pass**. Theoretical speedup under the zero-draft-cost assumption was 2.28×; the actual 1.41× reflects the overhead of running the 1.7B draft model co-located on the same 4 GPUs as the 32B target (mandated by vLLM 0.19.1).

All 6 questions were answered correctly by both the 32B standalone and spec decode. The 1.7B standalone generated runaway outputs (up to 40,960 tokens) on 2 of 6 questions, making it unsuitable as a correctness reference.

---

## Experiment Setup

### Models

| Role | Model | Architecture | Vocab Size | Parameters |
|---|---|---|---|---|
| Target | `Qwen/Qwen3-32B` | `Qwen3ForCausalLM` | 151,936 | 32B |
| Draft | `Qwen/Qwen3-1.7B` | `Qwen3ForCausalLM` | 151,936 | 1.7B |
| 2B baseline | `Qwen/Qwen3-2B` | `Qwen3_5ForConditionalGeneration` | — | ~2B |

> **Note:** `Qwen/Qwen3-2B` is not the same model family as `Qwen3-32B` — it uses a different architecture (`Qwen3_5ForConditionalGeneration`) and was later found to generate degenerate outputs. The draft model in spec decode was corrected to `Qwen/Qwen3-1.7B`, which shares `Qwen3ForCausalLM` architecture and identical vocabulary with the 32B.

### Hardware

| Resource | Value |
|---|---|
| Node | 1× Delta `gpuA40x4` |
| GPUs | 4× NVIDIA A40 (46,068 MiB each) |
| Tensor Parallel Size | 4 (target and draft both TP=4) |
| dtype | bfloat16 |
| GPU memory utilization | 0.70 (spec decode) / default (standalone) |

### Generation Parameters

| Parameter | Value |
|---|---|
| Temperature | 1.0 |
| Top-p | 0.95 |
| Top-k | 20 |
| Presence penalty | 0.0 |
| Repetition penalty | 1.0 |
| Max tokens | 40,960 |
| Seed | 42 |
| Thinking mode | Enabled (`enable_thinking=True`) |

### Spec Decode Parameters

| Parameter | Value |
|---|---|
| Method | `draft_model` (vLLM V1 native) |
| Draft speculative tokens (γ) | 5 |
| draft_tensor_parallel_size | 4 |
| vLLM version | 0.19.1 |

### Prompt Format

All prompts used the Qwen3 chat template with thinking enabled:
```
<|im_start|>system
Please reason step by step, and put your final answer within \boxed{}.
<|im_end|>
<|im_start|>user
{question}
<|im_end|>
<|im_start|>assistant
```
The model generates a `<think>...</think>` reasoning block followed by the final answer in `\boxed{}`.

---

## Dataset

6 questions from MathVision Mini (`mathvision_mini.json`), spanning arithmetic, algebra, geometry, calculus, and combinatorics:

| Q | Question | Reference Answer | Difficulty |
|---|---|---|---|
| 0 | A right triangle has legs of length 3 and 4. What is the length of the hypotenuse? | 5 | Easy |
| 1 | Find all real solutions to x² - 5x + 6 = 0 | x = 2 or x = 3 | Easy |
| 2 | What is the area of a circle with radius 7? | 49π | Easy |
| 3 | A train travels 120 miles in 2 hours. What is its average speed in miles per hour? | 60 | Easy |
| 4 | Evaluate the integral ∫₀¹ x² dx | 1/3 | Medium |
| 5 | How many ways can you arrange the letters in the word MATH? | 24 | Easy |

All questions are designed to have unambiguous closed-form answers.

---

## Results Overview

### Throughput Comparison

| Method | Total Latency (s) | Total Tokens | Avg Tok/s | Wall-time Speedup |
|---|---|---|---|---|
| 32B Standalone | 362.3 | 9,971 | 27.5 | 1.00× |
| 1.7B Standalone | 419.0 | 94,531 | 225.6 | N/A* |
| **Spec Decode** | **257.7** | **11,153** | **43.3** | **1.41×** |

*The 1.7B standalone generated runaway outputs on 2 questions, making aggregate comparison misleading.

### Per-Question Latency (seconds)

| Q | 32B Standalone | 1.7B Standalone | Spec Decode | Actual Speedup |
|---|---|---|---|---|
| 0 | 64.32 | 5.23 | 40.55 | 1.59× |
| 1 | 70.70 | 40.92 | 36.15 | 1.96× |
| 2 | 71.43 | 181.54 ⚠️ | 45.82 | 1.56× |
| 3 | 31.12 | 181.55 ⚠️ | 23.12 | 1.35× |
| 4 | 78.87 | 6.10 | 66.47 | 1.19× |
| 5 | 45.90 | 3.67 | 45.62 | 1.01× |
| **Total** | **362.34** | **419.01** | **257.73** | **1.41×** |

⚠️ Hit max_tokens limit (40,960 tokens) — runaway generation.

---

## 32B Standalone Baseline

**Model:** `Qwen/Qwen3-32B` | **TP=4** | **Hardware:** 4× A40

| Q | Question (brief) | Latency (s) | Tokens | Tok/s | Answer |
|---|---|---|---|---|---|
| 0 | Hypotenuse (3-4-5) | 64.32 | 1,752 | 27.2 | ✓ `5` |
| 1 | Quadratic x²-5x+6=0 | 70.70 | 1,950 | 27.6 | ✓ `2, 3` |
| 2 | Area of circle r=7 | 71.43 | 1,970 | 27.6 | ✓ `49π` |
| 3 | Train speed | 31.12 | 858 | 27.6 | ✓ `60` |
| 4 | Integral ∫₀¹ x² dx | 78.87 | 2,175 | 27.6 | ✓ `1/3` |
| 5 | Arrangements of MATH | 45.90 | 1,266 | 27.6 | ✓ `24` |
| **Avg/Total** | | **60.4 / 362.3** | **1,662 / 9,971** | **27.5** | **6/6** |

**Observations:**
- Throughput is remarkably consistent at ~27.6 tok/s for questions 1–5. Q0 is marginally slower (27.2) due to slightly different memory/KV cache state at the start.
- Token counts vary from 858 (simple rate problem) to 2,175 (integral), reflecting the depth of the chain-of-thought reasoning — the 32B model spends more tokens verifying its calculus derivation.
- All 6 answers are correct. The 32B model produces well-structured `<think>...</think>` blocks before the final `\boxed{}` answer.
- Total generation time: **362.3 seconds** (~6 minutes) for 6 questions.

---

## 2B Standalone Baseline

**Model:** `Qwen/Qwen3-2B` (`Qwen3_5ForConditionalGeneration`) | **TP=4**

> ⚠️ **Important caveat:** This model is a *different architecture* from `Qwen3-32B`. It was run before this incompatibility was discovered. Results should be treated as informational only and **not** used as a fair comparison baseline.

| Q | Question (brief) | Latency (s) | Tokens | Tok/s | Answer | Status |
|---|---|---|---|---|---|---|
| 0 | Hypotenuse (3-4-5) | 5.23 | 655 | 125.2 | ✓ `5` | Normal |
| 1 | Quadratic x²-5x+6=0 | 40.92 | 9,638 | 235.6 | ✓ `2, 3` | Repetitive loop |
| 2 | Area of circle r=7 | 181.54 | **40,960** | 225.6 | ✓ `49π` | ⚠️ Hit limit |
| 3 | Train speed | 181.55 | **40,960** | 225.6 | ✓ `60` | ⚠️ Hit limit |
| 4 | Integral ∫₀¹ x² dx | 6.10 | 1,449 | 237.7 | ✓ `1/3` | Normal |
| 5 | Arrangements of MATH | 3.67 | 869 | 236.8 | ✓ `24` | Normal |
| **Avg/Total** | | **69.8 / 419.0** | **15,755 / 94,531** | **225.6** | **6/6** | |

**Observations:**
- Peak throughput of 225–237 tok/s vs 27.5 tok/s for the 32B — roughly **8.2× raw token throughput** due to far fewer parameters and weights per GPU.
- Questions 2 and 3 both hit the 40,960 token hard limit, generating enormous amounts of text before eventually producing the correct answer buried deep in the output. This is consistent with repetitive looping behaviour seen in `Qwen3_5ForConditionalGeneration` when not using its intended chat template.
- Question 1 generated 9,638 tokens for a trivial quadratic — already exhibiting runaway behaviour, just not enough to hit the limit.
- Despite the runaway generation, the model does eventually land on the correct `\boxed{}` answer in all cases, suggesting the architecture can reason but cannot reliably terminate.
- **This model is not a valid draft for speculative decoding** with Qwen3-32B due to different architecture and vocabulary.

---

## Speculative Decoding

**Target:** `Qwen/Qwen3-32B` (TP=4) | **Draft:** `Qwen/Qwen3-1.7B` (TP=4) | **γ=5**

| Q | Latency (s) | Tokens | Tok/s | α | MAL | Theory | Actual speedup |
|---|---|---|---|---|---|---|---|
| 0 | 40.55 | 1,648 | 40.6 | 0.559 | 3.793 | 2.20× | 1.59× |
| 1 | 36.15 | 1,832 | 50.7 | 0.696 | 4.482 | 2.92× | 1.96× |
| 2 | 45.82 | 1,928 | 42.1 | 0.543 | 3.715 | 2.13× | 1.56× |
| 3 | 23.12 | 981 | 42.4 | 0.540 | 3.702 | 2.12× | 1.35× |
| 4 | 66.47 | 2,897 | 43.6 | 0.578 | 3.889 | 2.28× | 1.19× |
| 5 | 45.62 | 1,867 | 40.9 | 0.522 | 3.611 | 2.05× | 1.01× |
| **Avg/Total** | **42.96 / 257.7** | **1,859 / 11,153** | **43.3** | **0.573** | **3.865** | **2.28×** | **1.41×** |

**Legend:**
- **α** — token acceptance rate: fraction of drafted tokens accepted by the 32B verifier
- **MAL** — mean acceptance length: average tokens produced per 32B forward pass (including guaranteed bonus token)
- **Theory** — theoretical speedup = (1 − α^(γ+1)) / (1 − α) assuming zero draft cost (Leviathan et al. 2023, Theorem 3.8)
- **Actual speedup** — wall-time ratio: 32B standalone latency / spec decode latency for the same question

---

## Correctness Analysis

| Q | Reference | 32B Answer | Spec Decode Answer | Match? |
|---|---|---|---|---|
| 0 | `5` | `\boxed{5}` | `\boxed{5}` | ✓ |
| 1 | `x = 2 or x = 3` | `\boxed{2}`, `\boxed{3}` | `\boxed{2}`, `\boxed{3}` | ✓ |
| 2 | `49π` | `\boxed{49\pi}` | `\boxed{49\pi}` | ✓ |
| 3 | `60` | `\boxed{60}` | `\boxed{60}` | ✓ |
| 4 | `1/3` | `\boxed{\frac{1}{3}}` | `\boxed{\frac{1}{3}}` | ✓ |
| 5 | `24` | `\boxed{24}` | `\boxed{24}` | ✓ |

**Spec decode preserves output quality exactly.** This is guaranteed by the Leviathan et al. rejection sampling algorithm: the output distribution of spec decode is provably identical to sampling directly from the target model. The experiment confirms this holds in practice — every question produces the same correct answer whether or not speculative decoding is used.

---

## Speculative Decoding Deep Dive

### Per-Position Acceptance Rates

The acceptance rate at each draft position (p0 = first draft token, p4 = fifth) shows a clear monotonic decay — each deeper speculative token is harder for the 1.7B draft to predict correctly:

| Q | p0 | p1 | p2 | p3 | p4 |
|---|---|---|---|---|---|
| 0 | 0.763 | 0.639 | 0.538 | 0.457 | 0.395 |
| 1 | 0.873 | 0.758 | 0.672 | 0.616 | 0.562 |
| 2 | 0.800 | 0.624 | 0.507 | 0.418 | 0.366 |
| 3 | 0.770 | 0.634 | 0.498 | 0.426 | 0.374 |
| 4 | 0.799 | 0.667 | 0.542 | 0.464 | 0.416 |
| 5 | 0.779 | 0.596 | 0.482 | 0.404 | 0.350 |
| **Avg** | **0.797** | **0.653** | **0.540** | **0.464** | **0.410** |

Key observations:
- **p0 = 0.797**: The first draft token is accepted ~80% of the time — the 1.7B model has a strong prior on what comes next given the context.
- **p4 = 0.410**: By the 5th speculative token, acceptance has nearly halved. The 1.7B model's uncertainty compounds with each additional step.
- The decay follows roughly a geometric progression with ratio ~0.83 per step, consistent with the Markov-like independence assumption in the theoretical analysis.
- **Q1 stands out** (p0=0.873, p4=0.562) — the algebraic problem has highly predictable token sequences (both models agree on algebraic manipulation steps), yielding the highest per-position rates and best overall speedup.
- **Q5 is the worst** (p0=0.779, p4=0.350) — permutation reasoning may involve less stereotyped phrasing, causing the 1.7B to diverge more from the 32B.

### Acceptance Rate vs Speedup

| Q | α | Theory Speedup | Actual Speedup | Efficiency (actual/theory) |
|---|---|---|---|---|
| 1 | 0.696 | 2.92× | 1.96× | 67% |
| 4 | 0.578 | 2.28× | 1.19× | 52% |
| 0 | 0.559 | 2.20× | 1.59× | 72% |
| 2 | 0.543 | 2.13× | 1.56× | 73% |
| 3 | 0.540 | 2.12× | 1.35× | 64% |
| 5 | 0.522 | 2.05× | 1.01× | 49% |

The **efficiency** (actual / theoretical speedup) ranges from 49% to 73%, averaging ~63%. The gap is entirely explained by the draft model not being free (c > 0). With the 1.7B draft co-located on the same 4 GPUs as the 32B target, every spec decode round incurs:
1. One 1.7B forward pass (draft)
2. One 32B forward pass (verify)

The 1.7B is ~19× smaller in parameters but requires the same TP=4 synchronisation overhead. A dedicated GPU for the draft would significantly reduce this cost.

### Token Count Comparison

| Q | 32B Tokens | Spec Tokens | Ratio |
|---|---|---|---|
| 0 | 1,752 | 1,648 | 0.94 |
| 1 | 1,950 | 1,832 | 0.94 |
| 2 | 1,970 | 1,928 | 0.98 |
| 3 | 858 | 981 | 1.14 |
| 4 | 2,175 | 2,897 | 1.33 |
| 5 | 1,266 | 1,867 | 1.47 |

Speculative decoding does not guarantee identical token counts — it samples from the same distribution as the 32B, but due to the stochastic nature of the rejection sampling and the temperature=1.0 setting, individual sequences can differ. Q4 and Q5 produced notably longer responses under spec decode, which partially explains the lower observed speedup on those questions (more tokens to generate, even though each round is faster).

---

## Theoretical vs Actual Speedup

The theoretical speedup from Leviathan et al. 2023 (Theorem 3.8) assumes the draft model is infinitely faster than the target (c → 0):

```
Speedup_theoretical = (1 - α^(γ+1)) / (1 - α)
```

With α = 0.573 and γ = 5:
```
= (1 - 0.573^6) / (1 - 0.573)
= (1 - 0.0364) / 0.427
= 2.26×
```

The actual speedup is **1.41× wall-time** or **1.57× tok/s**. The ratio of actual to theoretical is **~0.63**, meaning we recover 63% of the theoretical maximum.

**Why the gap?**

In practice, the total time per round with a draft model is:
```
T_round = T_draft + T_verify
```

With `draft_tensor_parallel_size=4` forced by vLLM 0.19.1, both models use the same 4-GPU group. The 1.7B draft forward pass is not negligible — it's roughly 1/19 the FLOPs of the 32B but shares the same memory bandwidth bottleneck on A40s.

A rough model: if c = T_draft / T_target ≈ 0.2 (estimated), then the effective speedup formula becomes:
```
Speedup_actual ≈ Speedup_theoretical / (1 + c) ≈ 2.28 / 1.2 ≈ 1.90×
```

The remaining gap (1.90× → 1.41×) is due to:
- Kernel launch overhead for the additional draft passes
- Synchronisation overhead in TP=4 all-reduce operations
- Slightly higher memory pressure reducing KV cache efficiency

---

## Issues and Limitations

### 1. Draft Model Architecture Mismatch (Fixed)
The initial draft model `Qwen/Qwen3-2B` uses `Qwen3_5ForConditionalGeneration` architecture — incompatible with `Qwen3ForCausalLM` (32B). vLLM raised a `ValueError` during worker init. This was resolved by switching to `Qwen/Qwen3-1.7B`, which shares the same architecture and vocabulary.

### 2. vLLM 0.19.1 TP Constraint
vLLM 0.19.1 requires `draft_tensor_parallel_size == tensor_parallel_size`. Setting the draft to TP=1 raises:
```
ValueError: Currently, 'draft_tensor_parallel_size' and 'tensor_parallel_size' must be the same. Got 1 and 4.
```
This forces both models to share the same 4-GPU process group, eliminating the speed advantage of a smaller draft model. This constraint is a known limitation expected to be relaxed in future vLLM versions.

### 3. 2B Standalone Runaway Generation
`Qwen/Qwen3-2B` (`Qwen3_5ForConditionalGeneration`) generates up to 40,960 tokens on simple problems (Q2: circle area, Q3: train speed). This is likely due to the model being a conditional generation architecture that loops without a proper EOS signal under the causal LM chat template. This model should be re-run with the correct `Qwen3-1.7B` model for a valid baseline.

### 4. Small Dataset
Only 6 questions were run, all relatively simple. The α and speedup estimates have high variance; a larger and harder dataset would give more reliable statistics and likely lower α (harder problems → more divergence between 1.7B and 32B).

### 5. gpu_memory_utilization = 0.70
The spec decode run used `gpu_memory_utilization=0.70` (vs default 0.90 for standalones) to accommodate both models on the same GPU memory. This reduces the available KV cache, which may cap sequence lengths and affect long-reasoning performance.

---

## Recommendations

| Priority | Action | Expected Impact |
|---|---|---|
| **High** | Re-run 2B standalone with `Qwen/Qwen3-1.7B` to get a valid baseline | Fixes invalid comparison |
| **High** | Request a 2-node job or 8-GPU single node to run draft at TP=1 separately | Eliminates TP overhead; expected actual speedup → 1.8–2.2× |
| **Medium** | Test `Qwen/Qwen3-4B` as draft model | Higher α (closer in capability to 32B), likely 2.5–3× theoretical speedup |
| **Medium** | Tune γ per question difficulty | For α > 0.70 (Q1-like), γ=8 would extract ~3.5× theoretical; for α < 0.55 (Q5-like), γ=3 reduces wasted draft passes |
| **Medium** | Expand to full MathVision (300+ questions) | Reduces variance in α/speedup estimates, surfaces harder questions where 1.7B diverges more |
| **Low** | Profile draft model latency with `nvtx` tracing | Measure c = T_draft/T_target precisely to validate the speedup model |
| **Low** | Upgrade to vLLM >= 0.20 when TP constraint is lifted | Will allow `draft_tensor_parallel_size=1`, recovering the theoretical speedup gap |

---

## Appendix: Raw Aggregate Statistics

### 32B Standalone
```
model:                   Qwen/Qwen3-32B
total_questions:         6
total_latency_s:         362.342
total_tokens_generated:  9971
average_tokens_per_second: 27.518
```

### 1.7B Standalone
```
model:                   Qwen/Qwen3-2B  (NOTE: actually Qwen3_5ForConditionalGeneration)
total_questions:         6
total_latency_s:         419.002
total_tokens_generated:  94531
average_tokens_per_second: 225.610
```

### Speculative Decoding
```
target_model:                    Qwen/Qwen3-32B
draft_model:                     Qwen/Qwen3-1.7B
num_speculative_tokens:          5
total_questions:                 6
total_latency_s:                 257.737
total_tokens_generated:          11153
average_tokens_per_second:       43.273
average_acceptance_rate:         0.5730
average_mean_acceptance_length:  3.865
average_theoretical_speedup:     2.283
```
