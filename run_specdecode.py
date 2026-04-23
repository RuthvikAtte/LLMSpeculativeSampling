"""
run_specdecode.py
Speculative decoding via vLLM's native draft-model method.
  Target: Qwen/Qwen3-32B  (TP=4)
  Draft:  Qwen/Qwen3-2B   (speculative_draft_tensor_parallel_size=1)
  num_speculative_tokens=5
Hardware: 4x A100 40GB, Perlmutter
"""

import json
import os
import time

from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TARGET_MODEL_ID = "Qwen/Qwen3-32B"
DRAFT_MODEL_ID = "Qwen/Qwen3-2B"
NUM_SPEC_TOKENS = 5
DATASET_PATH = "mathvision_mini.json"
RESULTS_PATH = "results/run_specdecode_results.json"
SYSTEM_PREFIX = "Please reason step by step, and put your final answer within \\boxed{}."

SAMPLING_PARAMS = SamplingParams(
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    presence_penalty=0.0,
    repetition_penalty=1.0,
    max_tokens=40960,
    seed=42,
)


def load_dataset(path: str) -> list[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data


def build_prompt(question: str) -> str:
    return f"{SYSTEM_PREFIX}\n\n{question}"


def extract_acceptance_rate(output) -> float | None:
    """
    Attempt to read acceptance rate (alpha) from vLLM's RequestMetrics.
    vLLM >= 0.4.3 surfaces spec-decode stats in output.metrics when
    speculative decoding is active.  The field name may vary across versions;
    we try the known locations and return None gracefully if unavailable.
    """
    metrics = getattr(output, "metrics", None)
    if metrics is None:
        return None
    # vLLM 0.5+ exposes spec_decode_worker_metrics on the scheduler output;
    # at the RequestOutput level the accepted-token ratio is sometimes stored
    # under different attribute names depending on build.
    for attr in (
        "spec_token_acceptance_rate",
        "acceptance_rate",
        "draft_acceptance_rate",
    ):
        val = getattr(metrics, attr, None)
        if val is not None:
            return float(val)
    return None


def main():
    os.makedirs("results", exist_ok=True)

    print(f"Loading target model: {TARGET_MODEL_ID}")
    print(f"Draft model:          {DRAFT_MODEL_ID}  (num_spec_tokens={NUM_SPEC_TOKENS})")

    llm = LLM(
        model=TARGET_MODEL_ID,
        tensor_parallel_size=4,
        dtype="bfloat16",
        trust_remote_code=True,
        speculative_config={
            "method": "draft_model",
            "model": DRAFT_MODEL_ID,
            "num_speculative_tokens": NUM_SPEC_TOKENS,
            "speculative_draft_tensor_parallel_size": 1,
            "draft_dtype": "bfloat16",
        },
    )

    dataset = load_dataset(DATASET_PATH)
    print(f"Loaded {len(dataset)} questions from {DATASET_PATH}\n")

    per_question_results = []

    for idx, item in enumerate(dataset):
        question = item["question"]
        answer = item.get("answer", "")
        prompt = build_prompt(question)

        t0 = time.perf_counter()
        outputs = llm.generate([prompt], SAMPLING_PARAMS)
        t1 = time.perf_counter()

        output = outputs[0]
        generated_text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        latency = t1 - t0
        tps = num_tokens / latency if latency > 0 else 0.0
        alpha = extract_acceptance_rate(output)

        result = {
            "question_idx": idx,
            "question": question,
            "reference_answer": answer,
            "generated_text": generated_text,
            "latency_s": latency,
            "tokens_generated": num_tokens,
            "tokens_per_second": tps,
            "acceptance_rate": alpha,
        }
        per_question_results.append(result)

        alpha_str = f"{alpha:.4f}" if alpha is not None else "n/a"
        print(
            f"[Q{idx}] latency={latency:.2f}s  tokens={num_tokens}"
            f"  tok/s={tps:.1f}  alpha={alpha_str}"
        )

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------
    total_latency = sum(r["latency_s"] for r in per_question_results)
    total_tokens = sum(r["tokens_generated"] for r in per_question_results)
    avg_tps = total_tokens / total_latency if total_latency > 0 else 0.0

    alphas = [r["acceptance_rate"] for r in per_question_results if r["acceptance_rate"] is not None]
    avg_alpha = sum(alphas) / len(alphas) if alphas else None

    summary = {
        "target_model": TARGET_MODEL_ID,
        "draft_model": DRAFT_MODEL_ID,
        "num_speculative_tokens": NUM_SPEC_TOKENS,
        "total_questions": len(per_question_results),
        "total_latency_s": total_latency,
        "total_tokens_generated": total_tokens,
        "average_tokens_per_second": avg_tps,
        "average_acceptance_rate": avg_alpha,
        "per_question": per_question_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n--- Summary Table ---")
    print(f"{'Q':>4}  {'Latency (s)':>12}  {'Tokens':>8}  {'Tok/s':>8}  {'Alpha':>8}")
    print("-" * 50)
    for r in per_question_results:
        alpha_str = f"{r['acceptance_rate']:.4f}" if r["acceptance_rate"] is not None else "     n/a"
        print(
            f"{r['question_idx']:>4}  {r['latency_s']:>12.2f}  "
            f"{r['tokens_generated']:>8}  {r['tokens_per_second']:>8.1f}  {alpha_str:>8}"
        )
    print("-" * 50)
    avg_alpha_str = f"{avg_alpha:.4f}" if avg_alpha is not None else "     n/a"
    print(
        f"{'ALL':>4}  {total_latency:>12.2f}  {total_tokens:>8}  {avg_tps:>8.1f}  {avg_alpha_str:>8}"
    )


if __name__ == "__main__":
    main()
