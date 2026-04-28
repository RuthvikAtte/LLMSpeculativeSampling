"""
run_2b_standalone.py
Draft model only (Qwen3-2B), no speculative decoding.
Hardware: 4x A100 40GB, TP=4 (Perlmutter)
"""

import json
import os
import time

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_ID = "Qwen/Qwen3-2B"
DATASET_PATH = "mathvision_mini.json"
RESULTS_DIR  = os.environ.get("RESULTS_DIR", "results")
RESULTS_PATH = f"{RESULTS_DIR}/run_2b_standalone_results.json"
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


def build_prompt(question: str, tokenizer: AutoTokenizer) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PREFIX},
        {"role": "user",   "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    llm = LLM(
        model=MODEL_ID,
        tensor_parallel_size=4,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    dataset = load_dataset(DATASET_PATH)
    print(f"Loaded {len(dataset)} questions from {DATASET_PATH}\n")

    per_question_results = []

    for idx, item in enumerate(dataset):
        question = item["question"]
        answer = item.get("answer", "")
        prompt = build_prompt(question, tokenizer)

        t0 = time.perf_counter()
        outputs = llm.generate([prompt], SAMPLING_PARAMS)
        t1 = time.perf_counter()

        output = outputs[0]
        generated_text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        latency = t1 - t0
        tps = num_tokens / latency if latency > 0 else 0.0

        result = {
            "question_idx": idx,
            "question": question,
            "reference_answer": answer,
            "generated_text": generated_text,
            "latency_s": latency,
            "tokens_generated": num_tokens,
            "tokens_per_second": tps,
        }
        per_question_results.append(result)

        print(
            f"[Q{idx}] latency={latency:.2f}s  tokens={num_tokens}  tok/s={tps:.1f}"
        )

    # ------------------------------------------------------------------
    # Aggregate summary
    # ------------------------------------------------------------------
    total_latency = sum(r["latency_s"] for r in per_question_results)
    total_tokens = sum(r["tokens_generated"] for r in per_question_results)
    avg_tps = total_tokens / total_latency if total_latency > 0 else 0.0

    summary = {
        "model": MODEL_ID,
        "total_questions": len(per_question_results),
        "total_latency_s": total_latency,
        "total_tokens_generated": total_tokens,
        "average_tokens_per_second": avg_tps,
        "per_question": per_question_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n--- Summary Table ---")
    print(f"{'Q':>4}  {'Latency (s)':>12}  {'Tokens':>8}  {'Tok/s':>8}")
    print("-" * 40)
    for r in per_question_results:
        print(
            f"{r['question_idx']:>4}  {r['latency_s']:>12.2f}  "
            f"{r['tokens_generated']:>8}  {r['tokens_per_second']:>8.1f}"
        )
    print("-" * 40)
    print(
        f"{'ALL':>4}  {total_latency:>12.2f}  {total_tokens:>8}  {avg_tps:>8.1f}"
    )


if __name__ == "__main__":
    main()
