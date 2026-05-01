"""
run_32b_vl_standalone.py
Qwen3-VL-32B-Thinking standalone baseline on MathVision VL questions.
Runs against a vLLM OpenAI-compatible server started by job_vl_standalone.sh.
"""

import json
import os
import time

from openai import OpenAI

MODEL_ID      = "Qwen/Qwen3-VL-32B-Thinking"
SERVER_URL    = os.environ.get("VLLM_SERVER_URL", "http://localhost:8000/v1")
DATASET_PATH  = "mathvision_vl.json"
RESULTS_DIR   = os.environ.get("RESULTS_DIR", "results")
RESULTS_PATH  = f"{RESULTS_DIR}/run_32b_vl_standalone_results.json"
SAMPLING_KWARGS = dict(
    temperature=1.0,
    top_p=0.95,
    presence_penalty=0.0,
    max_tokens=40960,
    seed=42,
    extra_body={
        "top_k": 20,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": True},
    },
)


def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_prompt(item: dict) -> str:
    question = item["question"]
    choices  = item.get("options", "")
    len_choices = len(choices)
    opts = [chr(ord("A") + i) for i in range(len_choices)]
    choices_str = "\n".join(f"{o}. {c}" for o, c in zip(opts, choices))
    query_prompt = 'Please solve the problem step by step and put your answer in one "\\boxed{}".'
    if choices_str:
        query_prompt += f"{question}\nChoices: {choices_str}"
    else:
        query_prompt += question
    return query_prompt


def build_user_content(item: dict) -> list:
    content = []
    if item.get("image_b64"):
        content.append({
            "type":      "image_url",
            "image_url": {"url": f"data:image/png;base64,{item['image_b64']}"},
        })
    content.append({"type": "text", "text": build_prompt(item)})
    return content


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    client = OpenAI(base_url=SERVER_URL, api_key="none")
    print(f"Connected to vLLM server at {SERVER_URL}  (model: {MODEL_ID})")

    dataset = load_dataset(DATASET_PATH)
    print(f"Loaded {len(dataset)} questions from {DATASET_PATH}\n")

    per_question_results = []

    for item in dataset:
        qid      = item["qid"]
        answer   = item.get("answer", "")
        category = item.get("category", "")

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "user", "content": build_user_content(item)},
            ],
            **SAMPLING_KWARGS,
        )
        t1 = time.perf_counter()

        generated_text = response.choices[0].message.content
        num_tokens     = response.usage.completion_tokens
        latency        = t1 - t0
        tps            = num_tokens / latency if latency > 0 else 0.0

        result = {
            "qid":               qid,
            "category":          category,
            "reference_answer":  answer,
            "generated_text":    generated_text,
            "latency_s":         latency,
            "tokens_generated":  num_tokens,
            "tokens_per_second": tps,
        }
        per_question_results.append(result)
        print(f"[QID {qid:3d}|{category}] latency={latency:.2f}s  tokens={num_tokens}  tok/s={tps:.1f}")

    total_latency = sum(r["latency_s"] for r in per_question_results)
    total_tokens  = sum(r["tokens_generated"] for r in per_question_results)
    avg_tps       = total_tokens / total_latency if total_latency > 0 else 0.0

    summary = {
        "model":                     MODEL_ID,
        "total_questions":           len(per_question_results),
        "total_latency_s":           total_latency,
        "total_tokens_generated":    total_tokens,
        "average_tokens_per_second": avg_tps,
        "per_question":              per_question_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    print("\n--- Summary Table ---")
    print(f"{'QID':>5}  {'Category':<28}  {'Lat(s)':>7}  {'Toks':>6}  {'Tok/s':>7}")
    print("-" * 62)
    for r in per_question_results:
        print(f"{r['qid']:>5}  {r['category']:<28}  {r['latency_s']:>7.2f}  "
              f"{r['tokens_generated']:>6}  {r['tokens_per_second']:>7.1f}")
    print("-" * 62)
    print(f"{'ALL':>5}  {'':28}  {total_latency:>7.2f}  {total_tokens:>6}  {avg_tps:>7.1f}")


if __name__ == "__main__":
    main()
