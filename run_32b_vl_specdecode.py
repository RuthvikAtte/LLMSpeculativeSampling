"""
run_32b_vl_specdecode.py
Speculative decoding (vLLM-native draft model) for Qwen3-VL on MathVision.
  Target: Qwen/Qwen3-VL-32B-Thinking  (TP=4)
  Draft:  Qwen/Qwen3-VL-2B-Thinking   (draft_tensor_parallel_size=1)
  num_speculative_tokens=5

Token-level verification is handled entirely by vLLM — the client sends
normal chat completions and reads alpha from the Prometheus /metrics endpoint.
"""

import json
import os
import re
import time

import requests
from openai import OpenAI

TARGET_MODEL_ID = "Qwen/Qwen3-VL-32B-Thinking"
DRAFT_MODEL_ID  = "Qwen/Qwen3-VL-2B-Thinking"
NUM_SPEC_TOKENS = 5
SERVER_BASE     = os.environ.get("VLLM_SERVER_BASE", "http://localhost:8000")
SERVER_URL      = f"{SERVER_BASE}/v1"
METRICS_URL     = f"{SERVER_BASE}/metrics"
DATASET_PATH    = "mathvision_vl.json"
RESULTS_DIR     = os.environ.get("RESULTS_DIR", "results")
RESULTS_PATH    = f"{RESULTS_DIR}/run_32b_vl_specdecode_results.json"

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


# ---------------------------------------------------------------------------
# Prompt builder (matches lmms-eval mathvision_doc_to_text)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Prometheus helpers
# ---------------------------------------------------------------------------

def _scrape_metrics() -> dict:
    text = requests.get(METRICS_URL, timeout=10).text
    result: dict = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r'^([^\s{]+)(?:\{([^}]*)\})?\s+([\d.e+\-]+)', line)
        if not m:
            continue
        name   = re.sub(r'_total$', '', m.group(1))
        labels = frozenset(re.findall(r'(\w+)="([^"]*)"', m.group(2) or ""))
        value  = float(m.group(3))
        result.setdefault(name, {})[labels] = value
    return result


def _spec_snapshot(metrics: dict) -> dict:
    def scalar(name):
        return sum(metrics.get(name, {}).values())

    per_pos_d = metrics.get("vllm:spec_decode_num_accepted_tokens_per_pos", {})
    per_pos   = [0.0] * NUM_SPEC_TOKENS
    for labels, v in per_pos_d.items():
        pos = int(dict(labels).get("pos", 0))
        if pos < NUM_SPEC_TOKENS:
            per_pos[pos] = v

    return {
        "drafts":   scalar("vllm:spec_decode_num_drafts"),
        "drafted":  scalar("vllm:spec_decode_num_draft_tokens"),
        "accepted": scalar("vllm:spec_decode_num_accepted_tokens"),
        "per_pos":  per_pos,
    }


def _diff_snapshots(before: dict, after: dict) -> dict:
    return {
        "drafts":   after["drafts"]   - before["drafts"],
        "drafted":  after["drafted"]  - before["drafted"],
        "accepted": after["accepted"] - before["accepted"],
        "per_pos":  [a - b for a, b in zip(after["per_pos"], before["per_pos"])],
    }


def _derive_metrics(d: dict) -> tuple:
    drafted  = d["drafted"]
    accepted = d["accepted"]
    rounds   = d["drafts"]
    per_pos  = d["per_pos"]

    alpha        = accepted / drafted  if drafted > 0  else None
    mal          = (1.0 + accepted / rounds) if rounds > 0 else None
    per_pos_rate = [v / rounds for v in per_pos] if rounds > 0 else None

    if alpha is None:
        speedup = None
    elif abs(1 - alpha) < 1e-9:
        speedup = float(NUM_SPEC_TOKENS + 1)
    else:
        gamma   = NUM_SPEC_TOKENS
        speedup = (1 - alpha ** (gamma + 1)) / (1 - alpha)

    return alpha, mal, per_pos_rate, speedup


# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    client = OpenAI(base_url=SERVER_URL, api_key="none")
    print(f"Target model: {TARGET_MODEL_ID}")
    print(f"Draft model:  {DRAFT_MODEL_ID}  (num_spec_tokens={NUM_SPEC_TOKENS})")
    print(f"Server:       {SERVER_BASE}\n")

    dataset = load_dataset(DATASET_PATH)
    print(f"Loaded {len(dataset)} questions from {DATASET_PATH}\n")

    per_question_results = []

    for item in dataset:
        qid      = item["qid"]
        category = item.get("category", "")
        answer   = item.get("answer", "")

        snap_before = _spec_snapshot(_scrape_metrics())

        t0 = time.perf_counter()
        response = client.chat.completions.create(
            model=TARGET_MODEL_ID,
            messages=[
                {"role": "user", "content": build_user_content(item)},
            ],
            **SAMPLING_KWARGS,
        )
        t1 = time.perf_counter()

        snap_after = _spec_snapshot(_scrape_metrics())
        diff       = _diff_snapshots(snap_before, snap_after)

        generated_text = response.choices[0].message.content
        num_tokens     = response.usage.completion_tokens
        latency        = t1 - t0
        tps            = num_tokens / latency if latency > 0 else 0.0

        alpha, mal, per_pos_rate, speedup_t = _derive_metrics(diff)

        result = {
            "qid":                      qid,
            "category":                 category,
            "reference_answer":         answer,
            "generated_text":           generated_text,
            "latency_s":                latency,
            "tokens_generated":         num_tokens,
            "tokens_per_second":        tps,
            "acceptance_rate":          alpha,
            "mean_acceptance_length":   mal,
            "theoretical_speedup":      speedup_t,
            "per_pos_acceptance_rate":  per_pos_rate,
        }
        per_question_results.append(result)

        alpha_str   = f"{alpha:.4f}"       if alpha     is not None else "n/a"
        mal_str     = f"{mal:.3f}"         if mal       is not None else "n/a"
        speedup_str = f"{speedup_t:.2f}x"  if speedup_t is not None else "n/a"
        print(
            f"[QID {qid:3d}|{category}]  latency={latency:.2f}s  "
            f"tokens={num_tokens}  tok/s={tps:.1f}  "
            f"α={alpha_str}  MAL={mal_str}  theory_speedup={speedup_str}"
        )
        if per_pos_rate:
            rates = "  ".join(f"p{i}={v:.3f}" for i, v in enumerate(per_pos_rate))
            print(f"          per-position: {rates}")

    total_latency = sum(r["latency_s"] for r in per_question_results)
    total_tokens  = sum(r["tokens_generated"] for r in per_question_results)
    avg_tps       = total_tokens / total_latency if total_latency > 0 else 0.0

    def _mean(key):
        vals = [r[key] for r in per_question_results if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "target_model":                   TARGET_MODEL_ID,
        "draft_model":                    DRAFT_MODEL_ID,
        "num_speculative_tokens":         NUM_SPEC_TOKENS,
        "total_questions":                len(per_question_results),
        "total_latency_s":                total_latency,
        "total_tokens_generated":         total_tokens,
        "average_tokens_per_second":      avg_tps,
        "average_acceptance_rate":        _mean("acceptance_rate"),
        "average_mean_acceptance_length": _mean("mean_acceptance_length"),
        "average_theoretical_speedup":    _mean("theoretical_speedup"),
        "per_question":                   per_question_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    def _f(v, fmt):
        return fmt.format(v) if v is not None else "n/a"

    print("\n--- Summary Table ---")
    print(f"{'QID':>5}  {'Category':<28}  {'Lat(s)':>7}  {'Toks':>6}  "
          f"{'Tok/s':>7}  {'α':>7}  {'MAL':>5}  {'Theory':>7}")
    print("-" * 82)
    for r in per_question_results:
        print(
            f"{r['qid']:>5}  {r['category']:<28}  "
            f"{r['latency_s']:>7.2f}  {r['tokens_generated']:>6}  "
            f"{r['tokens_per_second']:>7.1f}  "
            f"{_f(r['acceptance_rate'],       '{:.4f}'):>7}  "
            f"{_f(r['mean_acceptance_length'], '{:.3f}'):>5}  "
            f"{_f(r['theoretical_speedup'],    '{:.2f}x'):>7}"
        )
    print("-" * 82)
    print(
        f"{'ALL':>5}  {'':28}  "
        f"{total_latency:>7.2f}  {total_tokens:>6}  {avg_tps:>7.1f}  "
        f"{_f(_mean('acceptance_rate'),       '{:.4f}'):>7}  "
        f"{_f(_mean('mean_acceptance_length'), '{:.3f}'):>5}  "
        f"{_f(_mean('theoretical_speedup'),    '{:.2f}x'):>7}"
    )


if __name__ == "__main__":
    main()
