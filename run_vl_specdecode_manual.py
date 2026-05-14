"""
run_vl_specdecode_manual.py
Token-level speculative decoding (manual) for Qwen3-VL on MathVision.
  Draft:  Qwen/Qwen3-VL-2B-Thinking  — generates k tokens per round
  Target: Qwen/Qwen3-VL-32B-Thinking — verifies k tokens via rejection sampling

Algorithm (per round):
  1. Draft model generates k tokens with logprobs.
  2. Target model generates k+1 tokens from the SAME context, with top_logprobs.
  3. For position i: accept draft token t_i if U(0,1) <= min(1, p_target(t_i)/p_draft(t_i)).
     p_target(t_i) is looked up in target's top_logprobs at position i.
  4. At first rejection, use target's token at that position as the correction.
  5. If all k accepted, append target's (k+1)-th token as the bonus.
  6. Repeat until </think>, then target generates the final boxed answer.

Two separate vLLM servers — no --speculative-config needed, no multimodal guard.
  VLLM_32B_URL  e.g. http://node001:8000/v1
  VLLM_2B_URL   e.g. http://node002:8001/v1
"""

import json
import math
import os
import random
import time

from openai import OpenAI

TARGET_MODEL_ID     = "Qwen/Qwen3-VL-32B-Thinking"
DRAFT_MODEL_ID      = "Qwen/Qwen3-VL-2B-Thinking"
K                   = 5        # draft tokens per round
TOP_LOGPROBS        = 20       # candidates scanned per position for p_target lookup (vLLM max=20)
MAX_THINKING_TOKENS = 32768    # safety cap on thinking-phase tokens

DATASET_PATH  = "mathvision_vl.json"
RESULTS_DIR   = os.environ.get("RESULTS_DIR", "results")
RESULTS_PATH  = f"{RESULTS_DIR}/run_vl_specdecode_manual_results.json"
SERVER_32B_URL = os.environ.get("VLLM_32B_URL", "http://localhost:8000/v1")
SERVER_2B_URL  = os.environ.get("VLLM_2B_URL",  "http://localhost:8001/v1")

_EXTRA_THINK = {
    "top_k": 20,
    "repetition_penalty": 1.0,
    "continue_final_message": True,
    "add_generation_prompt": False,
    "chat_template_kwargs": {"enable_thinking": True},
}


# ---------------------------------------------------------------------------
# Dataset / prompt
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def build_prompt(item: dict) -> str:
    import re
    question = re.sub(r'<image\d+>', '', item["question"]).strip()
    choices  = item.get("options", "")
    opts = [chr(ord("A") + i) for i in range(len(choices))]
    choices_str = "\n".join(f"{o}. {c}" for o, c in zip(opts, choices))
    q = 'Please solve the problem step by step and put your answer in one "\\boxed{}".'
    if choices_str:
        q += f"{question}\nChoices: {choices_str}"
    else:
        q += question
    return q


def make_base_messages(item: dict) -> list:
    image_b64 = item.get("image_b64")
    content = []
    if image_b64:
        content.append({
            "type":      "image_url",
            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
        })
    content.append({"type": "text", "text": build_prompt(item)})
    return [{"role": "user", "content": content}]


def _make_messages(base: list, partial: str) -> list:
    return base + [{"role": "assistant", "content": partial}]


# ---------------------------------------------------------------------------
# Token-level spec decode helpers
# ---------------------------------------------------------------------------

def _generate_with_logprobs(client: OpenAI, model: str, messages: list,
                             max_tokens: int):
    """Single chat-completions call returning logprob entries."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        logprobs=True,
        top_logprobs=TOP_LOGPROBS,
        temperature=1.0,
        top_p=0.95,
        seed=42,
        extra_body=_EXTRA_THINK,
    )
    return resp.choices[0].logprobs.content or []


def _p_target(target_entry, token_str: str) -> float:
    """Look up p_target(token_str) in a target logprob entry's top_logprobs."""
    if target_entry.token == token_str:
        lp = target_entry.logprob
        return math.exp(lp) if lp is not None else 0.0
    for cand in (target_entry.top_logprobs or []):
        if cand.token == token_str:
            return math.exp(cand.logprob) if cand.logprob is not None else 0.0
    return 0.0  # not in top-K, treated as negligible


def rejection_sample(d_list: list, t_list: list) -> tuple:
    """
    Token-level rejection sampling.

    Args:
        d_list: draft logprob entries  (from 2B, k entries)
        t_list: target logprob entries (from 32B, k+1 entries)

    Returns:
        accepted_text   : string of accepted draft tokens
        n_accepted      : int count of accepted draft tokens
        correction_token: str — target's correction token (at rejection) or bonus (all accepted)
        per_pos         : list[bool] — acceptance result for each REACHED position
    """
    accepted_text = ""
    n_accepted    = 0
    per_pos       = []

    for i, d_entry in enumerate(d_list):
        dt  = d_entry.token
        p_q = math.exp(d_entry.logprob) if d_entry.logprob is not None else 0.0

        p_p = _p_target(t_list[i], dt) if i < len(t_list) else 0.0

        ratio = min(1.0, p_p / p_q) if p_q > 0 else 0.0

        if random.random() <= ratio:
            accepted_text += dt
            n_accepted    += 1
            per_pos.append(True)
        else:
            correction = t_list[i].token if i < len(t_list) else None
            per_pos.append(False)
            return accepted_text, n_accepted, correction, per_pos

    # All draft tokens accepted — take bonus from target position k
    bonus = t_list[n_accepted].token if n_accepted < len(t_list) else None
    return accepted_text, n_accepted, bonus, per_pos


# ---------------------------------------------------------------------------
# Per-question loop
# ---------------------------------------------------------------------------

def run_specdecode_question(client_32b: OpenAI, client_2b: OpenAI,
                             item: dict) -> dict:
    qid       = item["qid"]
    base_msgs = make_base_messages(item)
    partial   = "<think>\n"

    rounds           = 0
    total_drafted    = 0
    total_accepted   = 0
    thinking_tokens  = 0
    per_pos_drafted  = [0] * K
    per_pos_accepted = [0] * K

    t0   = time.perf_counter()
    done = False

    while not done and thinking_tokens < MAX_THINKING_TOKENS:
        rounds += 1
        msgs = _make_messages(base_msgs, partial)

        # ── Draft ────────────────────────────────────────────────────────────
        d_list = _generate_with_logprobs(client_2b, DRAFT_MODEL_ID, msgs, K)
        if not d_list:
            break
        k_this = len(d_list)

        # ── Verify ───────────────────────────────────────────────────────────
        t_list = _generate_with_logprobs(client_32b, TARGET_MODEL_ID, msgs, k_this + 1)

        # ── Rejection sampling ───────────────────────────────────────────────
        acc_text, n_acc, correction, pos_mask = rejection_sample(d_list, t_list)

        for i, ok in enumerate(pos_mask):
            if i < K:
                per_pos_drafted[i] += 1
                if ok:
                    per_pos_accepted[i] += 1

        total_drafted  += k_this
        total_accepted += n_acc
        partial        += acc_text

        if correction:
            partial += correction

        thinking_tokens += k_this + (1 if correction else 0)

        if "</think>" in partial:
            done    = True
            idx     = partial.index("</think>") + len("</think>")
            partial = partial[:idx]

        print(f"  round {rounds:3d}: k={k_this}  accepted={n_acc}  "
              f"α_round={n_acc/k_this:.3f}  "
              f"thinking_toks≈{thinking_tokens}  "
              f"{'DONE' if done else '...'}",
              flush=True)

    if "</think>" not in partial:
        partial += "\n</think>"

    # ── Final answer from target ─────────────────────────────────────────────
    fa_resp = client_32b.chat.completions.create(
        model=TARGET_MODEL_ID,
        messages=base_msgs + [{"role": "assistant", "content": partial}],
        max_tokens=40960,
        temperature=1.0,
        top_p=0.95,
        seed=42,
        extra_body={
            "top_k": 20,
            "repetition_penalty": 1.0,
            "continue_final_message": True,
            "add_generation_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    final_answer = fa_resp.choices[0].message.content or ""

    t1      = time.perf_counter()
    latency = t1 - t0

    alpha = total_accepted / total_drafted if total_drafted > 0 else None
    mal   = total_accepted / rounds        if rounds > 0 else None
    per_pos_rate = [
        per_pos_accepted[i] / per_pos_drafted[i] if per_pos_drafted[i] > 0 else None
        for i in range(K)
    ]

    if alpha is None:
        speedup = None
    elif abs(1 - alpha) < 1e-9:
        speedup = float(K + 1)
    else:
        speedup = (1 - alpha ** (K + 1)) / (1 - alpha)

    def _s(v, fmt): return fmt.format(v) if v is not None else "n/a"

    print(f"  → QID {qid}: latency={latency:.1f}s  rounds={rounds}  "
          f"drafted={total_drafted}  accepted={total_accepted}  "
          f"α={_s(alpha, '{:.4f}')}  MAL={_s(mal, '{:.3f}')}  "
          f"theory={_s(speedup, '{:.2f}x')}")

    return {
        "qid":                     qid,
        "category":                item.get("category", ""),
        "reference_answer":        item.get("answer", ""),
        "thinking_text":           partial,
        "final_answer":            final_answer,
        "latency_s":               latency,
        "rounds":                  rounds,
        "total_drafted_tokens":    total_drafted,
        "total_accepted_tokens":   total_accepted,
        "acceptance_rate":         alpha,
        "mean_accepted_length":    mal,
        "theoretical_speedup":     speedup,
        "per_pos_acceptance_rate": per_pos_rate,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(42)

    client_32b = OpenAI(base_url=SERVER_32B_URL, api_key="none")
    client_2b  = OpenAI(base_url=SERVER_2B_URL,  api_key="none")

    print(f"Target (32B): {SERVER_32B_URL}  model={TARGET_MODEL_ID}")
    print(f"Draft  (2B):  {SERVER_2B_URL}   model={DRAFT_MODEL_ID}")
    print(f"k={K}  top_logprobs={TOP_LOGPROBS}\n")

    dataset = load_dataset(DATASET_PATH)
    print(f"Loaded {len(dataset)} questions\n")

    results = []

    for item in dataset:
        print(f"[QID {item['qid']:3d} | {item.get('category', '')}]")
        r = run_specdecode_question(client_32b, client_2b, item)
        results.append(r)
        print()

    total_latency  = sum(r["latency_s"]            for r in results)
    total_drafted  = sum(r["total_drafted_tokens"]  for r in results)
    total_accepted = sum(r["total_accepted_tokens"] for r in results)
    overall_alpha  = total_accepted / total_drafted if total_drafted > 0 else None

    if overall_alpha is None:
        overall_speedup = None
    elif abs(1 - overall_alpha) < 1e-9:
        overall_speedup = float(K + 1)
    else:
        overall_speedup = (1 - overall_alpha ** (K + 1)) / (1 - overall_alpha)

    def _mean(key):
        vals = [r[key] for r in results if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    summary = {
        "target_model":                TARGET_MODEL_ID,
        "draft_model":                 DRAFT_MODEL_ID,
        "k":                           K,
        "top_logprobs":                TOP_LOGPROBS,
        "total_questions":             len(results),
        "total_latency_s":             total_latency,
        "total_drafted_tokens":        total_drafted,
        "total_accepted_tokens":       total_accepted,
        "overall_acceptance_rate":     overall_alpha,
        "overall_theoretical_speedup": overall_speedup,
        "per_question":                results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {RESULTS_PATH}")

    def _f(v, fmt): return fmt.format(v) if v is not None else "n/a"

    print("\n--- Summary Table ---")
    print(f"{'QID':>5}  {'Category':<28}  {'Lat(s)':>7}  {'Rounds':>6}  "
          f"{'Drafted':>7}  {'Accepted':>8}  {'α':>7}  {'MAL':>5}  {'Theory':>7}")
    print("-" * 95)
    for r in results:
        print(
            f"{r['qid']:>5}  {r['category']:<28}  "
            f"{r['latency_s']:>7.2f}  {r['rounds']:>6}  "
            f"{r['total_drafted_tokens']:>7}  {r['total_accepted_tokens']:>8}  "
            f"{_f(r['acceptance_rate'],      '{:.4f}'):>7}  "
            f"{_f(r['mean_accepted_length'], '{:.3f}'):>5}  "
            f"{_f(r['theoretical_speedup'],  '{:.2f}x'):>7}"
        )
    print("-" * 95)
    print(
        f"{'ALL':>5}  {'':28}  "
        f"{total_latency:>7.2f}  {'---':>6}  "
        f"{total_drafted:>7}  {total_accepted:>8}  "
        f"{_f(overall_alpha,   '{:.4f}'):>7}  "
        f"{'---':>5}  "
        f"{_f(overall_speedup, '{:.2f}x'):>7}"
    )

    if per_pos_rate := _mean("per_pos_acceptance_rate"):
        pass  # per-question already printed; aggregate below
    all_per_pos = [0.0] * K
    all_per_pos_n = [0] * K
    for r in results:
        for i, v in enumerate(r["per_pos_acceptance_rate"]):
            if v is not None:
                all_per_pos[i] += v
                all_per_pos_n[i] += 1
    print("\nPer-position acceptance (averaged across questions):")
    for i in range(K):
        v = all_per_pos[i] / all_per_pos_n[i] if all_per_pos_n[i] > 0 else None
        print(f"  p{i}: {_f(v, '{:.4f}')}")


if __name__ == "__main__":
    main()
