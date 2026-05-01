"""
run_specReason.py
Step-level speculative reasoning for VLMs.
  Drafter:  Qwen3-VL-2B-Thinking  → generates each reasoning step cheaply
  Scorer:   Qwen3-VL-32B-Thinking  → scores the step; fallback if score < threshold
  Hardware: 2 nodes — 32B on NODE1:8000, 2B on NODE2:8001

Algorithm (per question)
------------------------
1. Drafter generates the next reasoning step (up to MAX_STEP_TOKENS tokens,
   stopping at a step boundary "\n\n" or the end-of-thinking "</think>").
2. Scorer rates the draft step 1–10 (greedy, 1 token).
3. If score >= SCORE_THRESHOLD: accept the draft step.
   Else: discard draft, scorer regenerates that step (fallback).
4. Cycle detection: if the last CYCLE_WINDOW step hashes repeat, force a fallback.
5. Loop until "</think>" is generated, then scorer produces the final \boxed{} answer.

Server URLs are read from environment variables set by job_vl_specReason.sh:
  VLLM_32B_URL  e.g. http://node001:8000/v1
  VLLM_2B_URL   e.g. http://node002:8001/v1
"""

import json
import os
import time

from openai import OpenAI

TARGET_MODEL_ID  = "Qwen/Qwen3-VL-32B-Thinking"
DRAFT_MODEL_ID   = "Qwen/Qwen3-VL-2B-Thinking"
SCORE_THRESHOLD  = 7          # accept draft step if score >= this
MAX_STEP_TOKENS  = 512        # max tokens per reasoning step
CYCLE_WINDOW     = 5          # detect cycle if same step seen within last N steps
DATASET_PATH     = "mathvision_vl.json"
RESULTS_DIR      = os.environ.get("RESULTS_DIR", "results")
RESULTS_PATH     = f"{RESULTS_DIR}/run_specReason_results.json"

SERVER_32B_URL   = os.environ.get("VLLM_32B_URL", "http://localhost:8000/v1")
SERVER_2B_URL    = os.environ.get("VLLM_2B_URL",  "http://localhost:8001/v1")

STEP_STOPS    = ["\n\n", "</think>"]
STEP_SAMPLING = dict(temperature=1.0, top_p=0.95, seed=42,
                     extra_body={"top_k": 20, "repetition_penalty": 1.0})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _image_content(image_b64: str | None) -> list:
    parts = []
    if image_b64:
        parts.append({"type": "image_url",
                      "image_url": {"url": f"data:image/png;base64,{image_b64}"}})
    return parts


def _generate_step(client: OpenAI, model: str, messages: list,
                   thinking_so_far: str) -> tuple[str, str]:
    """
    Continue the partial assistant turn (the in-progress <think> block).
    Returns (step_text, finish_reason) where finish_reason is "stop" or "length".
    """
    # Append the current thinking content as the partial assistant turn.
    # continue_final_message=True tells vLLM to treat the last assistant
    # message as a prefix to continue rather than a completed turn.
    messages_with_partial = messages + [
        {"role": "assistant", "content": thinking_so_far}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages_with_partial,
        max_tokens=MAX_STEP_TOKENS,
        stop=STEP_STOPS,
        temperature=STEP_SAMPLING["temperature"],
        top_p=STEP_SAMPLING["top_p"],
        seed=STEP_SAMPLING["seed"],
        extra_body={
            **STEP_SAMPLING["extra_body"],
            "continue_final_message": True,
            "add_generation_prompt": False,
            "chat_template_kwargs": {"enable_thinking": True},
        },
    )
    choice        = response.choices[0]
    step_text     = choice.message.content or ""
    finish_reason = choice.finish_reason  # "stop" | "length"
    return step_text, finish_reason


def _score_step(client_32b: OpenAI, question: str, image_b64: str | None,
                thinking_so_far: str, draft_step: str) -> int:
    """
    Ask 32B to rate the draft step 1–10 (greedy, single token).
    Returns int score; defaults to 5 on parse failure.
    """
    scoring_text = (
        "You are evaluating a reasoning step in a math solution.\n\n"
        f"Problem:\n{question}\n\n"
        f"Reasoning so far:\n{thinking_so_far.strip()}\n\n"
        f"Proposed next step:\n{draft_step.strip()}\n\n"
        "Rate this reasoning step for correctness and progress "
        "(1 = wrong/off-track, 10 = correct and useful). "
        "Reply with a single digit only."
    )
    content = _image_content(image_b64) + [{"type": "text", "text": scoring_text}]
    response = client_32b.chat.completions.create(
        model=TARGET_MODEL_ID,
        messages=[{"role": "user", "content": content}],
        max_tokens=2,
        temperature=0.0,   # greedy
        top_p=1.0,
    )
    raw = (response.choices[0].message.content or "").strip()
    try:
        return int(raw[0])
    except (ValueError, IndexError):
        return 5  # neutral fallback on parse error


def _is_cycle(step_text: str, recent_hashes: list[int]) -> bool:
    h = hash(step_text.strip())
    return h in recent_hashes


def _generate_final_answer(client_32b: OpenAI, messages: list,
                            completed_thinking: str) -> tuple[str, int]:
    """
    After </think>, use 32B to produce the final \boxed{} answer.
    completed_thinking should already end with </think>.
    Returns (answer_text, num_tokens).
    """
    messages_with_thinking = messages + [
        {"role": "assistant", "content": completed_thinking}
    ]
    response = client_32b.chat.completions.create(
        model=TARGET_MODEL_ID,
        messages=messages_with_thinking,
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
    answer    = response.choices[0].message.content or ""
    num_toks  = response.usage.completion_tokens
    return answer, num_toks


# ---------------------------------------------------------------------------
# SpecReason loop for a single question
# ---------------------------------------------------------------------------

def run_specReason_question(
    client_32b: OpenAI,
    client_2b:  OpenAI,
    item:       dict,
) -> dict:
    qid       = item["qid"]
    question  = item["question"]
    image_b64 = item.get("image_b64")
    prompt    = build_prompt(item)

    base_messages = [
        {"role": "user", "content": _image_content(image_b64) + [{"type": "text", "text": prompt}]},
    ]

    thinking   = "<think>\n"
    done       = False
    step_num   = 0
    recent_hashes: list[int] = []

    stats = {
        "steps_total":    0,
        "steps_accepted": 0,
        "steps_fallback": 0,
        "steps_cycle":    0,
        "scores":         [],
    }

    t0 = time.perf_counter()

    while not done:
        step_num += 1

        # ── Draft phase (2B) ──────────────────────────────────────────────
        draft_text, finish_reason = _generate_step(
            client_2b, DRAFT_MODEL_ID, base_messages, thinking
        )

        thinking_ended = ("</think>" in draft_text) or (finish_reason == "stop" and not draft_text.endswith("\n\n"))

        stats["steps_total"] += 1

        # Cycle detection
        if _is_cycle(draft_text, recent_hashes):
            stats["steps_cycle"]   += 1
            stats["steps_fallback"] += 1
            fallback, _ = _generate_step(
                client_32b, TARGET_MODEL_ID, base_messages, thinking
            )
            thinking += fallback
            recent_hashes.clear()
            done = "</think>" in fallback
            print(f"  step {step_num}: CYCLE → fallback ({len(fallback)} chars)")
            continue

        # Keep sliding window
        recent_hashes.append(hash(draft_text.strip()))
        if len(recent_hashes) > CYCLE_WINDOW:
            recent_hashes.pop(0)

        # If draft reached </think> or EOS, accept directly (no scoring needed)
        if thinking_ended:
            thinking += draft_text
            stats["steps_accepted"] += 1
            done = True
            print(f"  step {step_num}: draft → EOS/think-end ({len(draft_text)} chars)")
            break

        # ── Score phase (32B, greedy) ─────────────────────────────────────
        score = _score_step(client_32b, prompt, image_b64, thinking, draft_text)
        stats["scores"].append(score)

        if score >= SCORE_THRESHOLD:
            thinking += draft_text + "\n\n"
            stats["steps_accepted"] += 1
            print(f"  step {step_num}: ACCEPT  score={score}  ({len(draft_text)} chars)")
        else:
            # ── Fallback phase (32B generates this step) ──────────────────
            fallback, _ = _generate_step(
                client_32b, TARGET_MODEL_ID, base_messages, thinking
            )
            thinking += fallback + ("\n\n" if not fallback.endswith("\n\n") else "")
            stats["steps_fallback"] += 1
            print(f"  step {step_num}: REJECT  score={score} → fallback  ({len(fallback)} chars)")
            if "</think>" in fallback:
                done = True

    # Ensure thinking block is closed
    if "</think>" not in thinking:
        thinking += "\n</think>"

    # ── Final answer (always 32B) ─────────────────────────────────────────
    final_answer, final_toks = _generate_final_answer(client_32b, base_messages, thinking)

    t1 = time.perf_counter()
    latency    = t1 - t0
    # Count thinking tokens separately from final answer tokens
    # (approximation: full generated text)
    full_output  = thinking + final_answer
    # Use len(thinking.split()) as rough proxy; actual token count from API calls
    # is not directly available for the multi-call SpecReason loop, so we sum
    # usage from the final answer call and annotate that clearly.
    total_tokens_approx = len(full_output.split())  # word-level approx

    avg_score = sum(stats["scores"]) / len(stats["scores"]) if stats["scores"] else None

    print(
        f"  → total {step_num} steps  accepted={stats['steps_accepted']}  "
        f"fallback={stats['steps_fallback']}  cycle={stats['steps_cycle']}  "
        f"avg_score={f'{avg_score:.2f}' if avg_score is not None else 'n/a'}  "
        f"latency={latency:.1f}s"
    )

    return {
        "qid":                  qid,
        "category":             item.get("category", ""),
        "reference_answer":     item.get("answer", ""),
        "thinking_text":        thinking,
        "final_answer":         final_answer,
        "latency_s":            latency,
        "words_generated":      total_tokens_approx,
        "steps_total":          stats["steps_total"],
        "steps_accepted":       stats["steps_accepted"],
        "steps_fallback":       stats["steps_fallback"],
        "steps_cycle":          stats["steps_cycle"],
        "acceptance_rate":      stats["steps_accepted"] / stats["steps_total"] if stats["steps_total"] else None,
        "average_step_score":   avg_score,
        "per_step_scores":      stats["scores"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    client_32b = OpenAI(base_url=SERVER_32B_URL, api_key="none")
    client_2b  = OpenAI(base_url=SERVER_2B_URL,  api_key="none")

    print(f"Scorer  (32B): {SERVER_32B_URL}  model={TARGET_MODEL_ID}")
    print(f"Drafter (2B):  {SERVER_2B_URL}   model={DRAFT_MODEL_ID}")
    print(f"score_threshold={SCORE_THRESHOLD}  max_step_tokens={MAX_STEP_TOKENS}  "
          f"cycle_window={CYCLE_WINDOW}\n")

    dataset = load_dataset(DATASET_PATH)
    print(f"Loaded {len(dataset)} questions from {DATASET_PATH}\n")

    per_question_results = []

    for item in dataset:
        qid      = item["qid"]
        category = item.get("category", "")
        print(f"[QID {qid:3d} | {category}]")

        result = run_specReason_question(client_32b, client_2b, item)
        per_question_results.append(result)
        print()

    total_latency   = sum(r["latency_s"] for r in per_question_results)
    total_steps     = sum(r["steps_total"] for r in per_question_results)
    total_accepted  = sum(r["steps_accepted"] for r in per_question_results)
    total_fallback  = sum(r["steps_fallback"] for r in per_question_results)
    overall_accept  = total_accepted / total_steps if total_steps > 0 else None

    all_scores = [s for r in per_question_results for s in r["per_step_scores"]]
    overall_avg_score = sum(all_scores) / len(all_scores) if all_scores else None

    summary = {
        "target_model":              TARGET_MODEL_ID,
        "draft_model":               DRAFT_MODEL_ID,
        "score_threshold":           SCORE_THRESHOLD,
        "max_step_tokens":           MAX_STEP_TOKENS,
        "cycle_window":              CYCLE_WINDOW,
        "total_questions":           len(per_question_results),
        "total_latency_s":           total_latency,
        "total_steps":               total_steps,
        "total_steps_accepted":      total_accepted,
        "total_steps_fallback":      total_fallback,
        "overall_step_acceptance_rate": overall_accept,
        "overall_average_step_score":   overall_avg_score,
        "per_question":              per_question_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {RESULTS_PATH}")

    print("\n--- Summary Table ---")
    print(f"{'QID':>5}  {'Category':<28}  {'Lat(s)':>7}  {'Steps':>5}  {'Acc':>5}  {'Fall':>5}  {'AvgScore':>9}")
    print("-" * 72)
    for r in per_question_results:
        avg_s = f"{r['average_step_score']:.2f}" if r['average_step_score'] is not None else "n/a"
        print(f"{r['qid']:>5}  {r['category']:<28}  {r['latency_s']:>7.2f}  "
              f"{r['steps_total']:>5}  {r['steps_accepted']:>5}  "
              f"{r['steps_fallback']:>5}  {avg_s:>9}")
    print("-" * 72)
    accept_str = f"{overall_accept:.3f}" if overall_accept is not None else "n/a"
    score_str  = f"{overall_avg_score:.2f}" if overall_avg_score is not None else "n/a"
    print(f"{'ALL':>5}  {'':28}  {total_latency:>7.2f}  {total_steps:>5}  "
          f"{total_accepted:>5}  {total_fallback:>5}  {score_str:>9}")
    print(f"\nOverall step acceptance rate: {accept_str}")


if __name__ == "__main__":
    main()
