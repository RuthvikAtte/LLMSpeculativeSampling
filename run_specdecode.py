"""
run_specdecode.py
Speculative decoding via vLLM's native draft-model method.
  Target: Qwen/Qwen3-32B  (TP=4)
  Draft:  Qwen/Qwen3-1.7B (draft_tensor_parallel_size=4)
  num_speculative_tokens=5
Hardware: 4x A100 40GB, Perlmutter

Alpha measurement
-----------------
vLLM tracks spec-decode counters in SpecDecodingLogging, which runs in
the FRONTEND process.  By default offline LLM disables the stat pipeline
entirely (disable_log_stats=True), so we:
  1. Pass disable_log_stats=False to re-enable it.
  2. Monkey-patch SpecDecodingLogging.observe() (class-level, before LLM
     creation) to intercept raw num_draft_tokens / num_accepted_tokens on
     every engine step.
  3. Suppress the periodic 10-second log spam by patching log() to a no-op.
  4. Call tracker.reset() before each generate() → per-question α window.
"""

import json
import os
import time

# ---------------------------------------------------------------------------
# Alpha tracker — installed BEFORE vLLM creates the engine
# ---------------------------------------------------------------------------
from vllm.v1.spec_decode.metrics import SpecDecodingLogging, SpecDecodingStats


class SpecDecodeAlphaTracker:
    """
    Intercepts SpecDecodingLogging to accumulate raw counters per question.

    Metrics (matching the paper's notation, Leviathan et al. 2023):
      alpha                  = accepted / drafted          (§3.1, Definition 3.1)
      mean_acceptance_length = 1 + accepted / rounds       (Eq. 1, incl. bonus token)
      per_pos_rate[i]        = P(pos i accepted) over all rounds (marginal)
      theoretical_speedup    = (1 - α^(γ+1)) / (1 - α)   (Theorem 3.8, c → 0)
    """

    def __init__(self, num_spec_tokens: int):
        self._num_spec_tokens = num_spec_tokens
        self.reset()

        _orig_observe = SpecDecodingLogging.observe
        _s = self

        def _patched_observe(logger_self, stats: SpecDecodingStats):
            _orig_observe(logger_self, stats)
            _s._draft    += stats.num_draft_tokens
            _s._accepted += stats.num_accepted_tokens
            _s._rounds   += stats.num_drafts
            if stats.num_accepted_tokens_per_pos:
                if _s._per_pos is None:
                    _s._per_pos = [0] * len(stats.num_accepted_tokens_per_pos)
                for i, v in enumerate(stats.num_accepted_tokens_per_pos):
                    _s._per_pos[i] += v

        # Suppress the periodic "SpecDecoding metrics: ..." log line so stdout
        # stays clean; our summary table replaces it.
        SpecDecodingLogging.observe = _patched_observe
        SpecDecodingLogging.log    = lambda *_, **__: None

        self._orig_observe = _orig_observe

    def reset(self):
        self._draft    = 0
        self._accepted = 0
        self._rounds   = 0
        self._per_pos: list[int] | None = None

    # ------------------------------------------------------------------
    # Derived metrics
    # ------------------------------------------------------------------

    @property
    def alpha(self) -> float | None:
        return self._accepted / self._draft if self._draft > 0 else None

    @property
    def mean_acceptance_length(self) -> float | None:
        """Expected tokens per round, including the bonus token."""
        return (1.0 + self._accepted / self._rounds) if self._rounds > 0 else None

    @property
    def per_pos_rate(self) -> list[float] | None:
        """Marginal acceptance rate at each draft position (drops with depth)."""
        if self._per_pos is None or self._rounds == 0:
            return None
        return [v / self._rounds for v in self._per_pos]

    @property
    def theoretical_speedup(self) -> float | None:
        """
        Upper-bound walltime speedup from Theorem 3.8 (Leviathan 2023),
        assuming negligible draft cost (c → 0):
            (1 - α^(γ+1)) / (1 - α)
        """
        a = self.alpha
        if a is None:
            return None
        gamma = len(self._per_pos) if self._per_pos else self._num_spec_tokens
        if abs(1 - a) < 1e-9:
            return float(gamma + 1)
        return (1 - a ** (gamma + 1)) / (1 - a)

    def restore(self):
        SpecDecodingLogging.observe = self._orig_observe


# ---------------------------------------------------------------------------
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Config
# ---------------------------------------------------------------------------
TARGET_MODEL_ID = "Qwen/Qwen3-32B"
DRAFT_MODEL_ID  = "Qwen/Qwen3-1.7B"
NUM_SPEC_TOKENS = 5
DATASET_PATH    = "mathvision_mini.json"
RESULTS_DIR     = os.environ.get("RESULTS_DIR", "results")
RESULTS_PATH    = f"{RESULTS_DIR}/run_specdecode_results.json"
SYSTEM_PREFIX   = "Please reason step by step, and put your final answer within \\boxed{}."

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
        return json.load(f)


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

    # Install tracker before LLM creation so the class-level patch is in place
    # when LoggingStatLogger instantiates SpecDecodingLogging inside LLM.__init__.
    tracker = SpecDecodeAlphaTracker(NUM_SPEC_TOKENS)

    print(f"Loading target model: {TARGET_MODEL_ID}")
    print(f"Draft model:          {DRAFT_MODEL_ID}  (num_spec_tokens={NUM_SPEC_TOKENS})")

    tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL_ID)
    llm = LLM(
        model=TARGET_MODEL_ID,
        tensor_parallel_size=4,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=0.7,
        disable_log_stats=False,   # enable stat pipeline so observe() is called
        speculative_config={
            "method": "draft_model",
            "model": DRAFT_MODEL_ID,
            "num_speculative_tokens": NUM_SPEC_TOKENS,
            "draft_tensor_parallel_size": 4,
        },
    )

    dataset = load_dataset(DATASET_PATH)
    print(f"Loaded {len(dataset)} questions from {DATASET_PATH}\n")

    per_question_results = []

    for idx, item in enumerate(dataset):
        question = item["question"]
        answer   = item.get("answer", "")
        prompt   = build_prompt(question, tokenizer)

        tracker.reset()                           # start fresh α window for this Q
        t0 = time.perf_counter()
        outputs = llm.generate([prompt], SAMPLING_PARAMS)
        t1 = time.perf_counter()                  # all observe() calls are now done

        output         = outputs[0]
        generated_text = output.outputs[0].text
        num_tokens     = len(output.outputs[0].token_ids)
        latency        = t1 - t0
        tps            = num_tokens / latency if latency > 0 else 0.0

        alpha     = tracker.alpha
        mal       = tracker.mean_acceptance_length
        speedup_t = tracker.theoretical_speedup
        per_pos   = tracker.per_pos_rate

        result = {
            "question_idx":             idx,
            "question":                 question,
            "reference_answer":         answer,
            "generated_text":           generated_text,
            "latency_s":                latency,
            "tokens_generated":         num_tokens,
            "tokens_per_second":        tps,
            "acceptance_rate":          alpha,
            "mean_acceptance_length":   mal,
            "theoretical_speedup":      speedup_t,
            "per_pos_acceptance_rate":  per_pos,
        }
        per_question_results.append(result)

        alpha_str   = f"{alpha:.4f}"      if alpha     is not None else "n/a"
        mal_str     = f"{mal:.3f}"        if mal       is not None else "n/a"
        speedup_str = f"{speedup_t:.2f}x" if speedup_t is not None else "n/a"
        print(
            f"[Q{idx}] latency={latency:.2f}s  tokens={num_tokens}  tok/s={tps:.1f}"
            f"  α={alpha_str}  MAL={mal_str}  theory_speedup={speedup_str}"
        )
        if per_pos:
            rates = "  ".join(f"p{i}={v:.3f}" for i, v in enumerate(per_pos))
            print(f"       per-position: {rates}")

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    total_latency = sum(r["latency_s"] for r in per_question_results)
    total_tokens  = sum(r["tokens_generated"] for r in per_question_results)
    avg_tps       = total_tokens / total_latency if total_latency > 0 else 0.0

    def _mean(key):
        vals = [r[key] for r in per_question_results if r[key] is not None]
        return sum(vals) / len(vals) if vals else None

    avg_alpha   = _mean("acceptance_rate")
    avg_mal     = _mean("mean_acceptance_length")
    avg_speedup = _mean("theoretical_speedup")

    summary = {
        "target_model":                   TARGET_MODEL_ID,
        "draft_model":                    DRAFT_MODEL_ID,
        "num_speculative_tokens":         NUM_SPEC_TOKENS,
        "total_questions":                len(per_question_results),
        "total_latency_s":                total_latency,
        "total_tokens_generated":         total_tokens,
        "average_tokens_per_second":      avg_tps,
        "average_acceptance_rate":        avg_alpha,
        "average_mean_acceptance_length": avg_mal,
        "average_theoretical_speedup":    avg_speedup,
        "per_question":                   per_question_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    def _f(v, fmt):
        return fmt.format(v) if v is not None else "n/a"

    print("\n--- Summary Table ---")
    print(f"{'Q':>3}  {'Lat(s)':>7}  {'Toks':>6}  {'Tok/s':>7}  "
          f"{'α':>7}  {'MAL':>5}  {'Theory':>7}")
    print("-" * 56)
    for r in per_question_results:
        print(
            f"{r['question_idx']:>3}  "
            f"{r['latency_s']:>7.2f}  "
            f"{r['tokens_generated']:>6}  "
            f"{r['tokens_per_second']:>7.1f}  "
            f"{_f(r['acceptance_rate'],       '{:.4f}'):>7}  "
            f"{_f(r['mean_acceptance_length'], '{:.3f}'):>5}  "
            f"{_f(r['theoretical_speedup'],    '{:.2f}x'):>7}"
        )
    print("-" * 56)
    print(
        f"{'ALL':>3}  "
        f"{total_latency:>7.2f}  "
        f"{total_tokens:>6}  "
        f"{avg_tps:>7.1f}  "
        f"{_f(avg_alpha,   '{:.4f}'):>7}  "
        f"{_f(avg_mal,     '{:.3f}'):>5}  "
        f"{_f(avg_speedup, '{:.2f}x'):>7}"
    )

    tracker.restore()


if __name__ == "__main__":
    main()
