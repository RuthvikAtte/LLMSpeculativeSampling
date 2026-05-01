"""
run_h200_benchmark.py

Speculative decoding benchmark for Qwen3-VL-32B-Thinking on a single H200.
Uses vLLM native draft-model spec decode (Qwen3-VL-2B-Thinking, k=5).

Subcommands:
  run      — run prompts against a live vLLM server, save JSON
  compare  — print side-by-side comparison of two saved JSONs

Usage (called by job_h200_benchmark.sh):
  python run_h200_benchmark.py run --mode spec     --out results/spec.json
  python run_h200_benchmark.py run --mode baseline --out results/baseline.json
  python run_h200_benchmark.py compare results/spec.json results/baseline.json
"""

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

import requests
from openai import OpenAI

TARGET_MODEL    = "Qwen/Qwen3-VL-32B-Thinking"
NUM_SPEC_TOKENS = 5
DEFAULT_SERVER  = os.environ.get("VLLM_SERVER_BASE", "http://localhost:8000")

SAMPLING_KWARGS = dict(
    temperature=1.0,
    top_p=0.95,
    presence_penalty=0.0,
    max_tokens=2048,
    seed=42,
    extra_body={
        "top_k": 20,
        "repetition_penalty": 1.0,
        "chat_template_kwargs": {"enable_thinking": True},
    },
)

# ---------------------------------------------------------------------------
# Hardcoded benchmark prompts (text-only + image)
# ---------------------------------------------------------------------------

TEXT_PROMPTS = [
    {
        "id":   "algebra_quadratic",
        "text": 'Find all real solutions to x² - 7x + 10 = 0 using factoring. '
                'Show every step and put your final answer in "\\boxed{}".',
    },
    {
        "id":   "arithmetic_gauss",
        "text": 'Compute the sum 1 + 2 + 3 + ... + 200 using Gauss\'s formula. '
                'Verify by an alternative method and put the answer in "\\boxed{}".',
    },
    {
        "id":   "geometry_inscribed",
        "text": 'A circle is inscribed in a square with side length 10 cm. '
                'What fraction of the square\'s area is inside the circle? '
                'Express as a decimal to 4 places and put in "\\boxed{}".',
    },
    {
        "id":   "systems_linear",
        "text": 'Solve the system: 3x + 2y = 16  and  x - y = 2. '
                'Show all algebraic steps and put the solution (x, y) in "\\boxed{}".',
    },
    {
        "id":   "train_problem",
        "text": 'A train leaves city A at 08:00 traveling at 90 km/h. '
                'A second train leaves city B (270 km away) at 09:00 traveling '
                'toward city A at 60 km/h. At what time do they meet? '
                'Show your work and put the answer in "\\boxed{}".',
    },
    {
        "id":   "pythagorean",
        "text": 'A right triangle has legs of length 5 and 12. '
                'Find the hypotenuse length and the area of the triangle. '
                'Put both answers in "\\boxed{}".',
    },
]

IMAGE_PROMPTS = [
    {
        "id":        "img_describe",
        "text":      "Describe this image in detail. What concept or idea does it illustrate? "
                     "Be specific about any notation, diagrams, or visual elements you observe.",
        "use_image": True,
    },
    {
        "id":        "img_algorithm",
        "text":      "Based on the figure, explain the algorithmic or mathematical idea "
                     "being illustrated. What is the high-level intuition? "
                     'Put your main conclusion in "\\boxed{}".',
        "use_image": True,
    },
]


# ---------------------------------------------------------------------------
# Image loading (uses imgs/sps.jpg from the repo; falls back to a PIL dummy)
# ---------------------------------------------------------------------------

def load_image_b64(img_path: str) -> str | None:
    path = Path(img_path)
    if path.exists():
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    # Fallback: generate a minimal placeholder image with PIL
    try:
        import io
        from PIL import Image, ImageDraw, ImageFont

        img  = Image.new("RGB", (256, 128), color=(30, 30, 80))
        draw = ImageDraw.Draw(img)
        draw.rectangle([8, 8, 248, 120], outline=(120, 180, 255), width=2)
        draw.text((20, 30), "Speculative Sampling", fill=(255, 255, 255))
        draw.text((20, 60), "draft → verify → accept/reject", fill=(200, 200, 200))
        draw.text((20, 90), f"k = {NUM_SPEC_TOKENS} draft tokens", fill=(180, 220, 180))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as exc:
        print(f"WARNING: Could not load or generate image ({exc}). "
              "Image prompts will be sent text-only.")
        return None


def build_messages(prompt: dict, image_b64: str | None) -> list:
    content = []
    if prompt.get("use_image") and image_b64:
        content.append({
            "type":      "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
        })
    content.append({"type": "text", "text": prompt["text"]})
    return [{"role": "user", "content": content}]


# ---------------------------------------------------------------------------
# Prometheus helpers (only meaningful against a spec-decode server)
# ---------------------------------------------------------------------------

def _scrape_metrics(server_base: str) -> dict:
    try:
        text = requests.get(f"{server_base}/metrics", timeout=5).text
    except Exception:
        return {}
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


def _derive_alpha_metrics(diff: dict) -> tuple[float | None, float | None, float | None]:
    drafted  = diff["drafted"]
    accepted = diff["accepted"]
    rounds   = diff["drafts"]

    alpha = accepted / drafted if drafted > 0 else None
    mal   = (1.0 + accepted / rounds) if rounds > 0 else None

    if alpha is None:
        speedup = None
    elif abs(1 - alpha) < 1e-9:
        speedup = float(NUM_SPEC_TOKENS + 1)
    else:
        k = NUM_SPEC_TOKENS
        speedup = (1 - alpha ** (k + 1)) / (1 - alpha)

    return alpha, mal, speedup


# ---------------------------------------------------------------------------
# Run benchmark against a live server
# ---------------------------------------------------------------------------

def run_benchmark(server_base: str, mode: str, image_b64: str | None) -> list[dict]:
    client  = OpenAI(base_url=f"{server_base}/v1", api_key="none")
    is_spec = (mode == "spec")
    prompts = TEXT_PROMPTS + IMAGE_PROMPTS

    print(f"\n{'='*70}")
    print(f"Mode : {mode.upper()}  |  server: {server_base}")
    print(f"Model: {TARGET_MODEL}")
    if is_spec:
        print(f"Draft: Qwen/Qwen3-VL-2B-Thinking  k={NUM_SPEC_TOKENS}")
    print(f"Prompts: {len(TEXT_PROMPTS)} text + {len(IMAGE_PROMPTS)} image = {len(prompts)} total")
    print(f"{'='*70}")

    results = []
    for prompt in prompts:
        pid      = prompt["id"]
        messages = build_messages(prompt, image_b64)
        has_img  = bool(prompt.get("use_image") and image_b64)

        snap_before = _spec_snapshot(_scrape_metrics(server_base)) if is_spec else None

        t0       = time.perf_counter()
        response = client.chat.completions.create(
            model=TARGET_MODEL,
            messages=messages,
            **SAMPLING_KWARGS,
        )
        t1 = time.perf_counter()

        snap_after = _spec_snapshot(_scrape_metrics(server_base)) if is_spec else None

        n_tokens = response.usage.completion_tokens
        latency  = t1 - t0
        tps      = n_tokens / latency if latency > 0 else 0.0

        alpha, mal, speedup, per_pos = None, None, None, None
        if is_spec and snap_before and snap_after:
            diff             = _diff_snapshots(snap_before, snap_after)
            alpha, mal, speedup = _derive_alpha_metrics(diff)
            rounds           = diff["drafts"]
            per_pos          = (
                [v / rounds for v in diff["per_pos"]] if rounds > 0 else None
            )

        result = {
            "id":        pid,
            "has_image": has_img,
            "latency_s": latency,
            "tokens":    n_tokens,
            "tps":       tps,
            "alpha":     alpha,
            "mal":       mal,
            "speedup":   speedup,
            "per_pos_acceptance_rate": per_pos,
        }
        results.append(result)

        img_tag   = "[img]" if has_img else "     "
        alpha_s   = f"{alpha:.4f}"    if alpha   is not None else "  n/a"
        mal_s     = f"{mal:.3f}"      if mal     is not None else "  n/a"
        speedup_s = f"{speedup:.2f}x" if speedup is not None else "   n/a"
        print(
            f"  {pid:<22} {img_tag}  lat={latency:6.2f}s  "
            f"tok={n_tokens:5d}  tok/s={tps:7.1f}  "
            f"α={alpha_s}  MAL={mal_s}  theory={speedup_s}"
        )
        if per_pos:
            rates = "  ".join(f"p{i}={v:.3f}" for i, v in enumerate(per_pos))
            print(f"    per-position: {rates}")

    return results


# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------

def print_comparison(spec_results: list[dict], base_results: list[dict]) -> None:
    spec_by_id = {r["id"]: r for r in spec_results}
    base_by_id = {r["id"]: r for r in base_results}
    all_ids    = [r["id"] for r in base_results]

    W = 104
    print("\n" + "=" * W)
    print("  COMPARISON SUMMARY — Speculative Decode vs Baseline")
    print("=" * W)
    print(f"  {'Prompt ID':<24}  {'Baseline tok/s':>14}  {'Spec tok/s':>10}  "
          f"{'Emp. speedup':>12}  {'α (accept)':>10}  {'MAL':>5}  {'Theory speedup':>14}")
    print("-" * W)

    emp_speedups, alphas, base_tps_all, spec_tps_all = [], [], [], []

    for pid in all_ids:
        b = base_by_id.get(pid, {})
        s = spec_by_id.get(pid, {})
        b_tps = b.get("tps", 0.0)
        s_tps = s.get("tps", 0.0)
        alpha = s.get("alpha")
        mal   = s.get("mal")
        theory_speedup = s.get("speedup")

        emp_sp = s_tps / b_tps if b_tps > 0 else None

        emp_s    = f"{emp_sp:.2f}x"       if emp_sp       is not None else "    n/a"
        alpha_s  = f"{alpha:.4f}"          if alpha        is not None else "       n/a"
        mal_s    = f"{mal:.3f}"            if mal          is not None else "  n/a"
        theory_s = f"{theory_speedup:.2f}x" if theory_speedup is not None else "           n/a"

        print(f"  {pid:<24}  {b_tps:>14.1f}  {s_tps:>10.1f}  "
              f"{emp_s:>12}  {alpha_s:>10}  {mal_s:>5}  {theory_s:>14}")

        base_tps_all.append(b_tps)
        spec_tps_all.append(s_tps)
        if emp_sp is not None:
            emp_speedups.append(emp_sp)
        if alpha is not None:
            alphas.append(alpha)

    print("-" * W)

    avg_base  = sum(base_tps_all) / len(base_tps_all) if base_tps_all else 0.0
    avg_spec  = sum(spec_tps_all) / len(spec_tps_all) if spec_tps_all else 0.0
    avg_emp   = sum(emp_speedups) / len(emp_speedups)  if emp_speedups  else None
    avg_alpha = sum(alphas)       / len(alphas)        if alphas        else None

    avg_emp_s   = f"{avg_emp:.2f}x"  if avg_emp   is not None else "    n/a"
    avg_alpha_s = f"{avg_alpha:.4f}" if avg_alpha is not None else "       n/a"

    avg_theory = None
    if avg_alpha is not None:
        k = NUM_SPEC_TOKENS
        avg_theory = (1 - avg_alpha ** (k + 1)) / (1 - avg_alpha)
    avg_theory_s = f"{avg_theory:.2f}x" if avg_theory is not None else "           n/a"

    print(f"  {'AVERAGE':<24}  {avg_base:>14.1f}  {avg_spec:>10.1f}  "
          f"{avg_emp_s:>12}  {avg_alpha_s:>10}  {'':>5}  {avg_theory_s:>14}")
    print("=" * W)

    print("\n  Key metrics:")
    print(f"    Avg baseline throughput : {avg_base:.1f} tok/s")
    print(f"    Avg spec-decode throughput: {avg_spec:.1f} tok/s")
    if avg_emp is not None:
        print(f"    Empirical speedup (tok/s ratio) : {avg_emp:.2f}x")
    if avg_alpha is not None:
        print(f"    Avg token acceptance rate (α)   : {avg_alpha:.4f}")
    if avg_theory is not None:
        print(f"    Theoretical speedup from α       : {avg_theory:.2f}x")
        print(f"    (formula: (1 - α^(k+1)) / (1 - α)  with k={NUM_SPEC_TOKENS})")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p   = argparse.ArgumentParser(description="H200 spec-decode benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Run benchmark against a live vLLM server")
    run_p.add_argument("--mode",   choices=["spec", "baseline"], required=True,
                       help="Spec-decode or standalone baseline mode")
    run_p.add_argument("--server", default=DEFAULT_SERVER,
                       help="vLLM server base URL (default: $VLLM_SERVER_BASE or http://localhost:8000)")
    run_p.add_argument("--out",    required=True, help="Output JSON path")
    run_p.add_argument("--image",  default="imgs/sps.jpg",
                       help="Path to image file for image prompts (default: imgs/sps.jpg)")

    cmp_p = sub.add_parser("compare", help="Compare two saved result JSONs")
    cmp_p.add_argument("spec_json",     help="JSON from spec-decode run")
    cmp_p.add_argument("baseline_json", help="JSON from baseline run")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.cmd == "compare":
        with open(args.spec_json)     as f:
            spec_data = json.load(f)
        with open(args.baseline_json) as f:
            base_data = json.load(f)
        print_comparison(spec_data["results"], base_data["results"])
        return

    # --- run ---
    image_b64 = load_image_b64(args.image)

    results = run_benchmark(args.server, args.mode, image_b64)

    # Aggregate
    total_lat  = sum(r["latency_s"] for r in results)
    total_tok  = sum(r["tokens"]    for r in results)
    avg_tps    = total_tok / total_lat if total_lat > 0 else 0.0
    alphas     = [r["alpha"]   for r in results if r["alpha"]   is not None]
    avg_alpha  = sum(alphas) / len(alphas) if alphas else None

    summary = {
        "mode":            args.mode,
        "server":          args.server,
        "target_model":    TARGET_MODEL,
        "num_spec_tokens": NUM_SPEC_TOKENS if args.mode == "spec" else None,
        "total_latency_s": total_lat,
        "total_tokens":    total_tok,
        "avg_tps":         avg_tps,
        "avg_alpha":       avg_alpha,
        "results":         results,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {args.out}")

    print(f"\n--- {args.mode.upper()} Summary ---")
    print(f"  Prompts run    : {len(results)}")
    print(f"  Total latency  : {total_lat:.2f}s")
    print(f"  Total tokens   : {total_tok}")
    print(f"  Avg tok/s      : {avg_tps:.1f}")
    if avg_alpha is not None:
        k      = NUM_SPEC_TOKENS
        theory = (1 - avg_alpha ** (k + 1)) / (1 - avg_alpha)
        print(f"  Avg α          : {avg_alpha:.4f}")
        print(f"  Theory speedup : {theory:.2f}x  (k={k})")


if __name__ == "__main__":
    main()
