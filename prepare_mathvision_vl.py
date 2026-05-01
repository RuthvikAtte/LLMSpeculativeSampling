"""
prepare_mathvision_vl.py
Download and save the 6 MathVision questions used in the VL experiment.
Run ONCE on a login node with internet access before submitting the job.

  HF_HOME=/scratch/bgum/ratte/hf_cache python3 prepare_mathvision_vl.py
"""

import base64
import io
import json
import os
import sys

TARGET_QIDS  = [5, 133, 123, 168, 117, 104]
OUTPUT_PATH  = "mathvision_vl.json"
HF_HOME      = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

# Category labels for documentation (matches experiment spec)
QID_META = {
    5:   {"category": "Arithmetic",              "expected_tokens": 317},
    133: {"category": "Algebra",                 "expected_tokens": 1967},
    123: {"category": "Descriptive geometry",    "expected_tokens": 3382},
    168: {"category": "Metric geometry - area",  "expected_tokens": 4804},
    117: {"category": "Logic",                   "expected_tokens": 8482},
    104: {"category": "Solid geometry",          "expected_tokens": 10458},
}


def img_to_b64(img) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("Installing datasets...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "-q"])
        from datasets import load_dataset

    print(f"Loading MathVision dataset (HF_HOME={HF_HOME})...")
    ds = load_dataset("MathLLMs/MathVision", split="test")
    print(f"Dataset loaded — {len(ds)} total questions")

    items = []
    for qid in TARGET_QIDS:
        row = ds[qid]

        # Handle various field name conventions across dataset versions
        question = row.get("question") or row.get("problem") or row.get("text", "")
        answer   = row.get("answer") or row.get("solution", "")
        category = row.get("subject") or row.get("category") or QID_META[qid]["category"]
        options  = row.get("options", "")

        image    = row.get("decoded_image") or row.get("image") or row.get("figure")
        image_b64 = img_to_b64(image) if image is not None and not isinstance(image, str) else None

        item = {
            "qid":               qid,
            "question":          question,
            "options":           options,
            "answer":            answer,
            "category":          category,
            "expected_tokens":   QID_META[qid]["expected_tokens"],
            "image_b64":         image_b64,
        }
        items.append(item)
        has_img = "✓ image" if image_b64 else "✗ no image"
        print(f"  QID {qid:3d} [{category}]  {has_img}  q={question[:60]}...")

    with open(OUTPUT_PATH, "w") as f:
        json.dump(items, f)
    print(f"\nSaved {len(items)} questions → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
