"""
Hieu chinh SCORE_THRESHOLD bang query in-domain va out-of-domain that, dung
dung dai luong ma retrieve() dem so sanh: best_dense_score = dense[0]["score"].

Dat ngoai src/ de khong bi test import nham (test_public_function_signatures
chi import src.task*, khong quet thu muc scripts/).

Cach dung:
    python scripts/calibrate_threshold.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.task5_semantic_search import semantic_search  # noqa: E402

DATASET_PATH = ROOT / "group_project" / "evaluation" / "golden_dataset.json"
OUTPUT_PATH = ROOT / "reports" / "threshold_calibration.json"

# Near-miss OOD: cung linh vuc thi vet hoc thuat nhung khong co trong corpus
# (chi thu thap IELTS Writing). Near-miss moi thuc su dinh ra bien, cau hoi
# vo thuong vo phat thi threshold nao cung dung.
OOD_QUERIES = [
    "What is the cue card format in IELTS Speaking Part 2?",
    "How many sections does the IELTS Listening test have?",
    "TOEFL Writing được chấm theo thang điểm nào?",
    "What is the PTE Academic writing scoring rubric?",
    "How do I book an IELTS Listening and Reading test?",
    "What is the format of the IELTS Reading test?",
    "How is the Cambridge English Proficiency exam scored?",
    "IELTS Speaking Part 3 hỏi những dạng câu nào?",
    "What is the passing score for the Duolingo English Test?",
    "How long is the IELTS Speaking interview?",
]


def _in_domain_queries() -> list[str]:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    return [
        case["question"] for case in dataset if case.get("query_type") != "out_of_domain"
    ]


def _best_score(query: str) -> float:
    results = semantic_search(query, top_k=1)
    return results[0]["score"] if results else 0.0


def calibrate() -> dict:
    in_domain_scores = [_best_score(q) for q in _in_domain_queries()]
    ood_scores = [_best_score(q) for q in OOD_QUERIES]

    best_threshold = 0.3
    best_tnr = -1.0
    sweep = []
    for i in range(10, 81):
        t = i / 100
        tpr = sum(1 for s in in_domain_scores if s >= t) / len(in_domain_scores)
        tnr = sum(1 for s in ood_scores if s < t) / len(ood_scores)
        sweep.append({"threshold": t, "tpr": tpr, "tnr": tnr})
        if tpr >= 0.95 and tnr > best_tnr:
            best_tnr = tnr
            best_threshold = t

    return {
        "recommended_threshold": best_threshold,
        "tpr_at_threshold": next(s["tpr"] for s in sweep if s["threshold"] == best_threshold),
        "tnr_at_threshold": best_tnr,
        "in_domain_scores": in_domain_scores,
        "ood_scores": ood_scores,
        "sweep": sweep,
    }


def main() -> None:
    result = calibrate()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"In-domain scores: min={min(result['in_domain_scores']):.3f} max={max(result['in_domain_scores']):.3f}")
    print(f"OOD scores:       min={min(result['ood_scores']):.3f} max={max(result['ood_scores']):.3f}")
    print(f"Recommended SCORE_THRESHOLD = {result['recommended_threshold']}")
    print(f"  TPR (in-domain retained) = {result['tpr_at_threshold']:.2f}")
    print(f"  TNR (OOD correctly refused) = {result['tnr_at_threshold']:.2f}")
    print(f"Full sweep written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
