"""
Eval runner: chay golden dataset qua Config A (dense-only) va Config B
(hybrid+RRF), cham 4 metric RAGAS + 1 metric khong dung LLM (context_hit),
va ghi ket qua tho + summary vao group_project/evaluation/results/.

Vi eval_runner khong import gi tu `ragas` truc tiep (chi qua eval_ragas.py),
neu ban RAGAS sau nay doi API thi chi eval_ragas.py can sua.

Cach chay:
    python -m src.eval_runner --config both
    python -m src.eval_runner --config both --limit 3 --skip-metrics   # smoke test re tien
    python -m src.eval_runner --config A                                # chi Config A
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from .eval_config import CONFIG_A, CONFIG_B, DEFAULT_TOP_K, RunConfig
from .observability import answer_with_trace

ROOT = Path(__file__).parent.parent
DATASET_PATH = ROOT / "group_project" / "evaluation" / "golden_dataset.json"
RESULTS_DIR = ROOT / "group_project" / "evaluation" / "results"
CACHE_DIR = RESULTS_DIR / "cache"


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", text).strip()


def _context_hit(expected_context: str, contexts: list[str]) -> bool:
    """Metric khong dung LLM: expected_context (chuan hoa) co la substring
    cua bat ky chunk nao da retrieve khong. Dung lam luoi an toan neu RAGAS
    truc trac ngay hom demo."""
    needle = _normalize(expected_context)
    if not needle or needle.startswith("(no supporting context"):
        return None  # ca out_of_domain khong co evidence that de so sanh
    haystack = " ".join(_normalize(c) for c in contexts)
    # so khop mot doan du dai cua expected_context thay vi toan bo (context
    # bi chunk/reorder nen hiem khi khop nguyen van dai)
    probe = needle[: min(len(needle), 120)]
    return probe in haystack


def _cache_key(case_id: str, config_name: str, top_k: int) -> str:
    raw = f"{case_id}|{config_name}|{top_k}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_cached(key: str) -> dict | None:
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return None


def _save_cache(key: str, record: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / f"{key}.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def run_one(case: dict, cfg: RunConfig, *, top_k: int, use_cache: bool = True) -> dict:
    """Chay mot golden case qua mot config, tra ve raw record (chua co metrics)."""
    key = _cache_key(case["id"], cfg.name, top_k)
    if use_cache:
        cached = _load_cached(key)
        if cached is not None:
            return cached

    start = time.perf_counter()
    result, trace_dict = answer_with_trace(
        case["question"], top_k=top_k, use_reranking=cfg.use_reranking
    )
    latency_ms = (time.perf_counter() - start) * 1000

    contexts = [s["content"] for s in result["sources"]]
    context_ids = [s["id"] for s in result["sources"]]
    scores = [s["score"] for s in result["sources"]]

    record = {
        "id": case["id"],
        "config": cfg.name,
        "query_type": case.get("query_type", "in_domain"),
        "question": case["question"],
        "expected_answer": case["expected_answer"],
        "expected_context": case["expected_context"],
        "answer": result["answer"],
        "retrieval_source": result["retrieval_source"],
        "contexts": contexts,
        "context_ids": context_ids,
        "scores": scores,
        "context_hit": _context_hit(case["expected_context"], contexts),
        "latency_ms": latency_ms,
        "trace_error": trace_dict.get("error"),
    }
    if use_cache:
        _save_cache(key, record)
    return record


def run_config(dataset: list[dict], cfg: RunConfig, *, top_k: int, limit: int | None, use_cache: bool) -> list[dict]:
    cases = dataset[:limit] if limit else dataset
    records = []
    for i, case in enumerate(cases, 1):
        print(f"  [{cfg.name}] {i}/{len(cases)} {case['id']}: {case['question'][:60]}...")
        records.append(run_one(case, cfg, top_k=top_k, use_cache=use_cache))
    return records


def score_records(records: list[dict], *, skip_metrics: bool) -> None:
    """Cham 4 metric RAGAS tai cho, ghi truc tiep vao record['metrics']."""
    if skip_metrics:
        for record in records:
            record["metrics"] = {}
        return

    from .eval_ragas import build_judge, score_one

    judge = build_judge()
    for i, record in enumerate(records, 1):
        if record.get("query_type") == "out_of_domain":
            # Khong cham context_recall/precision cho ca out_of_domain: reference
            # la chuoi tu choi, khong phai noi dung that -- cham se cho diem sai lech.
            record["metrics"] = {
                "faithfulness": None,
                "answer_relevancy": None,
                "context_recall": None,
                "context_precision": None,
                "refusal_correct": record["retrieval_source"] == "none",
            }
            continue
        print(f"  scoring {i}/{len(records)}: {record['id']}")
        record["metrics"] = score_one(judge, record)


def _average(values: list[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    return sum(clean) / len(clean) if clean else None


def build_summary(records_a: list[dict], records_b: list[dict]) -> dict:
    metric_names = ["faithfulness", "answer_relevancy", "context_recall", "context_precision"]

    def config_summary(records: list[dict]) -> dict:
        in_domain = [r for r in records if r.get("query_type") != "out_of_domain"]
        out_domain = [r for r in records if r.get("query_type") == "out_of_domain"]
        overall = {
            name: _average([r["metrics"].get(name) for r in in_domain]) for name in metric_names
        }
        overall["average"] = _average([v for v in overall.values() if v is not None])
        context_hits = [r["context_hit"] for r in in_domain if r["context_hit"] is not None]
        refusal_correct = [r["metrics"].get("refusal_correct") for r in out_domain]
        return {
            "overall": overall,
            "context_hit_rate": _average([1.0 if h else 0.0 for h in context_hits]),
            "refusal_accuracy": _average(
                [1.0 if r else 0.0 for r in refusal_correct if r is not None]
            ),
            "avg_latency_ms": _average([r["latency_ms"] for r in records]),
            "n_cases": len(records),
        }

    worst = sorted(
        [r for r in records_b if r.get("query_type") != "out_of_domain"],
        key=lambda r: _average([v for v in r["metrics"].values() if isinstance(v, (int, float))]) or 0.0,
    )[:3]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config_a": config_summary(records_a),
        "config_b": config_summary(records_b),
        "worst_performers_config_b": [
            {
                "id": r["id"],
                "question": r["question"],
                "metrics": r["metrics"],
            }
            for r in worst
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["A", "B", "both"], default="both")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH)
    parser.add_argument("--out", type=Path, default=RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    use_cache = not args.no_cache

    records_a, records_b = [], []
    if args.config in ("A", "both"):
        print(f"Running Config A (dense-only) on {len(dataset)} cases...")
        records_a = run_config(dataset, CONFIG_A, top_k=args.top_k, limit=args.limit, use_cache=use_cache)
        score_records(records_a, skip_metrics=args.skip_metrics)
        (args.out / "raw_A_dense.json").write_text(
            json.dumps(records_a, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if args.config in ("B", "both"):
        print(f"Running Config B (hybrid+RRF) on {len(dataset)} cases...")
        records_b = run_config(dataset, CONFIG_B, top_k=args.top_k, limit=args.limit, use_cache=use_cache)
        score_records(records_b, skip_metrics=args.skip_metrics)
        (args.out / "raw_B_hybrid.json").write_text(
            json.dumps(records_b, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if records_a and records_b:
        summary = build_summary(records_a, records_b)
        (args.out / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Summary written to {args.out / 'summary.json'}")

    run_meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "top_k": args.top_k,
        "dataset": str(args.dataset),
        "n_cases": len(dataset) if not args.limit else min(args.limit, len(dataset)),
        "skip_metrics": args.skip_metrics,
    }
    (args.out / "run_meta.json").write_text(
        json.dumps(run_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Done.")


if __name__ == "__main__":
    main()
