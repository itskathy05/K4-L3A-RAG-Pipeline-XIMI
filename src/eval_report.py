"""
Sinh group_project/evaluation/RESULT.md tu summary.json + mot file phan
tich viet tay (analysis.json). Sinh bang script thay vi dien tay de tranh
sot chu "TODO" khi phai chay lai o Pha 2.

Cach dung:
    python -m src.eval_report --check
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "group_project" / "evaluation" / "results"
ANALYSIS_PATH = ROOT / "group_project" / "evaluation" / "analysis.json"
OUTPUT_PATH = ROOT / "group_project" / "evaluation" / "RESULT.md"

REQUIRED_HEADINGS = ["overall scores", "a/b comparison", "worst performers", "recommendations"]


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}"
    return str(value)


def _load_json(path: Path, default):
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def render(summary: dict, run_meta: dict, analysis: dict) -> str:
    a = summary["config_a"]["overall"]
    b = summary["config_b"]["overall"]

    def delta(key: str) -> str:
        va, vb = a.get(key), b.get(key)
        if va is None or vb is None:
            return "n/a"
        return f"{vb - va:+.3f}"

    worst_rows = []
    for i, item in enumerate(summary.get("worst_performers_config_b", []), 1):
        case_analysis = analysis.get("worst_performers", {}).get(item["id"], {})
        m = item["metrics"]
        worst_rows.append(
            "| {n} | {q} | B | {faith} | {rel} | {rec} | {prec} | {stage} | {root} |".format(
                n=i,
                q=item["question"][:70].replace("|", "/"),
                faith=_fmt(m.get("faithfulness")),
                rel=_fmt(m.get("answer_relevancy")),
                rec=_fmt(m.get("context_recall")),
                prec=_fmt(m.get("context_precision")),
                stage=case_analysis.get("failure_stage", "n/a"),
                root=case_analysis.get("root_cause", "n/a"),
            )
        )
    while len(worst_rows) < 3:
        worst_rows.append(f"| {len(worst_rows) + 1} | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")

    reco_rows = []
    for i, reco in enumerate(analysis.get("recommendations", []), 1):
        reco_rows.append(
            "| {n} | {action} | {evidence} | {impact} | {verify} |".format(
                n=i,
                action=reco.get("action", "n/a"),
                evidence=reco.get("evidence", "n/a"),
                impact=reco.get("expected_impact", "n/a"),
                verify=reco.get("how_to_verify", "n/a"),
            )
        )
    while len(reco_rows) < 3:
        reco_rows.append(f"| {len(reco_rows) + 1} | n/a | n/a | n/a | n/a |")

    bonus_rows = []
    for exp in analysis.get("bonus_experiments", []):
        bonus_rows.append(
            "| {name} | {baseline} | {delta} | {cost} | {conclusion} |".format(
                name=exp.get("experiment", "n/a"),
                baseline=exp.get("baseline", "n/a"),
                delta=exp.get("metric_delta", "n/a"),
                cost=exp.get("latency_cost_delta", "n/a"),
                conclusion=exp.get("conclusion", "n/a"),
            )
        )
    if not bonus_rows:
        bonus_rows.append("| n/a | n/a | n/a | n/a | n/a |")

    return f"""# RAG evaluation results

## Run information

| Field                              | Value |
| ----------------------------------- | ----- |
| Evaluation date                    | {summary.get("generated_at", "n/a")} |
| Framework and version              | ragas 0.4.3 |
| Evaluator model                    | gpt-4o-mini (RAGAS judge) |
| Generator model                    | {analysis.get("generator_model", "gpt-4o-mini")} |
| Embedding model                    | {analysis.get("embedding_model", "text-embedding-3-small")} |
| Corpus version/commit              | {analysis.get("corpus_commit", "n/a")} |
| Golden dataset size                | {run_meta.get("n_cases", "n/a")} |
| `top_k`                            | {run_meta.get("top_k", "n/a")} |
| Fallback threshold and calibration | {analysis.get("threshold_note", "n/a")} |

## Configurations

- **Config A -- dense-only:** `retrieve(query, top_k, use_reranking=False)` -- semantic_search only, no BM25/RRF fusion.
- **Config B -- hybrid + RRF:** `retrieve(query, top_k, use_reranking=True)` -- dense + BM25 fused with Reciprocal Rank Fusion (k=60).

Hai config dung cung golden dataset, generator, evaluator, prompt va `top_k`; chi khac retrieval strategy (`use_reranking`).

## Overall scores

| Metric            | Config A | Config B | Delta B-A |
| ----------------- | -------: | -------: | --------: |
| Faithfulness      | {_fmt(a.get("faithfulness"))} | {_fmt(b.get("faithfulness"))} | {delta("faithfulness")} |
| Answer relevance  | {_fmt(a.get("answer_relevancy"))} | {_fmt(b.get("answer_relevancy"))} | {delta("answer_relevancy")} |
| Context recall    | {_fmt(a.get("context_recall"))} | {_fmt(b.get("context_recall"))} | {delta("context_recall")} |
| Context precision | {_fmt(a.get("context_precision"))} | {_fmt(b.get("context_precision"))} | {delta("context_precision")} |
| **Average**       | {_fmt(a.get("average"))} | {_fmt(b.get("average"))} | {delta("average")} |

Ca out_of_domain duoc loai khoi trung binh tren (danh gia rieng bang refusal accuracy) de khong am tham thuong/phat viec tu choi dung. Refusal accuracy: Config A = {_fmt(summary["config_a"].get("refusal_accuracy"))}, Config B = {_fmt(summary["config_b"].get("refusal_accuracy"))}. Context-hit rate (khong dung LLM, substring check doc lap voi RAGAS): Config A = {_fmt(summary["config_a"].get("context_hit_rate"))}, Config B = {_fmt(summary["config_b"].get("context_hit_rate"))}.

## A/B comparison

- Cau hinh tot hon: {analysis.get("better_config", "n/a")}
- Evidence: {analysis.get("ab_evidence", "n/a")}
- Trade-off ve latency/cost: Config A trung binh {_fmt(summary["config_a"].get("avg_latency_ms"), 0)} ms/cau; Config B trung binh {_fmt(summary["config_b"].get("avg_latency_ms"), 0)} ms/cau. {analysis.get("ab_tradeoff_note", "")}

## Worst performers

|   # | Question | Config | Faithfulness | Relevance | Recall | Precision | Failure stage             | Root cause |
| --: | -------- | ------ | -----------: | --------: | -----: | --------: | -------------------------- | ---------- |
{chr(10).join(worst_rows)}

## Recommendations

| Priority | Action | Evidence from failure analysis | Expected impact | How to verify |
| -------: | ------ | ------------------------------- | ---------------- | ------------- |
{chr(10).join(reco_rows)}

## Bonus experiments

| Experiment | Baseline | Metric delta | Latency/cost delta | Conclusion |
| ---------- | -------- | ------------: | -------------------: | ---------- |
{chr(10).join(bonus_rows)}
"""


def check(text: str) -> list[str]:
    problems = []
    if "TODO" in text:
        problems.append("Report still contains the literal string 'TODO'")
    lowered = text.lower()
    for heading in REQUIRED_HEADINGS:
        if heading not in lowered:
            problems.append(f"Missing required heading: {heading!r}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=RESULTS_DIR)
    parser.add_argument("--analysis", type=Path, default=ANALYSIS_PATH)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--check", action="store_true", help="Validate before writing; exit 1 on failure")
    args = parser.parse_args()

    summary = _load_json(args.results / "summary.json", {})
    run_meta = _load_json(args.results / "run_meta.json", {})
    analysis = _load_json(args.analysis, {})

    if not summary:
        raise SystemExit(f"No summary found at {args.results / 'summary.json'}. Run eval_runner first.")

    text = render(summary, run_meta, analysis)
    problems = check(text)
    if problems:
        for p in problems:
            print(f"CHECK FAILED: {p}")
        if args.check:
            raise SystemExit(1)

    args.out.write_text(text, encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
