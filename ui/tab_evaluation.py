"""Tab Evaluation -- doc file cache (group_project/evaluation/results/), KHONG
tinh lai truc tiep trong UI. Tinh lai nghia la 18 cau x 2 config x 4 metric
LLM-judge ~ hang tram API call, vai phut, giua buoi demo -- rui ro rate
limit ngay truoc mat nguoi xem. Nut "Run" (neu can) nen la subprocess rieng,
khong dat trong luong render chinh.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "group_project" / "evaluation" / "results"
DATASET_PATH = ROOT / "group_project" / "evaluation" / "golden_dataset.json"


@st.cache_data
def _load_json(path_str: str, _mtime: float):
    return json.loads(Path(path_str).read_text(encoding="utf-8"))


def _load_if_exists(path: Path):
    if not path.exists():
        return None
    return _load_json(str(path), path.stat().st_mtime)


def render_evaluation() -> None:
    summary = _load_if_exists(RESULTS_DIR / "summary.json")
    run_meta = _load_if_exists(RESULTS_DIR / "run_meta.json")
    raw_a = _load_if_exists(RESULTS_DIR / "raw_A_dense.json") or []
    raw_b = _load_if_exists(RESULTS_DIR / "raw_B_hybrid.json") or []

    if not summary:
        st.info(
            "Chua co ket qua evaluation. Chay:\n\n"
            "`python -m src.eval_runner --config both`\n\n"
            "roi `python -m src.eval_report --check` de sinh RESULT.md."
        )
        _render_golden_dataset_browser()
        return

    st.caption(f"Last run: {summary.get('generated_at', 'n/a')}" + (f" | top_k={run_meta.get('top_k')}" if run_meta else ""))

    _render_overall_scores(summary)
    st.divider()
    _render_per_question(raw_a, raw_b)
    st.divider()
    _render_worst_performers(summary)
    st.divider()
    _render_golden_dataset_browser()


def _render_overall_scores(summary: dict) -> None:
    st.subheader("Overall scores: dense-only (A) vs hybrid+RRF (B)")
    a, b = summary["config_a"]["overall"], summary["config_b"]["overall"]
    metric_names = ["faithfulness", "answer_relevancy", "context_recall", "context_precision", "average"]

    cols = st.columns(len(metric_names))
    for col, name in zip(cols, metric_names):
        va, vb = a.get(name), b.get(name)
        delta = None
        if va is not None and vb is not None:
            delta = f"{vb - va:+.3f}"
        col.metric(name, f"{vb:.3f}" if vb is not None else "n/a", delta=delta)

    rows = []
    for name in metric_names:
        rows.append({"metric": name, "config": "A (dense)", "value": a.get(name)})
        rows.append({"metric": name, "config": "B (hybrid)", "value": b.get(name)})
    df = pd.DataFrame(rows).dropna()
    if not df.empty:
        st.bar_chart(df, x="metric", y="value", color="config")

    st.caption(
        f"Context-hit rate (substring check, khong dung LLM): "
        f"A={summary['config_a'].get('context_hit_rate')}, B={summary['config_b'].get('context_hit_rate')} | "
        f"Refusal accuracy: A={summary['config_a'].get('refusal_accuracy')}, B={summary['config_b'].get('refusal_accuracy')}"
    )


def _render_per_question(raw_a: list[dict], raw_b: list[dict]) -> None:
    st.subheader("Per-question detail")
    all_rows = raw_a + raw_b
    if not all_rows:
        st.caption("no data")
        return

    query_types = sorted({r.get("query_type", "in_domain") for r in all_rows})
    selected_types = st.multiselect("Loc theo query_type", options=query_types, default=query_types)

    filtered = [r for r in all_rows if r.get("query_type", "in_domain") in selected_types]
    table_rows = []
    for r in filtered:
        m = r.get("metrics", {})
        table_rows.append(
            {
                "id": r["id"],
                "config": r["config"],
                "query_type": r.get("query_type"),
                "question": r["question"][:60],
                "faithfulness": m.get("faithfulness"),
                "answer_relevancy": m.get("answer_relevancy"),
                "context_recall": m.get("context_recall"),
                "context_precision": m.get("context_precision"),
                "context_hit": r.get("context_hit"),
            }
        )
    st.dataframe(pd.DataFrame(table_rows), hide_index=True)


def _render_worst_performers(summary: dict) -> None:
    st.subheader("Worst performers (Config B)")
    worst = summary.get("worst_performers_config_b", [])
    if not worst:
        st.caption("no data")
        return
    for item in worst:
        m = item["metrics"]
        st.markdown(f"**{item['id']}** -- {item['question']}")
        st.caption(
            f"faithfulness={m.get('faithfulness')} | relevancy={m.get('answer_relevancy')} | "
            f"recall={m.get('context_recall')} | precision={m.get('context_precision')}"
        )


def _render_golden_dataset_browser() -> None:
    st.subheader("Golden dataset browser")
    dataset = _load_if_exists(DATASET_PATH)
    if not dataset:
        st.caption("no golden dataset found")
        return
    query_types = sorted({d.get("query_type", "in_domain") for d in dataset})
    chip = st.selectbox("query_type", options=["all"] + query_types)
    rows = dataset if chip == "all" else [d for d in dataset if d.get("query_type") == chip]
    st.dataframe(
        pd.DataFrame(
            [{"id": d["id"], "type": d.get("query_type"), "lang": d.get("lang"), "question": d["question"]} for d in rows]
        ),
        hide_index=True,
    )
