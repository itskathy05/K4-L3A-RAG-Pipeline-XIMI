"""Tab Inspector -- trung tam cua demo. Render HOAN TOAN tu
st.session_state, khong bao gio tu goi lai pipeline. Cho thay tung buoc
cua retrieval: dense, BM25, RRF rank movement, reorder, fallback decision
(kem what-if slider), latency waterfall, raw trace.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.state import assistant_turns
from ui.trace_utils import summarize_trace


def render_inspector() -> None:
    turns = assistant_turns()
    if not turns:
        st.info("Hoi mot cau trong tab Chat truoc, roi quay lai day de xem chi tiet retrieval.")
        return

    labels = [f"{i + 1}. {t['content'][:60]}" for i, t in enumerate(turns)]
    default_index = len(turns) - 1
    choice = st.selectbox("Chon lai cau hoi da hoi", options=range(len(turns)), format_func=lambda i: labels[i], index=default_index)
    message = turns[choice]
    trace = message["trace"]
    summary = summarize_trace(trace)

    _render_verdict_strip(summary)
    st.divider()
    _render_query_processing(summary)
    st.divider()
    _render_dense_vs_bm25(summary)
    st.divider()
    _render_rrf_movement(summary)
    st.divider()
    _render_reorder(summary)
    st.divider()
    _render_fallback(summary)
    st.divider()
    _render_latency_waterfall(summary)
    st.divider()
    _render_raw_trace(trace)


def _render_verdict_strip(summary: dict) -> None:
    decision = summary["fallback_decision"] or {}
    fusion = summary["fusion"] or {}

    cols = st.columns(5)
    cols[0].metric(
        "Best dense cosine",
        f"{decision.get('best_dense_score', 0):.3f}",
        delta=f"{decision.get('best_dense_score', 0) - decision.get('threshold', 0):+.3f} vs threshold"
        if decision
        else None,
    )
    cols[1].metric("RRF k", fusion.get("k", "n/a"))
    cols[2].metric("top_k", fusion.get("top_k", "n/a"))
    reorder = summary.get("reorder") or {}
    cols[3].metric("Chunks in context", len(reorder.get("order_after", [])))
    cols[4].metric("Total latency", f"{summary['total_latency_ms']:.0f} ms")


def _render_query_processing(summary: dict) -> None:
    st.subheader("0. Query processing (decompose / expand / reformulate)")
    qp = summary.get("query_processing")
    if not qp:
        st.caption("Query processing tat cho cau hoi nay (toggle o sidebar tat, hoac fallback ve identity).")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.caption("Original")
        st.code(qp["original"], language=None)
    with col2:
        st.caption("Reformulated (dung de retrieve)")
        st.code(qp["reformulated"], language=None)

    if qp["expanded_terms"]:
        st.caption("Expanded terms")
        st.markdown(" ".join(f"`{t}`" for t in qp["expanded_terms"]))

    merge = summary.get("query_processing_merge")
    if qp["sub_queries"]:
        st.caption(f"Decomposed thanh {len(qp['sub_queries'])} sub-query (moi cau retrieve rieng qua Task 9, gop theo id giu score cao nhat):")
        for i, sq in enumerate(qp["sub_queries"], 1):
            st.markdown(f"{i}. {sq}")
        if merge:
            st.caption(f"-> Gop {len(merge['merged_ids'])} chunk sau khi dedupe: {', '.join(merge['merged_ids'])}")
        st.info(
            "Luu y: cac panel Dense/BM25/RRF ben duoi chi phan anh lan retrieve() "
            "CUOI CUNG (sub-query cuoi), vi moi sub-query chay retrieve rieng."
        )
    else:
        st.caption("Cau hoi duoc coi la atomic -- khong decompose, chi reformulate + expand roi retrieve 1 lan.")


def _render_dense_vs_bm25(summary: dict) -> None:
    st.subheader("1. Dense vs BM25")
    col1, col2 = st.columns(2)

    dense = summary["dense"]
    bm25 = summary["bm25"]

    with col1:
        st.caption("Dense (cosine similarity, 0-1)")
        if dense and dense.get("results"):
            rows = [
                {"rank": i + 1, "chunk": r["id"], "title": r["metadata"].get("title", "")[:40], "score": r["score"]}
                for i, r in enumerate(dense["results"])
            ]
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                column_config={"score": st.column_config.ProgressColumn("score", min_value=0, max_value=1)},
            )
        else:
            st.caption("no dense results")

    with col2:
        st.caption("BM25 (unbounded score -- KHONG so sanh truc tiep voi thanh cosine ben trai)")
        if bm25 and bm25.get("results"):
            max_score = max(r["score"] for r in bm25["results"]) or 1.0
            rows = [
                {"rank": i + 1, "chunk": r["id"], "title": r["metadata"].get("title", "")[:40], "score": r["score"]}
                for i, r in enumerate(bm25["results"])
            ]
            st.dataframe(
                pd.DataFrame(rows),
                hide_index=True,
                column_config={
                    "score": st.column_config.ProgressColumn("score", min_value=0, max_value=max_score)
                },
            )
        else:
            st.caption("no BM25 results (khong tu khoa nao trung)")


def _render_rrf_movement(summary: dict) -> None:
    st.subheader("2. RRF rank movement")
    fusion = summary["fusion"]
    if not fusion or not fusion.get("candidates"):
        st.caption("Config dense-only (use_reranking=False) -- khong co buoc fusion.")
        return

    candidates = fusion["candidates"]
    rows = []
    for c in candidates:
        rows.append(
            {
                "chunk": c["id"],
                "dense_rank": c["dense_rank"] if c["dense_rank"] is not None else None,
                "bm25_rank": c["bm25_rank"] if c["bm25_rank"] is not None else None,
                "rrf_score": c["rrf_score"],
                "final_rank": c["final_rank"],
                "delta_vs_dense": (c["dense_rank"] - c["final_rank"]) if c["dense_rank"] is not None else None,
                "in_context": c["selected"],
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        hide_index=True,
        column_config={
            "rrf_score": st.column_config.ProgressColumn(
                "rrf_score", min_value=0, max_value=max(df["rrf_score"].max(), 1e-9)
            ),
            "delta_vs_dense": st.column_config.NumberColumn("delta_vs_dense", format="%+d"),
            "in_context": st.column_config.CheckboxColumn("in_context", disabled=True),
        },
    )

    try:
        import altair as alt

        chart_rows = []
        for c in candidates[:12]:
            if c["dense_rank"] is not None:
                chart_rows.append({"chunk": c["id"], "stage": "Dense", "rank": c["dense_rank"]})
            if c["bm25_rank"] is not None:
                chart_rows.append({"chunk": c["id"], "stage": "BM25", "rank": c["bm25_rank"]})
            chart_rows.append({"chunk": c["id"], "stage": "Fused", "rank": c["final_rank"]})

        chart_df = pd.DataFrame(chart_rows)
        chart = (
            alt.Chart(chart_df)
            .mark_line(point=True, strokeWidth=3)
            .encode(
                x=alt.X("stage:N", sort=["Dense", "BM25", "Fused"], title=None),
                y=alt.Y("rank:Q", scale=alt.Scale(reverse=True), title="rank (1 = best)"),
                color=alt.Color("chunk:N", legend=None),
                tooltip=["chunk", "stage", "rank"],
            )
            .properties(height=280)
        )
        st.altair_chart(chart, use_container_width=True)
        st.caption("Mot duong di len tu cot BM25 va dap xuong Fused #1 la RRF dang 'cuu' mot chunk ma dense bo lo.")
    except Exception as error:
        st.caption(f"(bump chart unavailable: {error})")


def _render_reorder(summary: dict) -> None:
    st.subheader("3. Reorder (lost-in-the-middle)")
    reorder = summary["reorder"]
    if not reorder:
        st.caption("Khong co du lieu reorder (co the do safe refusal som).")
        return

    col1, col2, col3 = st.columns([2, 1, 2])
    with col1:
        st.caption("Before (score order)")
        for i, cid in enumerate(reorder["order_before"], 1):
            st.text(f"{i}. {cid}")
    with col2:
        st.markdown("&nbsp;\n\n➡️" * len(reorder["order_before"]))
    with col3:
        st.caption("After (dau/cuoi uu tien)")
        after = reorder["order_after"]
        for i, cid in enumerate(after, 1):
            style = "green-background" if i in (1, len(after)) else None
            text = f"{i}. {cid}"
            st.markdown(f":{style}[{text}]" if style else text)

    context = summary["context"]
    if context:
        with st.expander(f"Final context ({context.get('char_len', 0)} chars)"):
            st.code(context.get("text", ""))


def _render_fallback(summary: dict) -> None:
    st.subheader("4. Fallback decision")
    decision = summary["fallback_decision"]
    if not decision:
        st.caption("Khong co du lieu fallback decision.")
        return

    best = decision["best_dense_score"]
    threshold = decision["threshold"]
    st.metric("Best dense cosine", f"{best:.3f}", delta=f"{best - threshold:+.3f} vs threshold {threshold:.3f}")
    st.progress(min(max(best, 0.0), 1.0))

    # Luu y: day la quyet dinh o TANG RETRIEVAL (task9). retrieve() khong bao
    # gio tra ve rong -- neu PageIndex fallback cung khong tim thay gi thi no
    # ha ve dung hybrid results (degraded, khong phai la mot refusal). Safe
    # refusal THAT SU (khong sinh cau tra loi) la mot quyet dinh RIENG o tang
    # generation (task10, khi khong co citation hop le), hien o tab Chat.
    verdict_event = summary["fallback_verdict"]
    verdict = verdict_event.get("verdict") if verdict_event else ("hybrid" if not decision["triggered"] else "pageindex")

    if not decision["triggered"]:
        st.success(f"HYBRID (confident) -- best dense {best:.3f} >= threshold {threshold:.3f}")
    elif verdict == "pageindex":
        st.warning(f"FALLBACK -> PageIndex -- best dense {best:.3f} < threshold {threshold:.3f}, PageIndex tra ve ket qua")
    else:
        st.error(
            f"FALLBACK TRIGGERED nhung PageIndex rong -- best dense {best:.3f} < threshold {threshold:.3f}. "
            "Ha ve dung hybrid results (degraded), CHUA phai safe refusal -- xem tab Chat de biet cau tra loi cuoi cung co bi tu choi o tang generation hay khong."
        )

    if summary["fallback_error"]:
        st.caption(f"Fallback provider error (bat duoc, khong crash): {summary['fallback_error'].get('error')}")

    what_if = st.slider("What-if: neu threshold la...", 0.0, 1.0, float(threshold), 0.01, key="whatif-threshold")
    what_if_triggered = best < what_if
    st.caption(
        f"O threshold {what_if:.2f}, cau hoi nay se {'FALLBACK/refusal' if what_if_triggered else 'giu hybrid'}"
        " (tinh lai tu trace da cache, khong goi API)."
    )


def _render_latency_waterfall(summary: dict) -> None:
    st.subheader("5. Latency waterfall")
    latency = summary["latency"]
    if not latency:
        st.caption("Khong co du lieu latency.")
        return

    cols = st.columns(len(latency))
    for col, (name, ms) in zip(cols, latency.items()):
        col.metric(name, f"{ms:.0f} ms")

    try:
        import altair as alt

        rows = []
        cumulative = 0.0
        for name, ms in latency.items():
            rows.append({"stage": name, "start": cumulative, "end": cumulative + ms})
            cumulative += ms
        df = pd.DataFrame(rows)
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("start:Q", title="ms"),
                x2="end:Q",
                y=alt.Y("stage:N", sort=None),
                tooltip=["stage", "start", "end"],
            )
            .properties(height=140)
        )
        st.altair_chart(chart, use_container_width=True)
    except Exception:
        st.bar_chart(pd.Series(latency))


def _render_raw_trace(trace: dict) -> None:
    st.subheader("6. Raw trace")
    with st.expander("Xem trace JSON day du"):
        st.json(trace)
    import json

    st.download_button(
        "Download trace JSON",
        data=json.dumps(trace, ensure_ascii=False, indent=2),
        file_name=f"trace_{trace.get('trace_id', 'unknown')}.json",
        mime="application/json",
    )
