"""Component tai su dung cho tab Chat: citation rendering, source card,
refusal banner, dai metric trang thai."""

from __future__ import annotations

import re

import streamlit as st

_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


def status_metrics_row(result: dict, trace: dict) -> None:
    cols = st.columns(3)
    best_dense = None
    threshold = None
    fallback_decision = None
    for event in trace.get("events", []):
        if event["stage"] == "fallback.decision":
            fallback_decision = event
    if fallback_decision:
        best_dense = fallback_decision.get("best_dense_score")
        threshold = fallback_decision.get("threshold")

    with cols[0]:
        st.metric("Retrieval source", result.get("retrieval_source", "none"))
    with cols[1]:
        top_score = result["sources"][0]["score"] if result.get("sources") else None
        delta = None
        if best_dense is not None and threshold is not None:
            delta = f"{best_dense - threshold:+.3f} vs threshold"
        st.metric("Top score", f"{top_score:.3f}" if top_score is not None else "n/a", delta=delta)
    with cols[2]:
        st.metric("Latency", f"{trace.get('total_latency_ms', 0):.0f} ms")


def render_answer_with_citations(answer: str, sources: list[dict]) -> None:
    """Viet lai [Sn] thanh anchor markdown toi source card ben duoi."""

    def _replace(match: re.Match) -> str:
        n = int(match.group(1))
        if 1 <= n <= len(sources):
            return f"**[[S{n}]](#src-{n})**"
        return f":red[[S{n}]]"

    rendered = _CITATION_PATTERN.sub(_replace, answer)
    st.markdown(rendered)

    unresolved = [
        m.group(0) for m in _CITATION_PATTERN.finditer(answer) if not (1 <= int(m.group(1)) <= len(sources))
    ]
    if unresolved:
        st.caption(f":red[citation not matched to a source: {', '.join(unresolved)}]")


def source_card(index: int, source: dict, query: str = "", key_prefix: str = "") -> None:
    metadata = source["metadata"]
    title = metadata.get("title", "unknown")
    section_path = metadata.get("section_path")
    label = f"[S{index}] {title}"
    if section_path:
        label += f" -- {section_path}"
    label += f" | {source['retrieval_method']} | score {source['score']:.3f}"

    with st.container(key=f"src-{key_prefix}{index}"):
        with st.expander(label):
            st.caption(f"source: {metadata.get('source')} | doc_type: {metadata.get('doc_type')}")
            score_display = min(max(source["score"], 0.0), 1.0)
            st.progress(score_display)
            content = source["content"]
            for term in [t for t in query.split() if len(t) > 3]:
                content = re.sub(f"(?i)({re.escape(term)})", r"**\1**", content)
            st.markdown(content)
            if metadata.get("url"):
                st.link_button("Open source", metadata["url"])


def query_processing_card(trace: dict, *, expanded: bool = False) -> None:
    """Hien thi ket qua decompose/expand/reformulate cua Task query_processing
    (opt-in). No-op neu tab nay khong bat query expansion cho luot hoi do."""
    qp = None
    for event in trace.get("events", []):
        if event["stage"] == "query_processing.result":
            qp = event
    if qp is None:
        return

    merge = None
    for event in trace.get("events", []):
        if event["stage"] == "query_processing.retrieval_merge":
            merge = event

    changed = qp["reformulated"].strip() != qp["original"].strip()
    n_sub = len(qp["sub_queries"])
    if n_sub > 1:
        badge = f"🧩 decomposed thành {n_sub} câu hỏi con"
    elif changed:
        badge = "✏️ đã reformulate"
    else:
        badge = "✓ giữ nguyên câu hỏi"

    with st.expander(f"🧠 Query processing -- {badge}", expanded=expanded):
        st.caption("Câu hỏi gốc")
        st.code(qp["original"], language=None)

        if n_sub > 1:
            st.caption(f"Decomposed thành {n_sub} sub-query (mỗi câu retrieve riêng, gộp kết quả)")
            for i, sq in enumerate(qp["sub_queries"], 1):
                st.markdown(f"{i}. {sq}")
        else:
            st.caption("Reformulated (dùng để retrieve, không đổi câu hỏi gửi cho LLM sinh câu trả lời)")
            st.code(qp["reformulated"], language=None)

        if qp["expanded_terms"]:
            st.caption("Expanded terms (thêm vào query khi retrieve)")
            st.markdown(" ".join(f"`{t}`" for t in qp["expanded_terms"]))

        if merge and len(merge.get("queries_used", [])) > 1:
            st.caption(f"Đã chạy {len(merge['queries_used'])} lượt retrieve và gộp {len(merge['merged_ids'])} chunk theo score cao nhất.")


def refusal_banner(result: dict, trace: dict) -> None:
    reason_parts = []
    for event in trace.get("events", []):
        if event["stage"] == "fallback.decision" and event.get("triggered"):
            reason_parts.append(
                f"Best dense cosine {event['best_dense_score']:.3f} < threshold {event['threshold']:.3f}."
            )
        if event["stage"] == "pageindex.error":
            reason_parts.append("PageIndex fallback failed.")
        if event["stage"] == "pageindex" and not event.get("results"):
            reason_parts.append("PageIndex fallback returned 0 results.")
        if event["stage"] == "generation.result" and event.get("reason") == "no_valid_citations":
            reason_parts.append("The model could not produce a verifiable citation.")

    st.warning(result["answer"])
    if reason_parts:
        st.caption(" ".join(reason_parts))
