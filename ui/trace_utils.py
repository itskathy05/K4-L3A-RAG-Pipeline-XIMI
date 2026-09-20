"""Tien ich doc trace dict (JSON-plain, tu Trace.to_dict()) -- khong phu
thuoc Streamlit, co the test doc lap."""

from __future__ import annotations

from typing import Any


def find_event(events: list[dict], stage: str) -> dict | None:
    for event in reversed(events):
        if event["stage"] == stage:
            return event
    return None


def find_events(events: list[dict], stage: str) -> list[dict]:
    return [e for e in events if e["stage"] == stage]


def latency_by_stage(events: list[dict]) -> dict[str, float]:
    result = {}
    for event in events:
        if event["stage"].endswith(".latency"):
            name = event["stage"].removesuffix(".latency")
            result[name] = event.get("latency_ms", 0.0)
    return result


def summarize_trace(trace: dict) -> dict[str, Any]:
    """Tom tat cac gia tri Inspector can, tra ve dict phang de UI khong phai
    tu tim event moi lan render."""
    events = trace.get("events", [])
    query_processing = find_event(events, "query_processing.result")
    query_processing_merge = find_event(events, "query_processing.retrieval_merge")
    dense = find_event(events, "dense")
    bm25 = find_event(events, "bm25")
    fusion = find_event(events, "fusion")
    fallback_decision = find_event(events, "fallback.decision")
    fallback_verdict = find_event(events, "fallback.verdict")
    fallback_error = find_event(events, "fallback.error")
    reorder = find_event(events, "reorder")
    context = find_event(events, "context")
    generation_result = find_event(events, "generation.result")
    citation_dropped = find_events(events, "citation.dropped")
    pageindex = find_event(events, "pageindex")
    pageindex_error = find_event(events, "pageindex.error")

    return {
        "query_processing": query_processing,
        "query_processing_merge": query_processing_merge,
        "dense": dense,
        "bm25": bm25,
        "fusion": fusion,
        "fallback_decision": fallback_decision,
        "fallback_verdict": fallback_verdict,
        "fallback_error": fallback_error,
        "reorder": reorder,
        "context": context,
        "generation_result": generation_result,
        "citation_dropped": citation_dropped,
        "pageindex": pageindex,
        "pageindex_error": pageindex_error,
        "latency": latency_by_stage(events),
        "total_latency_ms": trace.get("total_latency_ms", 0.0),
        "error": trace.get("error"),
    }
