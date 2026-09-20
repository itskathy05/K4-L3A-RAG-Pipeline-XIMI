"""
Task 9 -- Retrieval pipeline hoan chinh.

Luong xu ly:
    1. Chay semantic_search va lexical_search.
    2. Fuse hai danh sach bang RRF dung mot lan.
    3. Lay best cosine score goc tu dense results.
    4. Neu score duoi threshold, thu PageIndex fallback.
    5. Neu fallback loi, tra hybrid results thay vi crash.

Khong so sanh threshold voi RRF score vi hai thang do khac nhau.

Cac ham stage (semantic_search, lexical_search, rerank_rrf, pageindex_search)
duoc goi nhu module-global cua chinh module nay de 3 test monkeypatch trong
tests/test_contracts.py hoat dung dung: test patch truc tiep
`pipeline.semantic_search` etc, nen ham retrieve() phai goi qua ten global
(khong duoc bind lai thanh alias private o muc module).
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from . import trace
from .task5_semantic_search import semantic_search
from .task6_lexical_search import lexical_search
from .task7_reranking import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search

load_dotenv()


def _default_threshold() -> float:
    raw = os.getenv("SCORE_THRESHOLD", "").strip()
    return float(raw) if raw else 0.3


SCORE_THRESHOLD = _default_threshold()
DEFAULT_TOP_K = 5


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
) -> list[dict]:
    """Tra ve hybrid hoac pageindex SearchResult."""
    with trace.stage("retrieve"):
        dense = semantic_search(query, top_k=top_k * 2)
        sparse = lexical_search(query, top_k=top_k * 2)
        hybrid = (
            rerank_rrf([dense, sparse], top_k=top_k) if use_reranking else dense[:top_k]
        )

        best_dense_score = dense[0]["score"] if dense else 0.0
        triggered = best_dense_score < score_threshold
        trace.record(
            "fallback.decision",
            best_dense_score=best_dense_score,
            threshold=score_threshold,
            triggered=triggered,
        )

        if triggered:
            try:
                fallback = pageindex_search(query, top_k=top_k)
                if fallback:
                    trace.record("fallback.verdict", verdict="pageindex")
                    return fallback
            except Exception as error:
                trace.record("fallback.error", error=repr(error))

        trace.record("fallback.verdict", verdict="hybrid" if not triggered else "hybrid_after_empty_fallback")
        return hybrid[:top_k]


if __name__ == "__main__":
    for result in retrieve("test query", top_k=3):
        print(result)
