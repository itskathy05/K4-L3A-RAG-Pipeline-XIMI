"""
Query processing nang cao -- KHONG thuoc rubric task1-10, la lop tien xu ly
tuy chon truoc khi vao Task 9 (retrieve). Gom 3 ky thuat trong MOT LLM call
de gioi han them latency/cost:

    - Reformulation: viet lai cau hoi cho ro rang, standalone, sua chinh ta.
    - Expansion: them tu dong nghia/lien quan de mo rong pham vi tim kiem.
    - Decomposition: neu cau hoi la CAU KEP (nhieu y), tach thanh sub-queries
      doc lap, moi sub-query duoc retrieve rieng qua Task 9 khong doi (khong
      them tham so vao retrieve(), tranh pha pinned contract), roi gop ket
      qua theo id (giu score cao nhat).

Chi anh huong QUERY dung de retrieve. Cau hoi hien thi cho nguoi dung va dua
vao prompt generation (task10) van la cau hoi goc, de khong lam lech y dinh
nguoi dung trong cau tra loi.

Khong bao gio raise: loi LLM hoac JSON parse deu fallback ve cau hoi goc.
"""

from __future__ import annotations

import json
import re

from . import trace
from .llm import LLMError, call_text
from .task9_retrieval_pipeline import SCORE_THRESHOLD as _DEFAULT_SCORE_THRESHOLD
from .task9_retrieval_pipeline import retrieve

_SYSTEM_PROMPT = """Ban la bo tien xu ly truy van cho he thong RAG ve IELTS Writing \
(corpus tieng Anh: band descriptors, assessment criteria, test format).

Voi cau hoi cua nguoi dung, tra ve DUY NHAT mot JSON object (khong markdown, \
khong giai thich them) voi cac truong:
- "reformulated": cau hoi duoc viet lai standalone, ro rang, sua loi chinh \
ta; dich sang tieng Anh neu cau hoi goc bang ngon ngu khac, vi corpus la \
tieng Anh.
- "expanded_terms": mang toi da 5 tu/cum tu tieng Anh dong nghia hoac lien \
quan de mo rong pham vi tim kiem (vi du "band score" -> "band level", \
"scoring criteria").
- "sub_queries": neu cau hoi goc la CAU HOI KEP (hoi nhieu y trong mot cau, \
vi du "X la gi va no khac Y nhu the nao"), tach thanh 2-4 cau hoi con doc \
lap moi cau chi hoi MOT y (tieng Anh). Neu cau hoi da atomic/don gian, tra \
ve mang RONG []."""

_JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _fallback(query: str) -> dict:
    return {"original": query, "reformulated": query, "expanded_terms": [], "sub_queries": []}


def process_query(query: str) -> dict:
    """Goi 1 LLM call de reformulate + expand + decompose. Khong bao gio raise;
    loi/parse that bai deu fallback ve cau hoi goc (identity)."""
    with trace.stage("query_processing"):
        try:
            raw = call_text(_SYSTEM_PROMPT, f"User question: {query}", temperature=0.0, max_tokens=400)
            match = _JSON_PATTERN.search(raw)
            data = json.loads(match.group(0) if match else raw)
            result = {
                "original": query,
                "reformulated": str(data.get("reformulated") or query).strip() or query,
                "expanded_terms": [str(t).strip() for t in (data.get("expanded_terms") or []) if str(t).strip()][:5],
                "sub_queries": [str(q).strip() for q in (data.get("sub_queries") or []) if str(q).strip()][:4],
            }
        except (LLMError, json.JSONDecodeError, AttributeError, TypeError, ValueError) as error:
            trace.record("query_processing.error", error=repr(error))
            result = _fallback(query)

        trace.record("query_processing.result", **result)
        return result


def build_search_query(processed: dict) -> str:
    """Gop reformulated + expanded terms thanh MOT chuoi query cho retrieval."""
    query = processed["reformulated"]
    if processed["expanded_terms"]:
        query = f"{query} ({', '.join(processed['expanded_terms'])})"
    return query


def retrieve_with_query_processing(
    query: str,
    top_k: int,
    use_reranking: bool,
    score_threshold: float = _DEFAULT_SCORE_THRESHOLD,
) -> tuple[list[dict], dict]:
    """Reformulate/expand/decompose roi retrieve tren tung (sub-)query bang
    chinh `retrieve()` cua Task 9 khong doi (khong pha pinned contract), gop
    ket qua theo id (giu score cao nhat), sort giam dan, cat top_k."""
    processed = process_query(query)
    queries_used = processed["sub_queries"] or [build_search_query(processed)]

    merged: dict[str, dict] = {}
    for sub_query in queries_used:
        for item in retrieve(sub_query, top_k=top_k, score_threshold=score_threshold, use_reranking=use_reranking):
            existing = merged.get(item["id"])
            if existing is None or item["score"] > existing["score"]:
                merged[item["id"]] = item

    final = sorted(merged.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    trace.record(
        "query_processing.retrieval_merge",
        queries_used=queries_used,
        merged_ids=[item["id"] for item in final],
    )
    return final, processed
