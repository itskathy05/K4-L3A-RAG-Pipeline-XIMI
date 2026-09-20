"""
Task 10 -- Generation co citation.

Huong dan:
    1. Retrieve top-k chunks.
    2. Reorder de giam lost-in-the-middle.
    3. Format context kem title va source.
    4. Goi provider duoc chon trong .env.
    5. Tra answer, sources va retrieval_source.

Neu context khong du hoac provider loi, tra safe refusal; khong bia thong tin.

Hai bay quan trong (xem docs/MODULE_CONTRACTS.md va ke hoach trien khai):
    1. KHONG duoc gan `sources = reordered`. reorder_for_llm pha thu tu
       score, ma validate_search_results doi score giam dan. Danh so [Sn]
       theo list da reorder, roi anh xa nguoc ve vi tri trong list goc da
       sort (`chunks`, chua bi dao).
    2. `retrieval_source = chunks[0]["retrieval_method"]` SAI khi
       use_reranking=False: gia tri la "dense", khong thuoc
       {hybrid, pageindex, none}. Phai map ro rang.
"""

from __future__ import annotations

import os
import re

from dotenv import load_dotenv

from . import trace
from .llm import LLMError, call_text
from .query_processing import retrieve_with_query_processing
from .task9_retrieval_pipeline import retrieve

load_dotenv()

TOP_K = 5
TOP_P = 0.9
TEMPERATURE = 0.3

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_MODEL = os.getenv("LLM_MODEL", "")

SYSTEM_PROMPT = """Trả lời bằng tiếng Việt có dấu đầy đủ, chỉ dựa vào context được cung cấp.
Mỗi khẳng định phải có citation dạng [Sn] ứng với số thứ tự Document trong \
context (ví dụ [S1], [S2]). Chỉ được dùng số [Sn] nằm trong phạm vi các \
Document thực sự có trong context, không được bịa thêm số [Sn] ngoài phạm vi \
đó. Nếu thiếu evidence, hãy từ chối xác minh."""

SAFE_REFUSAL_TEXT = "Toi khong the xac minh thong tin nay tu nguon hien co."

_CITATION_PATTERN = re.compile(r"\[S(\d+)\]")


def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """Dua chunks quan trong ve dau va cuoi context (lost-in-the-middle)."""
    if len(chunks) <= 2:
        return list(chunks)
    front = chunks[::2]
    back = chunks[1::2]
    return front + back[::-1]


def format_context(chunks: list[dict]) -> str:
    """Tao context co title va source label, danh so theo thu tu da reorder."""
    parts = []
    for index, chunk in enumerate(chunks, 1):
        metadata = chunk["metadata"]
        parts.append(
            f"[Document {index} | Title: {metadata['title']} | "
            f"Source: {metadata['source']}]\n{chunk['content']}"
        )
    return "\n\n---\n\n".join(parts)


def call_llm(system_prompt: str, user_message: str) -> str:
    """Goi OpenAI, Gemini hoac Anthropic theo cau hinh (xem src/llm.py)."""
    return call_text(
        system_prompt,
        user_message,
        model=LLM_MODEL or None,
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )


def _map_retrieval_source(method: str) -> str:
    if method == "pageindex":
        return "pageindex"
    if method in ("hybrid", "dense", "bm25"):
        return "hybrid"
    return "none"


def _repair_citations(answer: str, reordered: list[dict], sorted_chunks: list[dict]) -> tuple[str, set[str]]:
    """Anh xa [Sn] (theo vi tri trong `reordered`) ve id chunk, roi danh so
    lai theo vi tri 1-based trong `sorted_chunks` (list goc, con nguyen thu
    tu score giam dan). Marker ngoai pham vi bi bo, ghi vao trace."""
    id_to_sorted_position = {chunk["id"]: i + 1 for i, chunk in enumerate(sorted_chunks)}
    cited_ids: set[str] = set()
    dropped: list[str] = []

    def _replace(match: re.Match) -> str:
        n = int(match.group(1))
        if 1 <= n <= len(reordered):
            chunk_id = reordered[n - 1]["id"]
            if chunk_id in id_to_sorted_position:
                cited_ids.add(chunk_id)
                return f"[S{id_to_sorted_position[chunk_id]}]"
        dropped.append(match.group(0))
        return ""

    repaired = _CITATION_PATTERN.sub(_replace, answer)
    if dropped:
        trace.record("citation.dropped", markers=dropped)
    return repaired, cited_ids


def _safe_refusal() -> dict:
    return {"answer": SAFE_REFUSAL_TEXT, "sources": [], "retrieval_source": "none"}


def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """Tra ve GenerationResult. Wrapper cong khai, chu ky bi pin boi
    test_public_function_signatures_are_stable -- logic thuc su nam o
    _generate_impl() de eval_runner co the chay Config A (dense-only, tuc
    use_reranking=False) qua cung mot code path voi demo/UI, thay vi viet
    lai mot ban sao rieng co the lech ket qua."""
    return _generate_impl(query, top_k=top_k, use_reranking=True)


def _generate_impl(query: str, top_k: int, use_reranking: bool, expand_query: bool = False) -> dict:
    with trace.stage("generation"):
        if expand_query:
            chunks, _processed = retrieve_with_query_processing(query, top_k=top_k, use_reranking=use_reranking)
        else:
            chunks = retrieve(query, top_k=top_k, use_reranking=use_reranking)
        if not chunks:
            trace.record("generation.result", reason="no_chunks_retrieved")
            return _safe_refusal()

        reordered = reorder_for_llm(chunks)
        trace.record(
            "reorder",
            order_before=[c["id"] for c in chunks],
            order_after=[c["id"] for c in reordered],
        )

        context = format_context(reordered)
        trace.record("context", text=context, char_len=len(context))

        user_message = (
            f"Context:\n{context}\n\n"
            f"(Chi co {len(reordered)} Document o tren, danh so tu [S1] den "
            f"[S{len(reordered)}]. Khong duoc dung so [Sn] ngoai pham vi nay.)\n\n"
            f"Question: {query}"
        )

        try:
            raw_answer = call_llm(SYSTEM_PROMPT, user_message)
        except LLMError as error:
            trace.record("generation.error", error=repr(error))
            return _safe_refusal()

        answer, cited_ids = _repair_citations(raw_answer, reordered, chunks)

        # Neu model khong phat sinh marker [Sn] nao (quan sat thuc te: hay xay
        # ra hon voi cau hoi cross-lingual hoac context da bi RRF/reorder xao
        # tron thu tu so voi dense-only), thu lai MOT lan voi yeu cau ro rang
        # truoc khi ha xuong safe refusal toan phan. Neu ha thang xuong refusal
        # ngay, mot cau tra loi dung noi dung nhung thieu format se bi tinh 0
        # diem oan tren ca 4 metric evaluation, che khuat chat luong retrieval
        # thuc su (xem group_project/evaluation/analysis.json).
        if not cited_ids:
            trace.record("citation.retry_triggered", raw_answer=raw_answer)
            retry_message = (
                f"{user_message}\n\nYour previous answer did not include any "
                f"[Sn] citation markers matching the Document numbers above. "
                f"Rewrite the same answer and add a [Sn] marker after every "
                f"claim, referencing the Document number it came from."
            )
            try:
                retry_answer = call_llm(SYSTEM_PROMPT, retry_message)
                answer, cited_ids = _repair_citations(retry_answer, reordered, chunks)
            except LLMError as error:
                trace.record("generation.error", error=repr(error), phase="citation_retry")
            trace.record("citation.retry_result", recovered=bool(cited_ids))

        if not cited_ids:
            trace.record("generation.result", reason="no_valid_citations")
            return _safe_refusal()

        # sources phai giu nguyen toan bo `chunks` (thu tu sort goc) vi
        # _repair_citations danh so [Sn] theo vi tri 1-based trong list nay;
        # loc bot phan tu se lam lech so thu tu giua answer va source card.
        sources = chunks
        retrieval_source = _map_retrieval_source(chunks[0]["retrieval_method"])

        result = {
            "answer": answer.strip(),
            "sources": sources,
            "retrieval_source": retrieval_source,
        }
        trace.record("generation.result", answer=result["answer"], retrieval_source=retrieval_source)
        return result


if __name__ == "__main__":
    print(generate_with_citation("test query"))
