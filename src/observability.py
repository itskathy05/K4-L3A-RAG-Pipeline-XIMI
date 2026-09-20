"""
Diem hoi tu giua UI (app.py) va eval harness (eval_runner.py): ca hai deu
goi answer_with_trace() thay vi goi thang generate_with_citation(), de so
tren demo va so trong bao cao dung chung mot code path.

answer_with_trace KHONG BAO GIO raise -- moi loi (embedding API chet, Chroma
loi, LLM loi, v.v.) deu bi bat va chuyen thanh safe refusal + trace["error"].
Day la lop bao dam "provider loi khong duoc crash UI" o muc bien ma nguoi
dung thuc su cham vao, thay vi doi hoi tung ham noi bo (retrieve,
semantic_search, ...) tu chiu trach nhiem resilience rieng le.
"""

from __future__ import annotations

from . import trace as trace_module
from .task10_generation import SAFE_REFUSAL_TEXT, _generate_impl


def answer_with_trace(
    query: str,
    *,
    top_k: int = 5,
    use_reranking: bool = True,
    expand_query: bool = False,
) -> tuple[dict, dict]:
    """Chay pipeline day du va tra ve (GenerationResult, trace_dict).

    `use_reranking=False` chay Config A (dense-only) cho A/B eval, dung
    chung code path voi generate_with_citation() (Config B / demo UI mac
    dinh use_reranking=True) qua _generate_impl().

    `expand_query=True` bat lop tien xu ly query (decompose/expand/reformulate,
    xem src/query_processing.py) truoc khi retrieve. Mac dinh False de
    eval_runner (khong truyen tham so nay) giu nguyen so sanh A/B da co,
    khong bi lech them boi 1 LLM call moi.

    Khong bao gio raise. Loi duoc ghi vao trace["error"] va ket qua tra ve
    la safe refusal.
    """
    with trace_module.trace_run(label=query) as tr:
        try:
            result = _generate_impl(query, top_k=top_k, use_reranking=use_reranking, expand_query=expand_query)
        except Exception as error:  # bien cuoi cung: khong de UI/eval crash
            tr.error = repr(error)
            result = {
                "answer": SAFE_REFUSAL_TEXT,
                "sources": [],
                "retrieval_source": "none",
            }
        return result, tr.to_dict()
