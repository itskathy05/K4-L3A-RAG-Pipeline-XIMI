"""
Task 6 -- Lexical search bang BM25.

Dung cung corpus chunks voi Task 5 (nguon su that la ChromaDB, xem
load_corpus()). BM25 phu hop voi tu khoa chinh xac, ma tai lieu va ten
rieng. Output phai theo SearchResult va sort score giam dan.

Tokenizer dung \\w+ (Unicode) thay vi [a-z0-9]+: regex ASCII se bam vu cac
tu co dau (vd "tieu chi" -> "ti","u","ch") neu nguoi dung hoi bang tieng
Viet, du corpus hien tai la tieng Anh. Them ban bo dau giup cau hoi go
khong dau van khop.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

from . import trace

CORPUS: list[dict] = []
_BM25_CACHE: tuple[str, object] | None = None


def _strip_accents(token: str) -> str:
    decomposed = unicodedata.normalize("NFD", token)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"\w+", text.lower(), re.UNICODE)
    extra = [_strip_accents(t) for t in tokens if _strip_accents(t) != t]
    return tokens + extra


def load_corpus() -> list[dict]:
    """Doc toan bo chunk da index tu ChromaDB (lazy, khong chay luc import)."""
    global CORPUS
    if CORPUS:
        return CORPUS

    from .task4_chunking_indexing import get_collection

    collection = get_collection()
    response = collection.get(include=["documents", "metadatas"])
    corpus = []
    for item_id, content, metadata in zip(
        response["ids"], response["documents"], response["metadatas"]
    ):
        corpus.append({"id": item_id, "content": content, "metadata": metadata})
    corpus.sort(key=lambda c: c["id"])
    CORPUS = corpus
    return CORPUS


def build_bm25_index(corpus: list[dict]):
    """Tao BM25 index tu cung corpus chunks cua Task 4."""
    from rank_bm25 import BM25Okapi

    tokenized = [_tokenize(item["content"]) for item in corpus]
    return BM25Okapi(tokenized)


def _corpus_fingerprint(corpus: list[dict]) -> str:
    ids = "|".join(item["id"] for item in corpus)
    return hashlib.sha1(ids.encode("utf-8")).hexdigest()


def _get_bm25(corpus: list[dict]):
    global _BM25_CACHE
    fingerprint = _corpus_fingerprint(corpus)
    if _BM25_CACHE is not None and _BM25_CACHE[0] == fingerprint:
        return _BM25_CACHE[1]
    index = build_bm25_index(corpus)
    _BM25_CACHE = (fingerprint, index)
    return index


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """Tra ve BM25 SearchResult theo score giam dan."""
    import numpy as np

    corpus = CORPUS if CORPUS else load_corpus()
    if not corpus:
        return []

    bm25 = _get_bm25(corpus)
    query_tokens = set(_tokenize(query))
    scores = bm25.get_scores(list(query_tokens))

    # Loc bang overlap token that su, KHONG loc theo dau cua score. BM25 idf
    # co the ra dung 0 cho moi tu khi corpus qua nho (vd 2 tai lieu, moi tu
    # chi xuat hien trong dung 1 doc: idf = log(N-1+0.5)-log(1+0.5) = 0 voi
    # N=2) du van co trung tu khoa that su -- loc theo "score <= 0" trong
    # truong hop do se xoa sach ket qua co y nghia. "Khong trung tu nao" moi
    # la dieu kien dung de loai bo mot chunk.
    overlap_mask = [
        bool(query_tokens & set(_tokenize(item["content"]))) for item in corpus
    ]

    # argsort(-scores, kind="stable") thay vi argsort(scores)[::-1]: cach thu
    # hai dao nguoc thu tu ca voi cac phan tu diem bang nhau, lam dao lon thu
    # tu goc cua corpus. Stable sort tren -scores giu nguyen thu tu goc giua
    # cac phan tu hoa diem (quan trong khi corpus nho khien nhieu tu hoa 0).
    indices = np.argsort(-scores, kind="stable")

    results = []
    for index in indices:
        if len(results) >= top_k:
            break
        if not overlap_mask[index]:
            continue
        item = corpus[index]
        results.append(
            {
                "id": item["id"],
                "content": item["content"],
                "score": float(scores[index]),
                "metadata": item["metadata"],
                "retrieval_method": "bm25",
            }
        )
    trace.record(
        "bm25",
        query=query,
        top_k=top_k,
        results=[{"id": r["id"], "score": r["score"], "metadata": r["metadata"]} for r in results],
    )
    return results


if __name__ == "__main__":
    for result in lexical_search("test query", top_k=3):
        print(result)
