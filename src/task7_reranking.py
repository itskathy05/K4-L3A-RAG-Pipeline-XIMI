"""
Task 7 -- Reciprocal Rank Fusion.

RRF gop nhieu bang xep hang ma khong cong truc tiep cosine score voi BM25
score. Cong thuc: RRF(d) = sum(1 / (k + rank)), rank bat dau tu 1.

Luu y: RRF score chi phan anh thu hang, khong dung de quyet dinh fallback.
"""

from . import trace


def rerank_rrf(
    ranked_lists: list[list[dict]],
    top_k: int = 5,
    k: int = 60,
) -> list[dict]:
    """Fuse nhieu ranked lists va tra hybrid SearchResult."""
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    per_list_rank: dict[str, dict[int, int]] = {}

    for list_index, ranked_list in enumerate(ranked_lists):
        for rank, item in enumerate(ranked_list, 1):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1 / (k + rank)
            items[item_id] = item
            per_list_rank.setdefault(item_id, {})[list_index] = rank

    ranked_ids = sorted(scores, key=scores.get, reverse=True)
    results = []
    for item_id in ranked_ids[:top_k]:
        result = items[item_id].copy()
        result["score"] = scores[item_id]
        result["retrieval_method"] = "hybrid"
        results.append(result)

    all_ids = sorted(scores, key=scores.get, reverse=True)
    candidates = []
    for rank, item_id in enumerate(all_ids, 1):
        ranks = per_list_rank.get(item_id, {})
        candidates.append(
            {
                "id": item_id,
                "dense_rank": ranks.get(0),
                "bm25_rank": ranks.get(1),
                "rrf_score": scores[item_id],
                "final_rank": rank,
                "selected": rank <= top_k,
            }
        )
    trace.record("fusion", k=k, top_k=top_k, candidates=candidates)

    return results


if __name__ == "__main__":
    print("Implement rerank_rrf, then run contract tests.")
