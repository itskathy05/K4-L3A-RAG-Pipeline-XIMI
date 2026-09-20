"""
Task 5 -- Semantic search.

Embed query bang chinh ham cua Task 4, query ChromaDB va doi cosine distance
thanh similarity. Output phai theo SearchResult, sort giam dan va khong qua
top_k.
"""

from . import trace
from .task4_chunking_indexing import embed_texts, get_collection


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """Tra ve dense SearchResult theo score giam dan."""
    query_vector = embed_texts([query])[0]
    response = get_collection().query(
        query_embeddings=[query_vector],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    results = []
    for item_id, content, metadata, distance in zip(
        response["ids"][0],
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        results.append(
            {
                "id": item_id,
                "content": content,
                "score": max(0.0, 1.0 - distance),
                "metadata": metadata,
                "retrieval_method": "dense",
            }
        )
    results = sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]
    trace.record(
        "dense",
        query=query,
        top_k=top_k,
        results=[{"id": r["id"], "score": r["score"], "metadata": r["metadata"]} for r in results],
    )
    return results


if __name__ == "__main__":
    for result in semantic_search("test query", top_k=3):
        print(result)
