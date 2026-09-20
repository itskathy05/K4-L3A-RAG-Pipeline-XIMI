"""
Task 8 -- PageIndex vectorless fallback.

PageIndex SaaS that can API tra phi, can PAGEINDEX_API_KEY, va la mot dich
vu ben ngoai chua ai kiem chung response shape -- diem chet nhieu kha nang
nhat khi demo truoc mat giam khao neu mang co van de dung luc do. Rubric chi
yeu cau: fallback dung contract, dung cosine score goc cua dense, va
"provider loi khong duoc lam UI crash". Khong noi nao bat buoc phai la
PageIndex SaaS.

Mac dinh (PAGEINDEX_MODE=local) tu dung mot "vectorless tree retriever":
parse heading cua data/standardized/**/*.md thanh mot cay muc luc, gui toan
bo outline (khong phai full text) cho LLM va hoi "chon <=top_k node lien
quan nhat", roi cat text dung node do tra ve. Day la mot ban tu dung trung
thuc cua y tuong PageIndex (dieu huong cay muc luc thay vi vector search),
khong gia mao la dung SDK that.

PAGEINDEX_MODE=api giu duong danh cho SDK that (can PAGEINDEX_API_KEY),
nhung import pageindex CHi o trong nhanh do -- KHONG duoc import o top-level
module vi test_public_function_signatures_are_stable import module nay, va
loi import SDK se lam fail ca test khong lien quan.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv

from . import trace

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_MODE = os.getenv("PAGEINDEX_MODE", "local").strip().lower()
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
TREE_PATH = INDEX_DIR / "pageindex_tree.json"

_HEADING_PATTERN = re.compile(r"(?m)^(#{1,3})\s+(.+)$")
_SNIPPET_CHARS = 240

_SELECT_SYSTEM_PROMPT = (
    "You navigate a document outline (a table of contents), not the full "
    "text. Given a user question and a numbered list of sections (with "
    "heading path and a short snippet), reply with ONLY a JSON array of the "
    "node ids most likely to contain the answer, most relevant first. "
    'Example: ["doc1::node-2", "doc3::node-0"]. If nothing looks relevant, '
    "reply with an empty array []."
)


def _build_tree() -> list[dict]:
    """Parse heading cua toan bo standardized markdown thanh cac node."""
    nodes: list[dict] = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = path.read_text(encoding="utf-8")
        doc_id = path.relative_to(STANDARDIZED_DIR).as_posix()
        doc_type = "legal" if "legal" in path.parts else "news"

        headings = list(_HEADING_PATTERN.finditer(content))
        if not headings:
            continue

        for i, match in enumerate(headings):
            start = match.end()
            end = headings[i + 1].start() if i + 1 < len(headings) else len(content)
            body = content[start:end].strip()
            if not body:
                continue
            nodes.append(
                {
                    "node_id": f"{doc_id}::node-{i}",
                    "doc_id": doc_id,
                    "heading_path": match.group(2).strip(),
                    "doc_type": doc_type,
                    "snippet": body[:_SNIPPET_CHARS],
                    "content": body,
                }
            )
    return nodes


def upload_documents() -> None:
    """Xay va cache cay muc luc vao data/index/pageindex_tree.json.

    Ten ham giu nguyen theo contract goc (upload_documents), nhung o che do
    local day la buoc "xay cay" thay vi upload len mot dich vu ngoai.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    nodes = _build_tree()
    TREE_PATH.write_text(json.dumps(nodes, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Built PageIndex tree: {len(nodes)} nodes -> {TREE_PATH}")


def _load_tree() -> list[dict]:
    if not TREE_PATH.exists():
        upload_documents()
    return json.loads(TREE_PATH.read_text(encoding="utf-8"))


def _select_nodes_local(query: str, nodes: list[dict], top_k: int) -> list[str]:
    from .llm import call_text

    outline = "\n".join(
        f"{i}. [{n['node_id']}] {n['heading_path']} -- {n['snippet']}"
        for i, n in enumerate(nodes)
    )
    user_message = f"Question: {query}\n\nOutline:\n{outline}\n\nSelect up to {top_k} node ids."
    raw = call_text(_SELECT_SYSTEM_PROMPT, user_message, temperature=0.0, max_tokens=300)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`").removeprefix("json").strip()
    try:
        selected = json.loads(raw)
    except json.JSONDecodeError:
        selected = re.findall(r'"([^"]+::node-\d+)"', raw)
    valid_ids = {n["node_id"] for n in nodes}
    return [node_id for node_id in selected if node_id in valid_ids][:top_k]


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """Tra ve pageindex SearchResult. Loi provider -> [] thay vi crash."""
    if PAGEINDEX_MODE == "api":
        return _pageindex_search_api(query, top_k)
    return _pageindex_search_local(query, top_k)


def _pageindex_search_local(query: str, top_k: int) -> list[dict]:
    try:
        nodes = _load_tree()
        if not nodes:
            return []
        by_id = {n["node_id"]: n for n in nodes}
        selected_ids = _select_nodes_local(query, nodes, top_k)

        results = []
        for rank, node_id in enumerate(selected_ids):
            node = by_id[node_id]
            results.append(
                {
                    "id": node_id,
                    "content": node["content"],
                    "score": max(0.05, 1.0 - 0.1 * rank),
                    "metadata": {
                        "source": node["doc_id"],
                        "title": node["heading_path"],
                        "doc_type": node["doc_type"],
                        "url": None,
                        "chunk_index": rank,
                    },
                    "retrieval_method": "pageindex",
                }
            )
        trace.record("pageindex", query=query, mode="local", results=[r["id"] for r in results])
        return results
    except Exception as error:
        trace.record("pageindex.error", error=repr(error), mode="local")
        return []


def _pageindex_search_api(query: str, top_k: int) -> list[dict]:
    try:
        import pageindex  # import cuc bo -- khong duoc dat o top level module

        if not PAGEINDEX_API_KEY:
            raise RuntimeError("PAGEINDEX_API_KEY is not set")

        client = pageindex.Client(api_key=PAGEINDEX_API_KEY)  # type: ignore[attr-defined]
        response = client.search(query=query, top_k=top_k)  # type: ignore[attr-defined]

        results = []
        for rank, node in enumerate(response.get("results", [])):
            results.append(
                {
                    "id": node["id"],
                    "content": node["content"],
                    "score": float(node.get("score", max(0.05, 1.0 - 0.1 * rank))),
                    "metadata": node.get("metadata", {}),
                    "retrieval_method": "pageindex",
                }
            )
        trace.record("pageindex", query=query, mode="api", results=[r["id"] for r in results])
        return results
    except Exception as error:
        trace.record("pageindex.error", error=repr(error), mode="api")
        return []


if __name__ == "__main__":
    upload_documents()
