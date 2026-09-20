"""
Task 4 -- Chunking, embedding va indexing.

Hai duong xu ly chunk khac nhau:
    - Band descriptor table (vd C01): markitdown lam nat cau truc bang 4 cot
      khi convert PDF, ket qua la text bi tron cot. Thay vi chunk mu 500 ky
      tu (se cat ngang giua mot hang band, mat het thong tin "day la Band
      may / tieu chi nao"), ta phat hien pattern lap "Writing Task N Band
      Descriptors" + cac anchor so band (9,8,7,6,5,4) va tach thanh 1 chunk
      cho moi muc band, giu nguyen 4 tieu chi bi tron trong cung 1 chunk.
      Day khong phai tach hoan hao theo tung o (Task Achievement / Coherence
      / Lexical / Grammar) vi text da bi tron cot boi PDF extraction, nhung
      tot hon nhieu so voi cat mu: query "Band 7 requirements" se keo dung
      toan bo doan Band 7 thay vi mot manh 500 ky tu ngau nhien.
    - Van ban thuong: MarkdownHeaderTextSplitter (giu section_path lam
      metadata, KHONG prepend vao content de khong pha rang buoc do dai) ->
      RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE).

Moi document/chunk phai theo docs/MODULE_CONTRACTS.md. Task 5 phai dung
chung embed_texts() -- import trong task5 la
`from .task4_chunking_indexing import embed_texts`, KHONG duoc doi thanh
`task4.embed_texts(...)` vi monkeypatch trong test nham vao binding do.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "chroma_db"
INDEX_DIR = Path(__file__).parent.parent / "data" / "index"
EMBED_CACHE_PATH = INDEX_DIR / "embed_cache.jsonl"
CHUNKS_MIRROR_PATH = INDEX_DIR / "chunks.jsonl"

# Giai thich lua chon tham so trong bao cao nhom.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "openai").strip().lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip()
_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "BAAI/bge-m3": 1024,
}
EMBEDDING_DIM = _DIMS.get(EMBEDDING_MODEL, 1536)

COLLECTION_NAME = "rag_documents"

_BAND_TASK_MARKER = re.compile(r"Writing Task (\d) Band Descriptors")
_BAND_SEQUENCE = [9, 8, 7, 6, 5, 4]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def _embed_cache_key(model: str, text: str) -> str:
    return hashlib.sha256(f"{model}\0{text}".encode("utf-8")).hexdigest()


def _load_embed_cache() -> dict[str, list[float]]:
    if not EMBED_CACHE_PATH.exists():
        return {}
    cache: dict[str, list[float]] = {}
    with EMBED_CACHE_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cache[record["key"]] = record["embedding"]
    return cache


def _append_embed_cache(entries: list[tuple[str, list[float]]]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with EMBED_CACHE_PATH.open("a", encoding="utf-8") as handle:
        for key, embedding in entries:
            handle.write(json.dumps({"key": key, "embedding": embedding}) + "\n")


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Dispatch theo EMBEDDING_PROVIDER, dung cache dia va batch <=100."""
    if not texts:
        return []

    if EMBEDDING_PROVIDER == "sentence_transformers":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(EMBEDDING_MODEL)
        return model.encode(texts).tolist()

    if EMBEDDING_PROVIDER not in {"openai", "gemini"}:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER!r}")

    cache = _load_embed_cache()
    keys = [_embed_cache_key(EMBEDDING_MODEL, text) for text in texts]
    results: list[list[float] | None] = [cache.get(key) for key in keys]

    missing_indices = [i for i, value in enumerate(results) if value is None]
    if missing_indices:
        missing_texts = [texts[i] for i in missing_indices]
        fresh = _embed_texts_remote(missing_texts)
        new_cache_entries = []
        for local_i, global_i in enumerate(missing_indices):
            results[global_i] = fresh[local_i]
            new_cache_entries.append((keys[global_i], fresh[local_i]))
        _append_embed_cache(new_cache_entries)

    return [r for r in results if r is not None]


def _embed_texts_remote(texts: list[str]) -> list[list[float]]:
    if EMBEDDING_PROVIDER == "openai":
        return _embed_openai(texts)
    if EMBEDDING_PROVIDER == "gemini":
        return _embed_gemini(texts)
    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {EMBEDDING_PROVIDER!r}")


def _embed_openai(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), max_retries=5, timeout=60)
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
                vectors.extend(item.embedding for item in response.data)
                break
            except Exception as error:  # rate limit / timeout -> backoff + retry
                last_error = error
                time.sleep(2**attempt)
        else:
            raise RuntimeError(f"OpenAI embeddings failed after retries: {last_error}")
    return vectors


def _embed_gemini(texts: list[str]) -> list[list[float]]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
    )
    return [e.values for e in response.embeddings]


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

def get_collection():
    """Mo hoac tao Chroma collection dung cosine distance."""
    import chromadb
    from chromadb.errors import NotFoundError

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    try:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=None)
    except NotFoundError:
        return client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
            embedding_function=None,
        )

    stored_model = (collection.metadata or {}).get("embedding_model")
    if stored_model and stored_model != EMBEDDING_MODEL:
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' was indexed with embedding_model="
            f"{stored_model!r} but current EMBEDDING_MODEL={EMBEDDING_MODEL!r}. "
            "Xoa thu muc chroma_db/ va chay lai `python -m src.task4_chunking_indexing` "
            "de index lai, hoac chinh lai .env cho khop."
        )
    return collection


# ---------------------------------------------------------------------------
# Load documents
# ---------------------------------------------------------------------------

def _extract_title(content: str, fallback: str) -> str:
    match = re.search(r"(?m)^#\s+(.+)$", content)
    return match.group(1).strip() if match else fallback


def _extract_source(content: str, fallback: str) -> str:
    match = re.search(r"(?m)^\*\*Source:\*\*\s*(.+)$", content)
    return match.group(1).strip() if match else fallback


def load_documents() -> list[dict]:
    """Doc moi .md va tao Document theo contract."""
    documents = []
    for path in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = path.read_text(encoding="utf-8")
        if not content.strip():
            continue
        doc_type = "legal" if "legal" in path.parts else "news"
        source_value = _extract_source(content, path.name)
        url = source_value if source_value.startswith("http") else None
        documents.append(
            {
                "id": path.relative_to(STANDARDIZED_DIR).as_posix(),
                "content": content,
                "metadata": {
                    "source": path.name,
                    "title": _extract_title(content, path.stem),
                    "doc_type": doc_type,
                    "url": url,
                },
            }
        )
    return documents


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _looks_like_band_descriptor_table(content: str) -> bool:
    """Phat hien tai lieu co cau truc bang band descriptor bi markitdown lam
    nat cot (vd C01), de ap dung chunker rieng thay vi cat mu 500 ky tu."""
    markers = _BAND_TASK_MARKER.findall(content)
    return len(markers) >= 2 and "9" in content and "8" in content


def _split_band_descriptor_sections(content: str) -> list[tuple[str, str]]:
    """Tra ve list (task_number, section_text) tach theo lan doi task dau
    tien trong chuoi marker 'Writing Task N Band Descriptors' lap lai."""
    markers = list(_BAND_TASK_MARKER.finditer(content))
    if not markers:
        return [("", content)]

    first_task = markers[0].group(1)
    boundary = next((m.start() for m in markers if m.group(1) != first_task), None)

    if boundary is None:
        return [(first_task, content)]

    second_task = next(m.group(1) for m in markers if m.start() >= boundary)
    return [(first_task, content[:boundary]), (second_task, content[boundary:])]


def _chunk_band_section(task_label: str, section_text: str) -> list[tuple[str, str]]:
    """Tach mot section (mot task) thanh chunk theo tung muc band.

    Tra ve list (band_label, band_text). Bang thap (0-4) duoc gop chung
    thanh mot chunk "Band 4 va thap hon" vi hiem khi duoc hoi rieng va tach
    tiep se chi lam vun mot doan text da bi tron cot san.
    """
    cursor = 0
    boundaries: list[tuple[int, str]] = []
    for band in _BAND_SEQUENCE:
        match = re.search(rf"(?m)^{band}(?:\s|\()", section_text[cursor:])
        if match is None:
            break
        boundaries.append((cursor + match.start(), str(band)))
        cursor = cursor + match.start() + 1

    if not boundaries:
        return [(f"Task {task_label}", section_text)]

    chunks: list[tuple[str, str]] = []
    for i, (start, band) in enumerate(boundaries):
        end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(section_text)
        label = f"Band {band}" if band != "4" else "Band 4 va thap hon"
        chunks.append((label, section_text[start:end].strip()))
    return chunks


def _chunk_band_descriptor_document(document: dict) -> list[dict]:
    content = document["content"]
    doc_id = document["id"]
    metadata = document["metadata"]

    chunks: list[dict] = []
    index = 0
    for task_label, section_text in _split_band_descriptor_sections(content):
        for band_label, band_text in _chunk_band_section(task_label, section_text):
            if not band_text.strip():
                continue
            heading = (
                f"Writing Task {task_label} Band Descriptors -- {band_label}"
                if task_label
                else band_label
            )
            chunks.append(
                {
                    "id": f"{doc_id}::chunk-{index}",
                    "content": band_text,
                    "metadata": {
                        **metadata,
                        "chunk_index": index,
                        "section_path": heading,
                    },
                }
            )
            index += 1
    return chunks


def _chunk_default_document(document: dict) -> list[dict]:
    from langchain_text_splitters import (
        MarkdownHeaderTextSplitter,
        RecursiveCharacterTextSplitter,
    )

    content = document["content"]
    doc_id = document["id"]
    metadata = document["metadata"]

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    try:
        header_docs = header_splitter.split_text(content) or [None]
    except Exception:
        header_docs = [None]

    if header_docs == [None] or not header_docs:
        sections = [(content, "")]
    else:
        sections = []
        for hdoc in header_docs:
            section_path = " > ".join(
                str(v) for k, v in hdoc.metadata.items() if k in ("h1", "h2", "h3") and v
            )
            sections.append((hdoc.page_content, section_path))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: list[dict] = []
    index = 0
    for section_text, section_path in sections:
        pieces = [p for p in splitter.split_text(section_text) if p.strip()]
        for piece in pieces:
            chunk_metadata = {**metadata, "chunk_index": index}
            if section_path:
                chunk_metadata["section_path"] = section_path
            chunks.append(
                {
                    "id": f"{doc_id}::chunk-{index}",
                    "content": piece,
                    "metadata": chunk_metadata,
                }
            )
            index += 1
    return chunks


def chunk_documents(documents: list[dict]) -> list[dict]:
    """Chia Document thanh chunks co id va chunk_index."""
    all_chunks: list[dict] = []
    for document in documents:
        if _looks_like_band_descriptor_table(document["content"]):
            all_chunks.extend(_chunk_band_descriptor_document(document))
        else:
            all_chunks.extend(_chunk_default_document(document))
    return all_chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Them embedding vao tung chunk, giu nguyen cac field khac."""
    vectors = embed_texts([chunk["content"] for chunk in chunks])
    embedded = []
    for chunk, vector in zip(chunks, vectors):
        embedded.append({**chunk, "embedding": vector})
    return embedded


def index_to_vectorstore(chunks: list[dict]) -> None:
    """Upsert chunks vao ChromaDB."""
    if not chunks:
        print("No chunks to index")
        return
    collection = get_collection()
    collection.upsert(
        ids=[chunk["id"] for chunk in chunks],
        documents=[chunk["content"] for chunk in chunks],
        embeddings=[chunk["embedding"] for chunk in chunks],
        metadatas=[chunk["metadata"] for chunk in chunks],
    )
    _write_chunks_mirror(chunks)


def _write_chunks_mirror(chunks: list[dict]) -> None:
    """Ghi ban sao offline (khong co embedding) dung cho script hieu chinh
    threshold va cay PageIndex. Nguon su that van la Chroma; day chi la mirror."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with CHUNKS_MIRROR_PATH.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            record = {k: v for k, v in chunk.items() if k != "embedding"}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_pipeline() -> None:
    """Chay load, chunk, embed va index."""
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")
    embedded_chunks = embed_chunks(chunks)
    index_to_vectorstore(embedded_chunks)
    print(f"Indexed {len(embedded_chunks)} chunks")


if __name__ == "__main__":
    run_pipeline()
