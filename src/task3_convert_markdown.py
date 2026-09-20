"""
Task 3 -- Chuan hoa du lieu sang Markdown.

Hai nhanh xu ly khac nhau:
    - legal/: PDF/DOCX -> MarkItDown -> Markdown, giu header metadata.
    - news/: JSON (content_markdown tu crawl4ai) -> loc bo boilerplate
      (menu dieu huong, footer) -> Markdown, giu header metadata.

Phat hien boilerplate: cac trang ielts.org/idp.com deu render toan bo menu
dieu huong TRUOC noi dung bai viet va footer logo/copyright SAU noi dung.
Quan sat thuc te tren 8 file da crawl: nguyen 100% file co dung MOT heading
"^# " (H1) danh dau tieu de bai viet that su, va cung 100% file ket thuc
noi dung bang cum "[](https://ielts.org/)\n[](https://takeielts.britishcouncil.org)"
(logo lockup cua footer). Dung hai marker nay de cat noi dung; neu khong tim
thay marker nao (vi du nguon khac ielts.org sau nay) thi giu nguyen text goc
thay vi doan sai.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "standardized"
URLS_CSV = Path(__file__).parent.parent / "data" / "ielts_writing_urls.csv"

_H1_PATTERN = re.compile(r"(?m)^#\s+\S")
_FOOTER_MARKERS = [
    re.compile(re.escape("[](https://ielts.org/)\n[](https://takeielts.britishcouncil.org")),
    re.compile(r"(?m)^## Why choose IELTS\?"),
]


def _load_csv_index() -> dict[str, dict[str, str]]:
    """Map id -> row de lay publisher/title/url khi ghi header."""
    if not URLS_CSV.exists():
        return {}
    with URLS_CSV.open(encoding="utf-8") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def _doc_id_from_filename(path: Path) -> str:
    return path.stem.split("_", 1)[0]


def _write_header(
    *, title: str, doc_id: str, publisher: str, source: str, doc_type: str, crawled: str
) -> str:
    return (
        f"# {title}\n\n"
        f"**Document ID:** {doc_id}\n\n"
        f"**Publisher:** {publisher}\n\n"
        f"**Source:** {source}\n\n"
        f"**Document type:** {doc_type}\n\n"
        f"**Crawled:** {crawled}\n\n---\n\n"
    )


def strip_boilerplate(markdown: str) -> str:
    """Cat menu dieu huong dau trang va footer cuoi trang khoi markdown."""
    text = markdown

    start_match = _H1_PATTERN.search(text)
    if start_match:
        text = text[start_match.start():]

    end_positions = [m.start() for pattern in _FOOTER_MARKERS if (m := pattern.search(text))]
    if end_positions:
        text = text[: min(end_positions)]

    return text.strip()


def convert_legal_docs() -> None:
    """Convert PDF/DOCX vao standardized/legal."""
    from markitdown import MarkItDown

    legal_dir = LANDING_DIR / "legal"
    output_dir = OUTPUT_DIR / "legal"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_index = _load_csv_index()
    converter = MarkItDown()

    for path in sorted(legal_dir.iterdir()):
        if path.suffix.lower() not in {".pdf", ".doc", ".docx"}:
            continue
        target = output_dir / f"{path.stem}.md"
        if target.exists() and target.stat().st_size > 200:
            print(f"Skip (exists): {target.name}")
            continue

        result = converter.convert(str(path))
        body = result.text_content.strip()
        if not body:
            print(f"Skip (empty conversion): {path.name}")
            continue

        doc_id = _doc_id_from_filename(path)
        row = csv_index.get(doc_id, {})
        header = _write_header(
            title=row.get("title", path.stem),
            doc_id=doc_id,
            publisher=row.get("publisher", "unknown"),
            source=row.get("url", path.name),
            doc_type="legal",
            crawled="source document",
        )
        target.write_text(header + body, encoding="utf-8")
        print(f"Converted: {target.name}")


def convert_news_articles() -> None:
    """Convert JSON vao standardized/news, loai boilerplate menu/footer."""
    news_dir = LANDING_DIR / "news"
    output_dir = OUTPUT_DIR / "news"
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_index = _load_csv_index()

    for path in sorted(news_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        raw_body = data.get("content_markdown", "")
        body = strip_boilerplate(raw_body)
        if len(body) < 200:
            print(f"Skip (too short after cleaning, {len(body)} chars): {path.name}")
            continue

        doc_id = _doc_id_from_filename(path)
        row = csv_index.get(doc_id, {})
        header = _write_header(
            title=data.get("title", path.stem),
            doc_id=doc_id,
            publisher=row.get("publisher", "unknown"),
            source=data.get("url", ""),
            doc_type="news",
            crawled=data.get("date_crawled", ""),
        )
        target = output_dir / f"{path.stem}.md"
        target.write_text(header + body, encoding="utf-8")
        print(f"Converted: {target.name} ({len(raw_body)} -> {len(body)} chars)")


def convert_all() -> None:
    """Convert toan bo du lieu landing."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    convert_legal_docs()
    convert_news_articles()
    print(f"Saved Markdown to: {OUTPUT_DIR}")


if __name__ == "__main__":
    convert_all()
