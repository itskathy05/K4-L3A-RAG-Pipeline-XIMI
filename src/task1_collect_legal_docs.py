"""
Task 1 -- Thu thap tai lieu chinh sach/quy dinh.

Doc nguon tu data/ielts_writing_urls.csv (cot use="pdf_legal" hoac
"pdf_distractor"), tai bang requests va luu vao data/landing/legal/.
Idempotent: bo qua file da ton tai va dung dung byte, chi tai lai neu thieu
hoac rong.
"""

from __future__ import annotations

import csv
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
URLS_CSV = Path(__file__).parent.parent / "data" / "ielts_writing_urls.csv"

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)


def setup_directory() -> None:
    """Tao thu muc luu tai lieu goc."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Ready: {DATA_DIR}")


def _legal_rows() -> list[dict[str, str]]:
    if not URLS_CSV.exists():
        raise FileNotFoundError(f"Missing source list: {URLS_CSV}")
    with URLS_CSV.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("use", "").startswith("pdf_")]


def download_documents() -> None:
    """Tai it nhat 3 PDF/DOCX tu nguon cong khai liet ke trong CSV."""
    setup_directory()
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    for row in _legal_rows():
        url = row["url"]
        suffix = Path(url).suffix or ".pdf"
        filename = f"{row['id']}_{_slugify(row['title'])}{suffix}"
        target = DATA_DIR / filename

        if target.exists() and target.stat().st_size > 1024:
            print(f"Skip (exists): {target.name}")
            continue

        try:
            response = session.get(url, timeout=30)
            response.raise_for_status()
            target.write_bytes(response.content)
            print(f"Downloaded: {target.name} ({len(response.content)} bytes)")
        except requests.RequestException as error:
            print(f"Failed: {url} -- {error}")


def _slugify(title: str) -> str:
    keep = "".join(c.lower() if c.isalnum() else "_" for c in title)
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep.strip("_")


if __name__ == "__main__":
    setup_directory()
    download_documents()
