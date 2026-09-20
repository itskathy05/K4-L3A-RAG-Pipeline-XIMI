"""
Task 2 -- Crawl bai viet/thong bao.

URL nguon tu data/ielts_writing_urls.csv (cot use="news"). Crawl bang
Crawl4AI, luu moi bai thanh mot JSON trong data/landing/news/. Idempotent:
bo qua URL da co file JSON hop le.
"""

from __future__ import annotations

import asyncio
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"
URLS_CSV = Path(__file__).parent.parent / "data" / "ielts_writing_urls.csv"

REQUIRED_KEYS = {"url", "title", "date_crawled", "content_markdown"}


def _news_rows() -> list[dict[str, str]]:
    if not URLS_CSV.exists():
        raise FileNotFoundError(f"Missing source list: {URLS_CSV}")
    with URLS_CSV.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [row for row in rows if row.get("use") == "news"]


def _slugify(title: str) -> str:
    keep = re.sub(r"[^a-z0-9]+", "_", title.lower())
    return keep.strip("_")


async def crawl_article(url: str, *, retries: int = 2, timeout_s: int = 60) -> dict:
    """Crawl mot URL va tra ve dict theo contract JSON cua Task 2."""
    from crawl4ai import AsyncWebCrawler, CrawlerRunConfig

    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            async with AsyncWebCrawler() as crawler:
                config = CrawlerRunConfig(page_timeout=timeout_s * 1000)
                result = await crawler.arun(url=url, config=config)
                if not result.success:
                    raise RuntimeError(result.error_message or "crawl4ai reported failure")
                title = (result.metadata or {}).get("title") or url
                return {
                    "url": url,
                    "title": title,
                    "date_crawled": datetime.now(timezone.utc).isoformat(),
                    "content_markdown": result.markdown or "",
                }
        except Exception as error:  # site chậm/site chặn -> thử lại
            last_error = error
            print(f"  attempt {attempt}/{retries} failed for {url}: {error}")
    raise RuntimeError(f"All {retries} attempts failed for {url}: {last_error}")


async def crawl_all() -> None:
    """Crawl va luu tung bai thanh mot file JSON."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for row in _news_rows():
        url = row["url"]
        output = DATA_DIR / f"{row['id']}_{_slugify(row['title'])}.json"

        if output.exists():
            try:
                existing = json.loads(output.read_text(encoding="utf-8"))
                if REQUIRED_KEYS <= existing.keys() and all(
                    str(existing[k]).strip() for k in REQUIRED_KEYS
                ):
                    print(f"Skip (exists): {output.name}")
                    continue
            except (json.JSONDecodeError, OSError):
                pass  # file hỏng -> crawl lại

        try:
            article = await crawl_article(url)
            output.write_text(
                json.dumps(article, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"Saved: {output.name}")
        except Exception as error:
            print(f"Failed: {url} -- {error}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
