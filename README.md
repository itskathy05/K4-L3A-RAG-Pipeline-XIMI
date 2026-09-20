# IELTS Writing RAG Chatbot

## Mục tiêu

Chatbot RAG trả lời câu hỏi về **IELTS Writing** (band descriptors, key assessment criteria, test format) dựa trên bộ tài liệu tự thu thập từ ielts.org. Sản phẩm có hybrid retrieval (dense + BM25 + RRF), fallback vectorless khi câu hỏi ngoài phạm vi, generation có citation, giao diện chat 3 tab (Chat / Inspector / Evaluation), và báo cáo đánh giá A/B.

## Kiến trúc & quyết định kỹ thuật chính

- **Corpus:** 3 tài liệu chính sách (band descriptors PDF, key assessment criteria PDF, academic writing sample tasks PDF) + 8 bài viết ielts.org, tất cả tiếng Anh. Nguồn liệt kê tại `data/ielts_writing_urls.csv`.
- **Embedding:** `text-embedding-3-small` qua OpenAI API (`EMBEDDING_PROVIDER=openai`). Đã kiểm chứng thủ công: câu hỏi tiếng Việt vẫn truy hồi đúng nội dung tiếng Anh (cross-lingual retrieval hoạt động tốt dù không dùng model `-large`).
- **Chunking:** hai nhánh. Band descriptors (PDF bị `markitdown` làm nát cấu trúc bảng 4 cột) được tách riêng theo từng mức band (Band 9→4, thấp hơn gộp chung) thay vì cắt mù 500 ký tự — xem `src/task4_chunking_indexing.py::_chunk_band_descriptor_document`. Tài liệu thường dùng `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter`.
- **BM25:** tokenizer Unicode (`\w+` + bản bỏ dấu) để chịu được câu hỏi tiếng Việt, dù corpus hiện tại toàn tiếng Anh.
- **Fallback (Task 8):** mặc định `PAGEINDEX_MODE=local` — tự dựng "vectorless tree retriever" (parse heading `data/standardized/**/*.md`, LLM chọn node liên quan) thay vì gọi PageIndex SaaS trả phí. Lý do: rubric chỉ yêu cầu đúng contract (`retrieval_method="pageindex"`, không crash khi lỗi), không bắt buộc dùng đúng vendor; một dịch vụ ngoài chưa kiểm chứng response shape là điểm chết tiềm ẩn khi demo. `PAGEINDEX_MODE=api` giữ đường dẫn cho SDK thật.
- **Threshold:** hiệu chỉnh bằng `scripts/calibrate_threshold.py` trên 16 câu in-domain (golden dataset) và 10 câu out-of-domain/near-miss thật. Kết quả: `SCORE_THRESHOLD=0.58` (TPR=1.0, TNR=0.7 — xem `reports/threshold_calibration.json`).
- **Observability:** `src/trace.py` dùng `ContextVar` để các hàm retrieval/generation ghi lại từng bước (dense results, BM25 results, RRF rank movement, fallback decision, reorder, context, citation) mà **không** thêm tham số vào các hàm đã bị `tests/test_contracts.py` pin chữ ký. `src/observability.py::answer_with_trace()` là điểm hội tụ duy nhất giữa UI và eval harness — không bao giờ raise.

## Sản phẩm phải nộp

- Repository nhóm chạy được.
- Tối thiểu 3 tài liệu chính sách và 5 bài viết/page do nhóm tự thu thập.
- Pipeline: convert → chunk → index → dense + BM25 → RRF → fallback → generation có citation.
- Chatbot Streamlit hiển thị câu trả lời và nguồn đã dùng.
- Golden dataset tối thiểu 15 câu; đánh giá 4 metric và so sánh A/B.
- `group_project/evaluation/RESULT.md`.
- Mỗi thành viên nộp báo cáo cá nhân theo template trong `group_project/ịndividual/INDIVIDUAL_REPORT.md`.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e ".[dev]"
python -m playwright install chromium
cp .env.example .env
```

Điền API key cần dùng trong `.env`; không commit file này. `.env` cần tối thiểu `OPENAI_API_KEY` (dùng chung cho embedding, generation và RAGAS judge với cấu hình mặc định).

```bash
# 1. Thu thập và chuẩn hoá (idempotent — bỏ qua file đã tồn tại)
python -m src.task1_collect_legal_docs
python -m src.task2_crawl_news
python -m src.task3_convert_markdown

# 2. Index (chunk + embed qua OpenAI + upsert Chroma)
python -m src.task4_chunking_indexing

# 3. Hiệu chỉnh lại threshold nếu đổi corpus/embedding model
python scripts/calibrate_threshold.py    # ghi kết quả vào reports/threshold_calibration.json

# 4. Kiểm tra
pytest -q

# 5. Chạy sản phẩm (3 tab: Chat / Inspector / Evaluation)
streamlit run app.py

# 6. Evaluation A/B (dense-only vs hybrid+RRF), tốn vài cent OpenAI API
python -m src.eval_runner --config both
python -m src.eval_report --check
```

## Lưu ý quy tắc để có code quality tốt:

- Dense và BM25 nên cùng trả về `SearchResult` theo một schema.
- RRF chỉ nên dùng để gộp thứ hạng và chỉ chạy một lần.
- Fallback dùng cosine score gốc của dense retrieval.
- Threshold phải được hiệu chỉnh trên query in domain và out of domain, không có một con số đúng cho mọi corpus.

## Tài liệu

- [Module contracts](docs/MODULE_CONTRACTS.md): schema, interface và invariant mà code/test nên tuân theo.
- [Step-by-step guide](docs/STEP_BY_STEP.md): thứ tự triển khai và tiêu chí hoàn thành từng bước.
- [Grading rubric](docs/GRADING_RUBRIC.md): Rubric thang điểm.
- [Individual report](group_project/ịndividual/INDIVIDUAL_REPORT.md): template báo cáo cá nhân.
- [Suggested topics](docs/SUGGESTED_TOPICS.md): danh sách chủ đề tham khảo, không bắt buộc.

## Kiểm tra

```bash
# Contract tests
pytest tests/test_contracts.py -q

# Acceptance tests
pytest tests/test_acceptance.py -q

# Toàn bộ
pytest -q
```
