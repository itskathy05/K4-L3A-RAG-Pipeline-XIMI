"""Tab Demo -- "slide" gioi thieu san pham va giai thich tung buoc pipeline.
Thuan tuy trinh bay tinh (khong goi pipeline that), dung de intro truoc khi
demo tab Chat/Inspector."""

from __future__ import annotations

import streamlit as st

SLIDES: list[dict] = [
    {
        "title": "IELTS Writing RAG Chatbot",
        "kicker": "Tong quan",
        "body": (
            "Chatbot tra loi cau hoi ve **IELTS Writing** (band descriptors, "
            "tieu chi cham diem, dinh dang bai thi) dua tren tai lieu that "
            "thu thap tu ielts.org.\n\n"
            "**Diem manh chinh:**\n"
            "- Hybrid retrieval: Dense (semantic) + BM25 (lexical) + RRF fusion\n"
            "- Fallback vectorless (PageIndex-style) khi cau hoi ngoai pham vi\n"
            "- Sinh cau tra loi **kem citation** [S1], [S2]... truy ve dung nguon\n"
            "- Ho tro hoi bang tieng Anh hoac tieng Viet (cross-lingual)\n"
            "- Inspector tab: xem duoc *tai sao* he thong tra loi nhu vay"
        ),
        "footer": "Corpus: 3 tai lieu chinh sach (PDF) + 8 bai viet ielts.org",
    },
    {
        "title": "1. Thu thap du lieu",
        "kicker": "Data collection -- task1 + task2",
        "body": (
            "**task1_collect_legal_docs.py** tai 3 tai lieu chinh sach IELTS "
            "(PDF) tu ielts.org: band descriptors, key assessment criteria, "
            "academic writing sample tasks.\n\n"
            "**task2_crawl_news.py** crawl 8 bai viet huong dan/blog ve IELTS "
            "Writing, luu JSON co metadata (title, url, ngay dang).\n\n"
            "Ca hai script deu **idempotent** -- chay lai se bo qua file da "
            "ton tai, khong tai trung."
        ),
        "footer": "Output: data/landing/legal/*.pdf, data/landing/news/*.json",
    },
    {
        "title": "2. Chuan hoa noi dung",
        "kicker": "Convert to Markdown -- task3",
        "body": (
            "**task3_convert_markdown.py** dung `markitdown` de chuyen PDF va "
            "JSON thanh Markdown thuan, giu heading structure de chunking o "
            "buoc sau bam theo section.\n\n"
            "Tat ca tai lieu sau buoc nay nam trong `data/standardized/` va "
            "co cung dinh dang, du nguon goc khac nhau (PDF vs web crawl)."
        ),
        "footer": "Output: data/standardized/{legal,news}/*.md",
    },
    {
        "title": "3. Chunking & Indexing",
        "kicker": "task4_chunking_indexing.py",
        "body": (
            "Hai chien luoc chunk khac nhau tuy loai tai lieu:\n\n"
            "- **Band descriptor PDF** (bang 4 cot bi markitdown lam tron): "
            "tach rieng theo tung muc Band (9 -> 4) thay vi cat mu 500 ky tu, "
            "de khong xe doi mot hang du lieu.\n"
            "- **Van ban thuong**: `MarkdownHeaderTextSplitter` (giu "
            "`section_path`) + `RecursiveCharacterTextSplitter`.\n\n"
            "Chunk duoc embed bang `text-embedding-3-small` (OpenAI) va "
            "upsert vao **ChromaDB**."
        ),
        "footer": "Output: ChromaDB collection (vector + metadata)",
    },
    {
        "title": "3.5. Xử lý câu hỏi (query processing)",
        "kicker": "Decompose + Expand + Reformulate -- src/query_processing.py",
        "body": (
            "Trước khi retrieve, một LLM call phụ (opt-in, bật/tắt được ở "
            "sidebar) sẽ xử lý câu hỏi người dùng theo 3 hướng cùng lúc:\n\n"
            "- **Reformulation**: viết lại câu hỏi cho standalone, sửa "
            "chính tả, dịch sang tiếng Anh nếu cần (khớp corpus tiếng Anh)\n"
            "- **Expansion**: thêm tối đa 5 từ khóa đồng nghĩa/liên quan để "
            "mở rộng phạm vi tìm\n"
            "- **Decomposition**: nếu câu hỏi là **câu kép** (nhiều ý trong "
            "một câu), tách thành 2-4 sub-query độc lập, mỗi sub-query "
            "retrieve riêng qua Task 9 rồi **gộp kết quả theo score cao "
            "nhất**\n\n"
            "Câu hỏi gốc vẫn được giữ nguyên khi đưa cho LLM sinh câu trả "
            "lời -- chỉ *query dùng để retrieve* bị biến đổi."
        ),
        "footer": "Xem chi tiết trong tab Inspector (mục 0) hoặc expander trên mỗi câu trả lời ở tab Chat",
    },
    {
        "title": "4. Semantic Search",
        "kicker": "Dense retrieval -- task5_semantic_search.py",
        "body": (
            "Embed cau hoi bang cung ham embedding voi Task 4, query "
            "ChromaDB bang cosine similarity.\n\n"
            "Vi dung chung embedding space, cau hoi **tieng Viet** van truy "
            "hoi dung noi dung **tieng Anh** trong corpus (cross-lingual "
            "retrieval) ma khong can model `-large`.\n\n"
            "Output chuan hoa theo schema `SearchResult`, sap xep giam dan "
            "theo score, gioi han `top_k`."
        ),
        "footer": "Score: cosine similarity trong khoang 0-1",
    },
    {
        "title": "5. Lexical Search",
        "kicker": "BM25 -- task6_lexical_search.py",
        "body": (
            "BM25 chay tren cung corpus chunk (nguon su that van la "
            "ChromaDB), manh o **tu khoa chinh xac** va **ten rieng** ma "
            "dense co the bo lo.\n\n"
            "Tokenizer dung regex Unicode `\\w+` (khong phai ASCII) kem ban "
            "**bo dau**, de chiu duoc cau hoi tieng Viet co dau lan khong "
            "dau, du corpus hien tai toan tieng Anh."
        ),
        "footer": "Score: BM25 (khong bi chan trong [0,1], khong so sanh truc tiep voi cosine)",
    },
    {
        "title": "6. Hop nhat ket qua",
        "kicker": "Reciprocal Rank Fusion -- task7_reranking.py",
        "body": (
            "RRF gop nhieu bang xep hang (dense + BM25) ma **khong** cong "
            "truc tiep hai loai score khac thang do voi nhau:\n\n"
            "> RRF(d) = sum( 1 / (k + rank) ) qua cac danh sach\n\n"
            "Chunk duoc ca hai nguon danh gia cao se **nhay hang**, dieu ma "
            "chi dung mot minh dense hoac BM25 se bo lo. RRF chi dung de xep "
            "hang, khong dung de quyet dinh fallback."
        ),
        "footer": "Xem truc tiep trong tab Inspector: bump chart Dense -> BM25 -> Fused",
    },
    {
        "title": "7. Fallback khi ngoai pham vi",
        "kicker": "Vectorless retriever -- task8_pageindex_vectorless.py",
        "body": (
            "Neu **best dense cosine score < threshold** (hieu chinh = "
            "0.58), he thong nghi cau hoi ngoai pham vi corpus va chuyen "
            "sang **PageIndex-style fallback**:\n\n"
            "- Parse heading cua `data/standardized/**/*.md` thanh cay muc "
            "luc\n"
            "- Gui outline (khong phai full text) cho LLM, hoi \"chon node "
            "lien quan nhat\"\n"
            "- Neu fallback cung khong tim thay gi -> ha ve hybrid result "
            "(degraded), **khong crash UI**"
        ),
        "footer": "Threshold hieu chinh tren 16 cau in-domain + 10 cau out-of-domain that",
    },
    {
        "title": "8. Pipeline retrieval hoan chinh",
        "kicker": "task9_retrieval_pipeline.py",
        "body": (
            "1. Chay song song `semantic_search` + `lexical_search`\n"
            "2. Fuse bang RRF (mot lan duy nhat)\n"
            "3. Lay best cosine score **goc** tu dense de xet fallback\n"
            "4. Neu duoi threshold, thu PageIndex fallback\n"
            "5. Neu fallback loi/rong -> tra hybrid ket qua thay vi crash\n\n"
            "Day la diem hoi tu duy nhat noi 4 buoc retrieval truoc do ghep "
            "lai thanh mot quyet dinh."
        ),
        "footer": "Khong bao gio raise exception ra ngoai UI",
    },
    {
        "title": "9. Sinh cau tra loi co trich dan",
        "kicker": "task10_generation.py",
        "body": (
            "1. Retrieve top-k chunk tu Task 9\n"
            "2. **Reorder** chunk (dau/cuoi uu tien) de giam \"lost-in-the-"
            "middle\"\n"
            "3. Format context kem title + source\n"
            "4. Goi LLM provider cau hinh trong `.env`\n"
            "5. Tra ve `answer` + `sources` + `retrieval_source`, danh so "
            "`[S1]`, `[S2]`... tro ve dung chunk\n\n"
            "Neu context khong du hoac provider loi -> **safe refusal**, "
            "khong bao gio bia thong tin."
        ),
        "footer": "Moi cau tra loi phai truy nguoc duoc ve nguon that",
    },
    {
        "title": "Giao dien san pham",
        "kicker": "3 tab lam viec",
        "body": (
            "**Chat** -- hoi dap chinh, hien cau tra loi kem citation, "
            "source card co the mo rong va nut mo link goc.\n\n"
            "**Inspector** -- \"mo hop den\" cho tung cau da hoi: so sanh "
            "Dense vs BM25, bump chart RRF, before/after reorder, quyet dinh "
            "fallback (co slider what-if threshold), latency waterfall, raw "
            "trace JSON tai duoc.\n\n"
            "**Evaluation** -- bao cao A/B dense-only vs hybrid+RRF tren "
            "golden dataset, 4 metric danh gia."
        ),
        "footer": "Moi cau hoi trong tab Chat deu co the bam \"Inspect this answer\" de xem chi tiet",
    },
]


def render_demo() -> None:
    if "demo_slide" not in st.session_state:
        st.session_state.demo_slide = 0

    total = len(SLIDES)
    idx = st.session_state.demo_slide
    idx = max(0, min(idx, total - 1))
    slide = SLIDES[idx]

    nav_prev, nav_dots, nav_next = st.columns([1, 6, 1])
    with nav_prev:
        if st.button("⬅ Trước", disabled=idx == 0, use_container_width=True):
            st.session_state.demo_slide = idx - 1
            st.rerun()
    with nav_dots:
        st.markdown(
            f"<div style='text-align:center; color: var(--text-color-secondary, #888); "
            f"padding-top: 0.4rem;'>Slide {idx + 1} / {total}</div>",
            unsafe_allow_html=True,
        )
    with nav_next:
        if st.button("Tiếp ➡", disabled=idx == total - 1, use_container_width=True):
            st.session_state.demo_slide = idx + 1
            st.rerun()

    st.progress((idx + 1) / total)

    with st.container(border=True):
        st.caption(slide["kicker"].upper())
        st.header(slide["title"])
        st.markdown(slide["body"])
        if slide.get("footer"):
            st.divider()
            st.caption(f"📌 {slide['footer']}")

    with st.expander("Xem tất cả slide (mục lục nhanh)"):
        for i, s in enumerate(SLIDES):
            marker = "➡️" if i == idx else f"{i + 1}."
            if st.button(f"{marker} {s['title']}", key=f"jump-{i}", use_container_width=True):
                st.session_state.demo_slide = i
                st.rerun()
