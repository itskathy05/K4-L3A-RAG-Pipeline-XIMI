import streamlit as st
from dotenv import load_dotenv

from src.observability import answer_with_trace
from ui.state import init_state, push_turn
from ui.tab_chat import render_chat_history
from ui.tab_demo import render_demo
from ui.tab_evaluation import render_evaluation
from ui.tab_inspector import render_inspector

load_dotenv()

st.set_page_config(
    page_title="IELTS Writing RAG Chatbot",
    page_icon="📝",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
    [data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        background: var(--secondary-background-color);
        border-radius: 10px;
        padding: 0.6rem 0.9rem;
        border: 1px solid rgba(128,128,128,0.18);
    }
    div[data-baseweb="tab-list"] { gap: 0.4rem; }
    button[data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    div[data-testid="stChatMessage"] { border-radius: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)

init_state()

with st.sidebar:
    st.title("📝 IELTS Writing RAG")
    st.caption(
        "Chatbot tra loi cau hoi ve IELTS Writing (band descriptors, "
        "assessment criteria, test format) dua tren hybrid retrieval "
        "(dense + BM25 + RRF) voi fallback vectorless va citation."
    )
    top_k = st.slider("So chunks (top_k)", 3, 10, 5)
    expand_query = st.toggle(
        "🧠 Query processing (decompose/expand/reformulate)",
        value=True,
        help=(
            "Bat 1 LLM call phu de reformulate cau hoi, mo rong tu khoa, va "
            "tach cau hoi kep thanh nhieu sub-query truoc khi retrieve. "
            "Tang do chinh xac retrieval nhung them latency/cost."
        ),
    )
    st.divider()
    st.markdown("**Corpus**")
    st.caption("3 tai lieu chinh sach IELTS + 8 bai viet ielts.org")
    st.markdown("**Ngon ngu**")
    st.caption("Ho tro hoi bang tieng Anh hoac tieng Viet (cross-lingual).")
    st.divider()
    st.caption("💡 Xem tab **Giới thiệu** để hiểu pipeline hoạt động ra sao trước khi hỏi.")

tab_demo, tab_chat, tab_inspector, tab_eval = st.tabs(
    ["🎬 Giới thiệu", "💬 Chat", "🔍 Inspector", "📊 Evaluation"]
)

with tab_demo:
    render_demo()

with tab_chat:
    st.title("IELTS Writing RAG Chatbot")
    render_chat_history()

    query = st.chat_input("Nhap cau hoi (tieng Anh hoac tieng Viet)...")

    if query:
        push_turn("user", query)
        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("Retrieving and generating..."):
            result, trace = answer_with_trace(query, top_k=top_k, expand_query=expand_query)

        push_turn("assistant", result["answer"], result=result, trace=trace)
        st.session_state.messages[-1]["query"] = query
        st.rerun()

with tab_inspector:
    st.title("Retrieval Inspector")
    st.caption("Xem tung buoc pipeline da chay cho mot cau hoi da hoi trong tab Chat.")
    render_inspector()

with tab_eval:
    st.title("Evaluation: dense-only vs hybrid+RRF")
    render_evaluation()
