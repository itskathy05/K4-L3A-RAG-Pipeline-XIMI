"""Tab Chat: dung rubric bat buoc -- hien answer, sources, retrieval method,
score. Pipeline chi duoc goi mot lan trong block `if query:` cua app.py;
tab nay chi render tu st.session_state.messages."""

from __future__ import annotations

import streamlit as st

from ui.components import (
    query_processing_card,
    refusal_banner,
    render_answer_with_citations,
    source_card,
    status_metrics_row,
)
from ui.state import assistant_turns


def render_chat_history() -> None:
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
                continue

            result = message.get("result")
            trace = message.get("trace")
            if result is None:
                st.markdown(message["content"])
                continue

            query_processing_card(trace)

            if result.get("retrieval_source") == "none" or not result.get("sources"):
                refusal_banner(result, trace)
            else:
                render_answer_with_citations(result["answer"], result["sources"])
                status_metrics_row(result, trace)
                for idx, source in enumerate(result["sources"], 1):
                    source_card(idx, source, query=message.get("query", ""), key_prefix=f"{i}-")

            turns = assistant_turns()
            if turns and message is turns[-1]:
                if st.button("Inspect this answer", key=f"inspect-{i}"):
                    st.session_state.inspect_index = i
                    st.info("Switch to the Inspector tab to see retrieval details.")
