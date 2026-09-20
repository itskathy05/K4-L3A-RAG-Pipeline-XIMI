"""Session state khoi tao va tien ich lien quan. Khong chua logic pipeline."""

from __future__ import annotations

import streamlit as st


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list[{"role","content","result","trace"}]
    if "inspect_index" not in st.session_state:
        st.session_state.inspect_index = None


def push_turn(role: str, content: str, *, result: dict | None = None, trace: dict | None = None) -> None:
    st.session_state.messages.append(
        {"role": role, "content": content, "result": result, "trace": trace}
    )
    if role == "assistant" and trace is not None:
        st.session_state.inspect_index = len(st.session_state.messages) - 1


def assistant_turns() -> list[dict]:
    return [m for m in st.session_state.messages if m["role"] == "assistant" and m.get("trace")]
