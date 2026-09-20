"""
Observability layer for the RAG pipeline.

Không thêm tham số vào bất kỳ hàm đã bị `tests/test_contracts.py::
test_public_function_signatures_are_stable` pin chữ ký (retrieve,
generate_with_citation, semantic_search, lexical_search, rerank_rrf,
pageindex_search, chunk_documents, load_documents). Thay vào đó các hàm thật
ghi "event" thuần vào một collector theo ContextVar; khi không có trace nào
đang mở, mọi lời gọi record()/stage() là no-op gần như miễn phí.

Dùng ContextVar (không phải biến module) vì Streamlit chạy mỗi session trong
một ScriptRunner thread riêng — biến module sẽ làm trace của user A lẫn vào
user B. contextvars xuyên qua thread mới nếu được copy_context() đúng cách,
nhưng ở đây ta chỉ cần nó không rò rỉ giữa các lần gọi tuần tự trong cùng
tiến trình, điều mà module-global không đảm bảo dưới Streamlit rerun.

Không import bất kỳ thứ gì từ src.task* để tránh vòng lặp import.
"""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator


class Trace:
    """Container thu thập event cho một lần chạy pipeline."""

    def __init__(self, label: str = "") -> None:
        self.trace_id = uuid.uuid4().hex[:12]
        self.label = label
        self.started_at = time.time()
        self.events: list[dict[str, Any]] = []
        self._stage_stack: list[tuple[str, float]] = []
        self.error: str | None = None

    def record(self, stage: str, **payload: Any) -> None:
        self.events.append({"stage": stage, "t": time.time() - self.started_at, **payload})

    def get(self, stage: str, default: Any = None) -> Any:
        """Trả event cuối cùng ghi cho `stage`, hoặc default."""
        for event in reversed(self.events):
            if event["stage"] == stage:
                return event
        return default

    def get_all(self, stage: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["stage"] == stage]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "label": self.label,
            "total_latency_ms": (time.time() - self.started_at) * 1000,
            "error": self.error,
            "events": self.events,
        }


_CURRENT: ContextVar["Trace | None"] = ContextVar("rag_trace", default=None)


def is_tracing() -> bool:
    return _CURRENT.get() is not None


def current() -> Trace | None:
    return _CURRENT.get()


def record(stage: str, **payload: Any) -> None:
    """Ghi một event vào trace đang mở. No-op nếu không có trace nào."""
    trace = _CURRENT.get()
    if trace is not None:
        trace.record(stage, **payload)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Đo latency của một khối code và ghi vào trace. Luôn re-raise lỗi."""
    start = time.perf_counter()
    trace = _CURRENT.get()
    try:
        yield
    except Exception as exc:
        if trace is not None:
            trace.record(f"{name}.error", error=repr(exc))
        raise
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        if trace is not None:
            trace.record(f"{name}.latency", latency_ms=elapsed_ms)


@contextmanager
def trace_run(label: str = "") -> Iterator[Trace]:
    """Mở một trace mới cho một lần chạy pipeline (một câu hỏi)."""
    trace = Trace(label=label)
    token = _CURRENT.set(trace)
    try:
        yield trace
    except Exception as exc:
        trace.error = repr(exc)
        raise
    finally:
        _CURRENT.reset(token)
