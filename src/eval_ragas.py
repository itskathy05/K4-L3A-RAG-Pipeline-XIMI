"""
File DUY NHAT import `ragas`. Neu ban 0.4.3 doi API o phien ban sau, chi
file nay can sua.

API thuc te cua ragas==0.4.3 (da xac minh bang cach cai va inspect truc
tiep, KHAC HOAN TOAN cac huong dan RAGAS cu tren mang dung
SingleTurnSample/EvaluationDataset/ragas.evaluate()):

    from ragas.metrics.collections import Faithfulness, AnswerRelevancy, \
        ContextRecall, ContextPrecision
    from ragas.llms import llm_factory

    judge = llm_factory("gpt-4o-mini", client=AsyncOpenAI(...))  # PHAI la
    # AsyncOpenAI -- .score() dong bo goi noi bo agenerate(), client dong bo
    # se raise "Cannot use agenerate() with a synchronous client."

    Faithfulness(llm=judge).score(user_input=q, response=answer, retrieved_contexts=ctxs)
    AnswerRelevancy(llm=judge, embeddings=emb).score(user_input=q, response=answer)
    ContextRecall(llm=judge).score(user_input=q, retrieved_contexts=ctxs, reference=expected)
    ContextPrecision(llm=judge).score(user_input=q, reference=expected, retrieved_contexts=ctxs)

Moi .score() tra ve MetricResult; doc .value de lay float.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

_JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o-mini")
_JUDGE_EMBEDDING_MODEL = os.getenv("RAGAS_JUDGE_EMBEDDING_MODEL", "text-embedding-3-small")


def build_judge():
    """Tra ve (faithfulness, answer_relevancy, context_recall, context_precision)."""
    from openai import AsyncOpenAI
    from ragas.embeddings.base import modern_embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    llm = llm_factory(_JUDGE_MODEL, client=client)
    embeddings = modern_embedding_factory("openai", model=_JUDGE_EMBEDDING_MODEL, client=client)

    return {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_recall": ContextRecall(llm=llm),
        "context_precision": ContextPrecision(llm=llm),
    }


def score_one(judge: dict, record: dict) -> dict[str, float | None]:
    """Cham 4 metric cho mot record. Loi tung metric rieng le -> None, khong
    lam sap ca record (mot cau hoi loi khong duoc pha vong lap eval)."""
    scores: dict[str, float | None] = {}
    user_input = record["question"]
    response = record["answer"]
    retrieved_contexts = record["contexts"] or [""]
    reference = record["expected_answer"]

    metric_calls = {
        "faithfulness": lambda: judge["faithfulness"].score(
            user_input=user_input, response=response, retrieved_contexts=retrieved_contexts
        ),
        "answer_relevancy": lambda: judge["answer_relevancy"].score(
            user_input=user_input, response=response
        ),
        "context_recall": lambda: judge["context_recall"].score(
            user_input=user_input, retrieved_contexts=retrieved_contexts, reference=reference
        ),
        "context_precision": lambda: judge["context_precision"].score(
            user_input=user_input, reference=reference, retrieved_contexts=retrieved_contexts
        ),
    }

    for name, call in metric_calls.items():
        try:
            result = call()
            scores[name] = float(result.value)
        except Exception as error:  # loi judge/API -> None, khong crash eval run
            print(f"    metric {name} failed for {record.get('id', '?')}: {error}")
            scores[name] = None

    return scores
