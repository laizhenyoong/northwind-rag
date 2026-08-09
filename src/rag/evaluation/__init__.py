"""Load, trace, and score the Northwind RAG evaluation set."""

from rag.evaluation.questions import GoldQuestion, load_gold_questions
from rag.evaluation.answer_metrics import (
    AnswerEvaluation,
    AnswerMetrics,
    AnswerSummary,
    evaluate_answers,
    score_answer,
)
from rag.evaluation.retrieval_metrics import RetrievalMetrics, score_retrieval
from rag.evaluation.traces import RetrievedChunk, RunTrace, load_traces, write_traces

__all__ = [
    "GoldQuestion",
    "AnswerEvaluation",
    "AnswerMetrics",
    "AnswerSummary",
    "RetrievedChunk",
    "RetrievalMetrics",
    "RunTrace",
    "load_gold_questions",
    "load_traces",
    "evaluate_answers",
    "score_retrieval",
    "score_answer",
    "write_traces",
]
