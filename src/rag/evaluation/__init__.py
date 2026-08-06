"""Load, trace, and score the Northwind RAG evaluation set."""

from rag.evaluation.questions import GoldQuestion, load_gold_questions
from rag.evaluation.retrieval_metrics import RetrievalMetrics, score_retrieval
from rag.evaluation.traces import RetrievedChunk, RunTrace, write_traces

__all__ = [
    "GoldQuestion",
    "RetrievedChunk",
    "RetrievalMetrics",
    "RunTrace",
    "load_gold_questions",
    "score_retrieval",
    "write_traces",
]
