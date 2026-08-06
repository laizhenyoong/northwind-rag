"""Load and score the Northwind RAG evaluation set."""

from rag.evaluation.questions import GoldQuestion, load_gold_questions

__all__ = ["GoldQuestion", "load_gold_questions"]
