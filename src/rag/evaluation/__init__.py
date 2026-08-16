"""Load, trace, and score the Northwind RAG evaluation set."""

from rag.evaluation.questions import GoldQuestion, load_gold_questions
from rag.evaluation.answer_metrics import (
    AnswerEvaluation,
    AnswerMetrics,
    AnswerSummary,
    evaluate_answers,
    score_answer,
)
from rag.evaluation.answer_judge import (
    AnswerJudgement,
    AnswerJudgementEvaluation,
    AnswerJudgementSummary,
    SemanticAnswerJudge,
    evaluate_semantic_answers,
    write_judgements,
)
from rag.evaluation.judge_agreement import (
    AgreementReport,
    LabelAgreement,
    compare_judgements,
    disagreements,
    load_judgements,
)
from rag.evaluation.retrieval_metrics import RetrievalMetrics, score_retrieval
from rag.evaluation.traces import RetrievedChunk, RunTrace, load_traces, write_traces

__all__ = [
    "AgreementReport",
    "GoldQuestion",
    "LabelAgreement",
    "AnswerEvaluation",
    "AnswerMetrics",
    "AnswerSummary",
    "AnswerJudgement",
    "AnswerJudgementEvaluation",
    "AnswerJudgementSummary",
    "RetrievedChunk",
    "RetrievalMetrics",
    "RunTrace",
    "compare_judgements",
    "disagreements",
    "load_gold_questions",
    "load_judgements",
    "load_traces",
    "evaluate_answers",
    "evaluate_semantic_answers",
    "score_retrieval",
    "score_answer",
    "SemanticAnswerJudge",
    "write_judgements",
    "write_traces",
]
