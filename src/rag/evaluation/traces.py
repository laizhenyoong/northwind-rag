"""Write inspectable, one-record-per-question evaluation traces."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """A chunk returned by retrieval for one question."""

    chunk_id: str
    source_path: str
    rank: int
    score: float


@dataclass(frozen=True, slots=True)
class RunTrace:
    """Everything needed to inspect one pipeline result."""

    question_id: str
    question: str
    pipeline_config: dict[str, Any]
    retrieved_chunks: tuple[RetrievedChunk, ...]
    context_sent_to_model: str | None = None
    answer: str | None = None
    timing_ms: int | None = None
    error: str | None = None


def write_traces(path: Path, traces: list[RunTrace]) -> None:
    """Write one JSON object per question, creating the output directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized_traces = (json.dumps(asdict(trace), ensure_ascii=False) for trace in traces)
    path.write_text("\n".join(serialized_traces) + "\n", encoding="utf-8")


def load_traces(path: Path) -> list[RunTrace]:
    """Load a JSONL trace file written by ``write_traces`` for offline analysis."""
    traces = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            traces.append(
                RunTrace(
                    question_id=record["question_id"],
                    question=record["question"],
                    pipeline_config=record["pipeline_config"],
                    retrieved_chunks=tuple(
                        RetrievedChunk(**chunk) for chunk in record["retrieved_chunks"]
                    ),
                    context_sent_to_model=record.get("context_sent_to_model"),
                    answer=record.get("answer"),
                    timing_ms=record.get("timing_ms"),
                    error=record.get("error"),
                )
            )
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"Invalid trace on line {line_number} of {path}") from error
    return traces
