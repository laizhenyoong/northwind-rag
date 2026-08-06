"""Read the gold questions used to evaluate the RAG pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GoldQuestion:
    """One question with known retrieval and answer ground truth."""

    id: str
    question: str
    expected_answer: str
    source_files: tuple[str, ...]
    concept: str
    difficulty: str
    answerable: bool
    distractor_files: tuple[str, ...] = ()
    note: str | None = None


def load_gold_questions(path: Path) -> list[GoldQuestion]:
    """Load one JSON object per line from the supplied evaluation file."""
    questions = []

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}") from error

        questions.append(
            GoldQuestion(
                id=record["id"],
                question=record["question"],
                expected_answer=record["expected_answer"],
                source_files=tuple(record["source_files"]),
                concept=record["concept"],
                difficulty=record["difficulty"],
                answerable=record["answerable"],
                distractor_files=tuple(record.get("distractor_files", [])),
                note=record.get("note"),
            )
        )

    return questions
