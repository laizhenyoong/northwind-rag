import json

from rag.evaluation import RetrievedChunk, RunTrace, write_traces


def test_write_traces_creates_one_json_record_per_question(tmp_path) -> None:
    output_path = tmp_path / "runs" / "baseline.jsonl"
    trace = RunTrace(
        question_id="Q001",
        question="What is the domestic per diem rate?",
        pipeline_config={"chunk_size": 500, "top_k": 5},
        retrieved_chunks=(
            RetrievedChunk(
                chunk_id="policies/travel-expense-v2.1.md:0003",
                source_path="data/policies/travel-expense-v2.1.md",
                rank=1,
                score=0.87,
            ),
        ),
        answer="RM 180 per day.",
        timing_ms=420,
    )

    write_traces(output_path, [trace])

    records = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert records == [
        {
            "question_id": "Q001",
            "question": "What is the domestic per diem rate?",
            "pipeline_config": {"chunk_size": 500, "top_k": 5},
            "retrieved_chunks": [
                {
                    "chunk_id": "policies/travel-expense-v2.1.md:0003",
                    "source_path": "data/policies/travel-expense-v2.1.md",
                    "rank": 1,
                    "score": 0.87,
                },
            ],
            "context_sent_to_model": None,
            "answer": "RM 180 per day.",
            "timing_ms": 420,
            "error": None,
        },
    ]
