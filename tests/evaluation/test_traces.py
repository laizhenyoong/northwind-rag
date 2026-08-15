import json

from rag.evaluation import RetrievedChunk, RunTrace, load_traces, write_traces


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
        queries_used=("What is the domestic per diem rate?",),
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
            "queries_used": ["What is the domestic per diem rate?"],
            "context_sent_to_model": None,
            "answer": "RM 180 per day.",
            "timing_ms": 420,
            "generation_usage": None,
            "error": None,
        },
    ]


def test_load_traces_reads_previously_written_jsonl(tmp_path) -> None:
    trace_path = tmp_path / "traces.jsonl"
    original = RunTrace(
        question_id="Q001",
        question="What is the rate?",
        pipeline_config={"top_k": 5},
        retrieved_chunks=(
            RetrievedChunk("data/policies/travel.md:0000", "data/policies/travel.md", 1, 0.9),
        ),
    )
    write_traces(trace_path, [original])

    assert load_traces(trace_path) == [original]
