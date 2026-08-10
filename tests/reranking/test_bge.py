from rag.reranking.bge import BGEReranker
from rag.retrieval import RetrievedPassage


class FakeCrossEncoder:
    def __init__(self) -> None:
        self.calls = []

    def predict(self, sentences, *, batch_size, show_progress_bar):
        self.calls.append((sentences, batch_size, show_progress_bar))
        return [0.2, 0.9]


def passage(chunk_id: str) -> RetrievedPassage:
    return RetrievedPassage(
        chunk_id=chunk_id,
        text=f"Text for {chunk_id}",
        score=0.1,
        metadata={"source_path": f"data/{chunk_id}.md"},
    )


def test_bge_reranker_scores_question_passage_pairs(monkeypatch) -> None:
    model = FakeCrossEncoder()
    monkeypatch.setattr("rag.reranking.bge._load_cross_encoder", lambda *_: model)
    reranker = BGEReranker(batch_size=4)

    passages = reranker.rerank("Which policy applies?", [passage("first"), passage("second")], top_k=1)

    assert [passage.chunk_id for passage in passages] == ["second"]
    assert passages[0].score == 0.9
    assert model.calls == [
        (
            [
                ("Which policy applies?", "Text for first"),
                ("Which policy applies?", "Text for second"),
            ],
            4,
            False,
        )
    ]
