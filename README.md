# Northwind RAG

A measurable Retrieval-Augmented Generation learning project using the
Northwind Hydraulics document corpus.

## Layout

```text
data/                     Source Markdown documents with YAML metadata
eval/                     Gold questions and retrieval ground truth
src/rag/
  ingestion/              Load and normalize source documents
tests/                    Tests that mirror the source-package structure
results/                  Generated evaluation runs (ignored by Git)
```

The pipeline will grow in the same order as `PLAN.md`:

1. `ingestion` loads documents and metadata.
2. `chunking` will create searchable chunks.
3. `embeddings` converts chunks and queries into vectors through Ollama.
4. `retrieval` will find relevant chunks.
5. `generation` will answer only from retrieved context.
6. `evaluation` will score retrieval and answers separately.

## Run tests

```zsh
source .venv/bin/activate
python -m pytest
```
