# Northwind RAG

A measurable Retrieval-Augmented Generation learning project using the
Northwind Hydraulics document corpus.

## Layout

```text
data/                     Source Markdown documents with YAML metadata
eval/                     Gold questions and retrieval ground truth
src/rag/
  ingestion/              Load and normalize source documents
  chunking/               Split documents into retrievable chunks
  retrieval/              Find relevant chunks and widen them for answering
  query_transformation/   Rewrite a question into document-search queries
  generation/             Answer only from retrieved context
  evaluation/             Score retrieval and answers separately
tests/                    Tests that mirror the source-package structure
results/                  Generated evaluation runs (ignored by Git)
```

The pipeline is built in the same order as `PLAN.md`:

1. `ingestion` loads documents and metadata.
2. `chunking` creates searchable chunks.
3. `embeddings` converts chunks and queries into vectors through Ollama.
4. `retrieval` finds relevant chunks and widens them for answering.
5. `generation` answers only from retrieved context.
6. `evaluation` scores retrieval and answers separately.

## Run tests

```zsh
source .venv/bin/activate
python -m pytest
```

## Pinecone setup

The optional Pinecone index stores vectors created locally by Ollama. Copy the
safe template, then put the API key in the local Git-ignored file:

```zsh
cp .env.example .env
```

Set `PINECONE_API_KEY` in `.env`. The application loads this file when it
connects to Pinecone.

## Index the corpus

With Ollama running and the `embeddinggemma` model installed, create embeddings
for the baseline 500-character chunks and upload them to Pinecone:

```zsh
python -m rag.indexing
```

This can be re-run safely with the same settings: Pinecone upserts records by
their stable chunk ID. The next step is to query this index for the top five
semantic matches.

## Retrieve semantic matches

```zsh
python -m rag.retrieval.semantic "What is the domestic per diem rate for business travel?"
```

This baseline retrieves by semantic meaning only. It deliberately does not yet
filter out superseded documents; that is a later metadata-filtering experiment.

## Measure the baseline

Run every gold question and write one inspectable JSONL trace per question:

```zsh
python -m rag.evaluation.run_retrieval
```

The command reports document-level P@5, Recall@5, MRR, and nDCG@5. The trace
file is written to `results/retrieval-baseline.jsonl` and is ignored by Git.

## Inspect failures before changing the system

```zsh
python -m rag.evaluation.analyze_retrieval
```

This reads the saved trace file locally. It groups hit rates by question concept
and lists each question whose first relevant source was not rank 1.

## Compare exact-token retrieval

Run the BM25 keyword baseline over the same chunks and gold questions:

```zsh
python -m rag.evaluation.run_retrieval --strategy keyword
```

BM25 is an in-memory index. Unlike semantic embeddings, it preserves an exact
identifier such as `FIN-EXP-22` or `DNV-TA-4471-B` as a matching token.

## Combine semantic and keyword retrieval

Run both retrievers, fuse their ranked candidate lists with reciprocal-rank
fusion (RRF), and evaluate the result:

```zsh
python -m rag.evaluation.run_retrieval --strategy hybrid
```

RRF awards a chunk a small score based on its rank in each list. It avoids
comparing incompatible cosine and BM25 scores directly.

## Rerank the hybrid candidates

The optional local BGE cross-encoder reads each question and candidate chunk
together, then reorders the hybrid top 20 before we keep the final five:

```zsh
python -m rag.evaluation.run_retrieval --strategy hybrid-reranked
python -m rag.answer --rerank "What is the maximum flow for NH-8840-X?"
```

The first reranking run downloads `BAAI/bge-reranker-v2-m3` to the local
Hugging Face cache. It is not sent to Ollama or Pinecone. Compare its scores
with `--strategy hybrid`, especially Q020 and Q021.

## Decompose multi-hop questions

For a question whose answer is split across documents, let the local query
model create focused sub-queries. Each query retrieves a top-20 candidate set;
their rankings are fused before optional reranking.

```zsh
python -m rag.answer --decompose --rerank \
  "Who will supply the FKM seals for the NH-8840-X, and what lead time improvement is expected?"
```

The trace stores `queries_used`, so you can inspect exactly how the original
question was transformed before judging whether the result improved.

## Generate a grounded answer

With Ollama running and the locally installed `gemma4` model available, answer
from the hybrid top-five context and save an inspectable trace:

```zsh
python -m rag.answer "What is the domestic per diem rate for business travel?"
```

The model is instructed to use only the supplied chunks. Every factual claim
must cite a context label such as `[S1]`; unknown or missing labels are
rejected and saved as an error rather than presented as a grounded answer. A
model may refuse with `I don't know based on the provided context.` The trace
at `results/answer.jsonl` retains the question, retrieved chunks, exact context,
answer, timing, and any error.

## Evaluate generated answers

Run the same grounded pipeline over the gold questions and write one answer
trace per question:

```zsh
python -m rag.evaluation.run_answers
```

This reports deterministic evidence checks: whether answerable questions got a
cited answer, whether citations point to an expected source, and whether the
unanswerable questions received the exact refusal. These are grounding checks,
not yet a semantic judgement that different wordings have the same meaning.
Use `--limit 3` for a quick local smoke run.

## Judge answers by meaning

Add a semantic judge to decide whether an answer reaches the same conclusion as
the gold answer:

```zsh
python -m rag.evaluation.run_answers --judge-model gemma4
```

The judge returns one JSON verdict per question: correctness, whether the cited
labels support the claims, whether the right version was used, and whether an
unanswerable question was refused. Verdicts are written to
`results/answer-judgements.jsonl`. Pass `--skip-semantic-judge` to skip it.

## Widen retrieved chunks before answering

Search works best on small chunks and answering works best on whole passages.
Three strategies close that gap:

```zsh
python -m rag.evaluation.run_answers --neighbor-window 1
python -m rag.evaluation.run_answers --parent-documents --max-document-chars 8000
python -m rag.evaluation.run_answers --document-scoped --document-k 3
```

Neighbor expansion appends the chunks either side of each hit. Parent documents
replace the hit with the file it came from, leaving anything past
`--max-document-chars` as a chunk. Document scoping searches twice, the second
time restricted to the files the first pass chose.

## Generate with a hosted model

The same pipeline runs against DeepSeek instead of local Ollama:

```zsh
python -m rag.evaluation.run_answers --provider deepseek
```

Set `DEEPSEEK_API_KEY` in `.env`. Embeddings stay local, because the Pinecone
index was built with `embeddinggemma`.

## Re-judge saved traces

Grading is separate from generation, so a judge can be changed without
regenerating the answers:

```zsh
python -m rag.evaluation.rejudge \
  --traces results/answer-evaluation.jsonl \
  --output results/answer-judgements-deepseek.jsonl \
  --judge-provider deepseek
```

Pass `--compare-with` an existing judgement file to report how far two judges
agree per label, discounted for the agreement they would reach by chance.

## Compare two answer runs

The correctness rate alone cannot tell a change that did nothing from a change
that fixed and broke the same number of questions:

```zsh
python -m rag.evaluation.compare_runs \
  results/baseline-judgements.jsonl \
  results/variant-judgements.jsonl
```

For each label this names the questions the change fixed and broke, and reports
the chance the split came from noise alone.

## Index semantically chunked documents

The alternative chunker splits where the meaning between neighbouring sentences
shifts rather than at a character count, into its own Pinecone namespace:

```zsh
python -m rag.semantic_sync --dry-run
python -m rag.semantic_sync
python -m rag.evaluation.run_answers \
  --chunking-strategy semantic --namespace semantic-v1
```

BM25 and chunk widening read chunks from disk, so `--chunking-strategy` must
match the namespace being queried.

## Retrieve current or historical versions

Use explicit version constraints when the user asks for a current policy, a
specific revision, or the policy that applied on a past date:

```zsh
python -m rag.retrieval.semantic "What is the domestic per diem rate?" \
  --document-id POL-FIN-004 --current

python -m rag.retrieval.semantic "What was the domestic per diem rate?" \
  --document-id POL-FIN-004 --as-of 2025-06-01
```

The index stores each version's document family, current flag, and numeric
effective/expiry dates. This is an explicit application policy, not something
the embedding is asked to infer.

The same constraints now apply to hybrid retrieval, reranking, and grounded
answers, so all stages receive only the selected version:

```zsh
python -m rag.answer --rerank --document-id POL-FIN-004 --current \
  "What is the current domestic per diem rate?"
```

## Synchronize later document changes

Preview the minimum required index changes before applying them:

```zsh
python -m rag.sync --dry-run
```

Then apply the sync:

```zsh
python -m rag.sync
```

The generated `.rag/ingestion-manifest.json` records hashes and chunk IDs. It
lets the sync skip unchanged files, update metadata without re-embedding, and
delete chunks belonging to files removed from the source corpus.
