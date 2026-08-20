"""Answer the gold questions with a tool-calling agent instead of a fixed pipeline.

The pipeline runners retrieve once and generate once. This runner hands the model
three tools and lets it decide how many times to search, whether to widen a chunk
into its whole document, and whether to check a document's version history. The
traces it writes use the same schema, so compare_runs can put the two side by side.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rag.evaluation.answer_judge import SemanticAnswerJudge, evaluate_semantic_answers, write_judgements
from rag.evaluation.answer_metrics import evaluate_answers
from rag.evaluation.questions import GoldQuestion, load_gold_questions
from rag.evaluation.traces import RetrievedChunk, RunTrace, write_traces
from rag.generation import DeepSeekChatModel, OllamaChatModel


# search_documents prints "[1] chunk_id: <id>", so the label precedes the field.
CHUNK_ID_PATTERN = re.compile(r"chunk_id: (\S+)")
DEFAULT_AGENT_PATH = Path("src/agent/app/NorthwindRagAgent")


def load_agent_tools(agent_path: Path) -> tuple[list[Any], str]:
    """Import the deployed agent's tools and system prompt from its own package."""
    resolved = str(agent_path.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
    from rag_tools import NORTHWIND_TOOLS  # noqa: PLC0415 - path set above
    import main as agent_main  # noqa: PLC0415

    return list(NORTHWIND_TOOLS), agent_main.DEFAULT_SYSTEM_PROMPT


def build_agent(tools: Sequence[Any], system_prompt: str, model_id: str) -> Any:
    """Create one fresh agent, so no question can see another's history."""
    from strands import Agent
    from strands.models.bedrock import BedrockModel

    return Agent(
        model=BedrockModel(model_id=model_id),
        system_prompt=system_prompt,
        tools=list(tools),
        callback_handler=None,
    )


def _tool_calls_from(messages: Sequence[dict]) -> tuple[list[dict[str, Any]], str]:
    """Recover which tools the model chose, and the evidence they returned."""
    calls: list[dict[str, Any]] = []
    evidence: list[str] = []
    pending: dict[str, dict[str, Any]] = {}
    for message in messages:
        for block in message.get("content", []) or []:
            if "toolUse" in block:
                use = block["toolUse"]
                record = {"name": use.get("name"), "input": use.get("input"), "result_chars": 0}
                pending[use.get("toolUseId", "")] = record
                calls.append(record)
            elif "toolResult" in block:
                result = block["toolResult"]
                body = "".join(part.get("text", "") for part in result.get("content", []) or [])
                evidence.append(body)
                record = pending.get(result.get("toolUseId", ""))
                if record is not None:
                    record["result_chars"] = len(body)
                    record["status"] = result.get("status", "success")
                    record["chunk_ids"] = CHUNK_ID_PATTERN.findall(body)
    return calls, "\n\n".join(evidence)


def _retrieved_chunks_from(calls: Sequence[dict[str, Any]]) -> tuple[RetrievedChunk, ...]:
    """Flatten every searched chunk into one ranked list, first sighting wins.

    An agent can search several times, so a chunk's rank here means "how early
    did this evidence reach the model", not "how similar was it to one query".
    """
    seen: dict[str, None] = {}
    for call in calls:
        for chunk_id in call.get("chunk_ids", ()) or ():
            seen.setdefault(chunk_id, None)
    return tuple(
        RetrievedChunk(
            chunk_id=chunk_id,
            source_path=chunk_id.rsplit(":", 1)[0],
            rank=rank,
            score=0.0,
        )
        for rank, chunk_id in enumerate(seen, start=1)
    )


def answer_one(
    question: GoldQuestion,
    *,
    tools: Sequence[Any],
    system_prompt: str,
    model_id: str,
    pipeline_config: dict[str, Any],
    attempts: int = 3,
) -> RunTrace:
    """Run the agent loop for one question, keeping the evidence even on failure."""
    started_at = time.perf_counter()
    error: str | None = None
    answer: str | None = None
    calls: list[dict[str, Any]] = []
    context = ""
    usage: dict[str, int | None] | None = None

    for attempt in range(1, attempts + 1):
        agent = build_agent(tools, system_prompt, model_id)
        try:
            result = agent(question.question)
            answer = str(result)
            calls, context = _tool_calls_from(agent.messages)
            accumulated = getattr(getattr(result, "metrics", None), "accumulated_usage", None)
            if accumulated:
                usage = {
                    "prompt_tokens": accumulated.get("inputTokens"),
                    "completion_tokens": accumulated.get("outputTokens"),
                    "total_tokens": accumulated.get("totalTokens"),
                }
            error = None
            break
        except Exception as exception:  # Preserve a debuggable record for evaluation.
            error = f"{type(exception).__name__}: {exception}"
            calls, context = _tool_calls_from(agent.messages)
            if attempt < attempts:
                time.sleep(2 ** attempt)

    return RunTrace(
        question_id=question.id,
        question=question.question,
        pipeline_config=pipeline_config,
        retrieved_chunks=_retrieved_chunks_from(calls),
        queries_used=tuple(
            str(call["input"].get("query"))
            for call in calls
            if call.get("name") == "search_documents" and isinstance(call.get("input"), dict)
        ),
        context_sent_to_model=context or None,
        answer=answer,
        timing_ms=int((time.perf_counter() - started_at) * 1000),
        generation_usage=usage,
        error=error,
        tool_calls=tuple(calls),
    )


def main() -> None:
    """Run the agentic answer evaluation and judge it with the same judge as the control."""
    parser = argparse.ArgumentParser(description="Evaluate agentic RAG answers")
    parser.add_argument("--questions", type=Path, default=Path("eval/gold_questions.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results/agent-answers.jsonl"))
    parser.add_argument("--judgements-output", type=Path, default=Path("results/agent-judgements.jsonl"))
    parser.add_argument("--agent-path", type=Path, default=DEFAULT_AGENT_PATH)
    parser.add_argument("--model-id", default="global.anthropic.claude-sonnet-4-5-20250929-v1:0")
    parser.add_argument("--judge-provider", choices=("ollama", "deepseek"), default="deepseek")
    parser.add_argument("--judge-model")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-judge", action="store_true")
    arguments = parser.parse_args()

    questions = load_gold_questions(arguments.questions)
    if arguments.limit:
        questions = questions[: arguments.limit]

    tools, system_prompt = load_agent_tools(arguments.agent_path)
    pipeline_config = {
        "retrieval_strategy": "agentic-semantic-bm25-rrf",
        "embedding_model": "embeddinggemma",
        "generation_model": arguments.model_id,
        "chunking_strategy": "markdown-aware",
        "chunk_size": 500,
        "candidate_k": 20,
        "rrf_k": 60,
        "agent_tools": [tool.tool_name for tool in tools],
        "top_k": 5,
    }

    with ThreadPoolExecutor(max_workers=arguments.workers) as pool:
        traces = list(
            pool.map(
                lambda question: answer_one(
                    question,
                    tools=tools,
                    system_prompt=system_prompt,
                    model_id=arguments.model_id,
                    pipeline_config=pipeline_config,
                ),
                questions,
            )
        )

    write_traces(arguments.output, traces)
    evaluation = evaluate_answers(questions, traces)
    print(json.dumps({"answered": len(traces), "deterministic": str(evaluation)}, default=str)[:400])

    if arguments.skip_judge:
        return
    judge_model = (
        DeepSeekChatModel.from_environment(model=arguments.judge_model, timeout_seconds=600.0)
        if arguments.judge_provider == "deepseek"
        else OllamaChatModel(model=arguments.judge_model or "gemma4")
    )
    judged = evaluate_semantic_answers(questions, traces, judge=SemanticAnswerJudge(judge_model))
    write_judgements(arguments.judgements_output, judged.judgements)
    print(f"Judgements: {arguments.judgements_output}")


if __name__ == "__main__":
    main()
