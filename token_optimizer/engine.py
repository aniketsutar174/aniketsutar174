from dataclasses import dataclass
from typing import Optional

from config import cfg
from token_optimizer.retrieval.retriever import retrieve
from token_optimizer.context_builder import build_context
from token_optimizer.router import route
from token_optimizer.core.llm import complete
from token_optimizer.core.db import total_repo_tokens


@dataclass
class TokenReport:
    baseline_tokens: int   # total tokens across all indexed chunks (full-repo proxy)
    context_tokens: int    # tokens actually sent as context
    input_tokens: int      # billed input tokens (from Anthropic usage)
    output_tokens: int     # billed output tokens
    savings_pct: float     # percentage of baseline tokens saved
    model_used: str


@dataclass
class OptimizeResult:
    task_type: str
    output: str
    token_report: TokenReport


def optimize(
    task_type: str,
    repo: str,
    user_input: str,
    *,
    override_model: Optional[str] = None,
) -> OptimizeResult:
    """
    Full RAG pipeline:
      1. Retrieve candidate chunks via Voyage embeddings + pgvector
      2. Route to fast/smart model based on retrieval confidence
      3. Build token-budgeted context
      4. Call Claude, return result + token report
    """
    budget = cfg.budgets[task_type]

    # 1. Retrieve
    chunks = retrieve(repo, user_input, k=cfg.candidate_chunks)
    top_chunks = chunks[: cfg.final_chunks]
    top_similarity = float(top_chunks[0]["similarity"]) if top_chunks else 0.0

    # 2. Route
    model = override_model or route(task_type, top_similarity)

    # 3. Build context
    system, user_message, context_tokens = build_context(
        top_chunks, task_type, user_input, budget
    )

    # 4. Baseline: sum of all token_counts in the indexed repo
    baseline_tokens = total_repo_tokens(cfg.db_url, repo)

    # 5. LLM call
    content, input_tokens, output_tokens = complete(
        system=system,
        user=user_message,
        model=model,
        max_tokens=budget.max_output_tokens,
    )

    savings_pct = (
        max(0.0, (baseline_tokens - context_tokens) / baseline_tokens * 100)
        if baseline_tokens > 0
        else 0.0
    )

    report = TokenReport(
        baseline_tokens=baseline_tokens,
        context_tokens=context_tokens,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        savings_pct=savings_pct,
        model_used=model,
    )

    return OptimizeResult(task_type=task_type, output=content, token_report=report)
