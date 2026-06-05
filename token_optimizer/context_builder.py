from typing import List, Tuple

from config import cfg, BudgetConfig
from token_optimizer.core.llm import count_tokens

TASK_SYSTEM_PROMPTS = {
    "jira": (
        "You are a technical project manager. "
        "Create a well-structured Jira ticket (title, description, acceptance criteria) "
        "based on the provided code context and request."
    ),
    "bug": (
        "You are a senior software engineer debugging an issue. "
        "Analyze the code context, identify the root cause, and provide a concise fix."
    ),
    "analysis": (
        "You are a senior software architect. "
        "Analyse the provided code context thoroughly and give actionable, specific insights."
    ),
    "coding": (
        "You are an expert software engineer. "
        "Provide a clear, working implementation based on the code context."
    ),
    "design": (
        "You are a software architect. "
        "Design a solution or explain the architecture based on the provided context."
    ),
    "response": (
        "You are a helpful technical assistant. "
        "Answer the question concisely and accurately using the provided context."
    ),
}


def build_context(
    chunks: List[dict],
    task_type: str,
    user_input: str,
    budget: BudgetConfig,
) -> Tuple[str, str, int]:
    """
    Returns (system_prompt, user_message, context_token_count).

    Fills context up to budget.max_input_tokens, picking highest-similarity
    chunks first.
    """
    system = TASK_SYSTEM_PROMPTS.get(task_type, TASK_SYSTEM_PROMPTS["response"])

    overhead = count_tokens(system) + count_tokens(user_input) + 150
    available = max(0, budget.max_input_tokens - overhead)

    selected: List[dict] = []
    used = 0
    for chunk in sorted(chunks, key=lambda c: float(c.get("similarity", 0)), reverse=True):
        ct = chunk.get("token_count", count_tokens(chunk["content"]))
        if used + ct <= available:
            selected.append(chunk)
            used += ct
        if used >= available:
            break

    if selected:
        parts = [
            f"### {c['file_path']} (lines {c['start_line']}–{c['end_line']})\n"
            f"```\n{c['content']}\n```"
            for c in selected
        ]
        context_block = "\n\n".join(parts)
        user_message = f"<context>\n{context_block}\n</context>\n\n{user_input}"
    else:
        user_message = user_input

    return system, user_message, used
