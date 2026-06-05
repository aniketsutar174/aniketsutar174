import os
from typing import Tuple

import anthropic


_client: anthropic.Anthropic = None  # type: ignore[assignment]


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set — add it to .env")
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def count_tokens(text: str) -> int:
    """Approximate token count (1 token ≈ 4 chars)."""
    return max(1, len(text) // 4)


def complete(
    system: str,
    user: str,
    model: str,
    max_tokens: int = 1024,
) -> Tuple[str, int, int]:
    """
    Returns (content, input_tokens, output_tokens).
    Uses real usage counts from the Anthropic response.
    """
    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    content = response.content[0].text
    return content, response.usage.input_tokens, response.usage.output_tokens
