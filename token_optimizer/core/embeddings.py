import os
import time
from typing import List

import voyageai


_client: voyageai.Client = None  # type: ignore[assignment]


def _get_client() -> voyageai.Client:
    global _client
    if _client is None:
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY not set — add it to .env")
        _client = voyageai.Client(api_key=api_key)
    return _client


def embed_texts(
    texts: List[str],
    input_type: str = "document",
    model: str = "voyage-code-2",
) -> List[List[float]]:
    client = _get_client()
    results: List[List[float]] = []
    batch_size = 128
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embed(batch, model=model, input_type=input_type)
        results.extend(resp.embeddings)
        if i + batch_size < len(texts):
            time.sleep(0.05)
    return results


def embed_query(text: str, model: str = "voyage-code-2") -> List[float]:
    return embed_texts([text], input_type="query", model=model)[0]
