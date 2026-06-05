from typing import List, Optional

from config import cfg
from token_optimizer.core.db import search_chunks
from token_optimizer.core.embeddings import embed_query


def retrieve(repo: str, query: str, k: Optional[int] = None) -> List[dict]:
    """Return up to k chunks most similar to query, ordered by similarity desc."""
    if k is None:
        k = cfg.candidate_chunks
    query_emb = embed_query(query, model=cfg.voyage_model)
    return search_chunks(cfg.db_url, repo, query_emb, k=k)
