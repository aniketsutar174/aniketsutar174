"""
Offline dense embeddings via sklearn HashingVectorizer + random projection.

No internet access or API keys required. Produces 384-dim L2-normalised
vectors. Cosine similarity is meaningful for code: files sharing function
names, identifiers, and keywords score higher than unrelated files.
"""
from typing import List

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

# Sparse hashing vectorizer — 2^14 features, sublinear TF
_hv = HashingVectorizer(
    n_features=2**14,
    alternate_sign=False,
    norm="l2",
    analyzer="word",
    token_pattern=r"[A-Za-z_][A-Za-z0-9_]{1,}",  # code identifiers
)

# Fixed random projection matrix: (2^14, 384)
_rng = np.random.default_rng(seed=42)
_PROJ = _rng.standard_normal((2**14, 384)).astype(np.float32)
_PROJ /= np.linalg.norm(_PROJ, axis=0, keepdims=True)  # column-normalise


def _dense(text: str) -> np.ndarray:
    sparse = _hv.transform([text])          # (1, 2^14) sparse
    dense = sparse.dot(_PROJ)              # (1, 384)
    vec = dense[0].astype(np.float32)
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm > 0 else vec


def embed_texts(
    texts: List[str],
    input_type: str = "document",   # kept for API compatibility
    model: str = "local-hash",      # ignored — always uses offline method
) -> List[List[float]]:
    return [_dense(t).tolist() for t in texts]


def embed_query(text: str, model: str = "local-hash") -> List[float]:
    return _dense(text).tolist()
