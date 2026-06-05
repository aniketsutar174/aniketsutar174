import pathlib
from typing import Iterator, List

from config import cfg
from token_optimizer.core.db import init_db, insert_chunks, delete_repo_chunks
from token_optimizer.core.embeddings import embed_texts
from token_optimizer.core.llm import count_tokens

CODE_EXTENSIONS = {
    ".py", ".go", ".ts", ".tsx", ".js", ".jsx", ".rs", ".java",
    ".c", ".cpp", ".h", ".hpp", ".cs", ".rb", ".php", ".swift",
    ".kt", ".scala", ".sh", ".yaml", ".yml", ".toml", ".json", ".md",
}

SKIP_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", ".pytest_cache", ".mypy_cache", ".tox",
}


def _iter_files(root: str) -> Iterator[pathlib.Path]:
    for p in pathlib.Path(root).rglob("*"):
        if not p.is_file():
            continue
        if p.suffix not in CODE_EXTENSIONS:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        yield p


def _chunk_file(text: str, file_path: str) -> List[dict]:
    lines = text.splitlines()
    if not lines:
        return []
    chunks: List[dict] = []
    step = max(1, cfg.chunk_lines - cfg.chunk_overlap_lines)
    start = 0
    while start < len(lines):
        end = min(start + cfg.chunk_lines, len(lines))
        content = "\n".join(lines[start:end]).strip()
        if content:
            chunks.append({
                "file_path": file_path,
                "start_line": start + 1,
                "end_line": end,
                "content": content,
                "token_count": count_tokens(content),
            })
        if end >= len(lines):
            break
        start += step
    return chunks


def index_repo(repo: str, root: str, reindex: bool = False) -> dict:
    """Index a directory into pgvector under the given repo name."""
    init_db(cfg.db_url, embedding_dim=cfg.embedding_dim)

    if reindex:
        deleted = delete_repo_chunks(cfg.db_url, repo)
        print(f"  Removed {deleted} existing chunks for '{repo}'")

    all_chunks: List[dict] = []
    files_seen = 0
    for path in _iter_files(root):
        try:
            text = path.read_text(errors="replace")
        except Exception:
            continue
        rel = str(path.relative_to(root))
        chunks = _chunk_file(text, rel)
        all_chunks.extend(chunks)
        files_seen += 1

    if not all_chunks:
        return {"files": 0, "chunks": 0, "total_tokens": 0}

    texts = [c["content"] for c in all_chunks]
    embeddings = embed_texts(texts, input_type="document", model=cfg.voyage_model)

    for chunk, emb in zip(all_chunks, embeddings):
        chunk["embedding"] = emb

    written = insert_chunks(cfg.db_url, repo, all_chunks)
    total_tokens = sum(c["token_count"] for c in all_chunks)

    return {"files": files_seen, "chunks": written, "total_tokens": total_tokens}
