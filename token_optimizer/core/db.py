import psycopg2
import psycopg2.extras
import numpy as np
from pgvector.psycopg2 import register_vector
from typing import List


def _connect(db_url: str):
    conn = psycopg2.connect(db_url)
    register_vector(conn)
    return conn


def init_db(db_url: str, embedding_dim: int = 1536) -> None:
    conn = _connect(db_url)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id         SERIAL PRIMARY KEY,
                repo       TEXT NOT NULL,
                file_path  TEXT NOT NULL,
                start_line INT  NOT NULL,
                end_line   INT  NOT NULL,
                content    TEXT NOT NULL,
                embedding  vector({embedding_dim}),
                token_count INT NOT NULL DEFAULT 0,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS chunks_repo_idx ON chunks (repo)")
        conn.commit()
    conn.close()


def insert_chunks(db_url: str, repo: str, rows: List[dict]) -> int:
    if not rows:
        return 0
    conn = _connect(db_url)
    prepared = []
    for r in rows:
        row = dict(r)
        emb = row.get("embedding")
        if emb is not None:
            row["embedding"] = np.array(emb, dtype=np.float32)
        prepared.append({"repo": repo, **row})
    with conn.cursor() as cur:
        psycopg2.extras.execute_batch(cur, """
            INSERT INTO chunks (repo, file_path, start_line, end_line, content, embedding, token_count)
            VALUES (%(repo)s, %(file_path)s, %(start_line)s, %(end_line)s,
                    %(content)s, %(embedding)s, %(token_count)s)
        """, prepared)
        conn.commit()
    conn.close()
    return len(rows)


def search_chunks(db_url: str, repo: str, query_embedding: List[float], k: int = 20) -> List[dict]:
    conn = _connect(db_url)
    emb = np.array(query_embedding, dtype=np.float32)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT id, file_path, start_line, end_line, content, token_count,
                   1 - (embedding <=> %s) AS similarity
            FROM chunks
            WHERE repo = %s
            ORDER BY embedding <=> %s
            LIMIT %s
        """, (emb, repo, emb, k))
        rows = cur.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_chunks(db_url: str, repo: str) -> int:
    conn = _connect(db_url)
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM chunks WHERE repo = %s", (repo,))
        n = cur.fetchone()[0]
    conn.close()
    return int(n)


def total_repo_tokens(db_url: str, repo: str) -> int:
    conn = _connect(db_url)
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(SUM(token_count), 0) FROM chunks WHERE repo = %s", (repo,))
        total = cur.fetchone()[0]
    conn.close()
    return int(total)


def delete_repo_chunks(db_url: str, repo: str) -> int:
    conn = _connect(db_url)
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE repo = %s", (repo,))
        deleted = cur.rowcount
        conn.commit()
    conn.close()
    return deleted
