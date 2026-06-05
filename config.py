import os
from dataclasses import dataclass, field
from typing import Dict
from dotenv import load_dotenv

load_dotenv()

TASK_TYPES = ["jira", "bug", "analysis", "coding", "design", "response"]

# Per-model embedding dimensions (voyage AI)
VOYAGE_EMBEDDING_DIMS: Dict[str, int] = {
    "voyage-code-2": 1536,
    "voyage-2": 1024,
    "voyage-large-2": 1536,
    "voyage-3": 1024,
    "voyage-code-3": 1024,
    "voyage-3-lite": 512,
}


@dataclass
class BudgetConfig:
    max_input_tokens: int = 4000
    max_output_tokens: int = 1024


@dataclass
class Config:
    # Models
    fast_model: str = "claude-haiku-4-5-20251001"
    smart_model: str = "claude-sonnet-4-6"

    # Use smart_model when top-chunk similarity is below this threshold
    difficulty_threshold: float = 0.65

    # Retrieval
    candidate_chunks: int = 20   # initial pool pulled from pgvector
    final_chunks: int = 5        # kept for context after similarity sort

    # Chunking
    chunk_lines: int = 50
    chunk_overlap_lines: int = 10

    # Database
    db_url: str = field(default_factory=lambda: os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/tokenopt",
    ))

    # Voyage embeddings
    voyage_model: str = "voyage-code-2"

    @property
    def embedding_dim(self) -> int:
        return VOYAGE_EMBEDDING_DIMS.get(self.voyage_model, 1536)

    # Token budgets per task type
    budgets: Dict[str, BudgetConfig] = field(default_factory=lambda: {
        "jira":     BudgetConfig(max_input_tokens=2000,  max_output_tokens=512),
        "bug":      BudgetConfig(max_input_tokens=4000,  max_output_tokens=1024),
        "analysis": BudgetConfig(max_input_tokens=4000,  max_output_tokens=2048),
        "coding":   BudgetConfig(max_input_tokens=6000,  max_output_tokens=2048),
        "design":   BudgetConfig(max_input_tokens=4000,  max_output_tokens=2048),
        "response": BudgetConfig(max_input_tokens=2000,  max_output_tokens=512),
    })


cfg = Config()
