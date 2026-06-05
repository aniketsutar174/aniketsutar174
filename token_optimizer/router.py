from config import cfg

# Task types that benefit from a smarter model regardless of similarity
ALWAYS_SMART = {"design", "analysis"}


def route(task_type: str, top_similarity: float) -> str:
    """
    Return the model to use.
    Uses smart_model when context similarity is low (uncertain retrieval)
    or the task type is inherently complex.
    """
    if task_type in ALWAYS_SMART:
        return cfg.smart_model
    if top_similarity < cfg.difficulty_threshold:
        return cfg.smart_model
    return cfg.fast_model
