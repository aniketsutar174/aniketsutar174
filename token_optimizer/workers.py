import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, List, TypeVar

T = TypeVar("T")

_executor = ThreadPoolExecutor(max_workers=4)


async def run_in_thread(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Run a blocking function in the shared thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, lambda: fn(*args, **kwargs))


async def batch_run(fn: Callable[..., T], items: List[Any]) -> List[Any]:
    """Run fn on each item concurrently via the thread pool."""
    tasks = [run_in_thread(fn, item) for item in items]
    return await asyncio.gather(*tasks, return_exceptions=True)
