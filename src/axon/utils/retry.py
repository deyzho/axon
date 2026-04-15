"""Exponential backoff retry utility."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


async def with_retry(
    fn: Callable[[], Any],
    *,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    should_retry: Callable[[Exception, int], bool] | None = None,
) -> Any:
    """
    Execute fn with exponential backoff.

    Delays: base_delay * 2^(attempt-1) ± 20% jitter.
    Raises the last exception if all attempts fail.

    Args:
        fn: Async callable to retry.
        max_attempts: Maximum number of attempts (default 3).
        base_delay: Base delay in seconds (default 1.0).
        should_retry: Optional callable(err, attempt) → bool.
                      If None, always retries.
    """
    last_err: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn()
        except Exception as exc:
            last_err = exc
            if attempt == max_attempts:
                raise
            if should_retry is not None and not should_retry(exc, attempt):
                raise
            delay = base_delay * (2 ** (attempt - 1)) * (0.8 + random.random() * 0.4)
            await asyncio.sleep(delay)
    raise last_err  # type: ignore[misc]  # unreachable but satisfies type checker
