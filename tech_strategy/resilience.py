from __future__ import annotations

import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import TypeVar

from .errors import ExternalServiceError

T = TypeVar("T")


def retry_with_backoff(
    operation: Callable[[], T],
    *,
    operation_name: str,
    max_retries: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
    logger,
) -> T:
    """retryable 예외에 대해 exponential backoff를 적용한다."""
    attempt = 0
    while True:
        try:
            return operation()
        except ExternalServiceError as exc:
            if not exc.retryable or attempt >= max_retries:
                raise
            sleep_seconds = min(max_delay_seconds, base_delay_seconds * (2**attempt))
            jitter = random.uniform(0.0, min(base_delay_seconds, 1.0))
            logger.warning(
                "%s failed with retryable error on attempt %d/%d: %s. retrying in %.2fs",
                operation_name,
                attempt + 1,
                max_retries + 1,
                exc,
                sleep_seconds + jitter,
            )
            time.sleep(sleep_seconds + jitter)
            attempt += 1


def run_with_timeout(
    operation: Callable[[], T],
    *,
    timeout_seconds: float,
    timeout_error_factory: Callable[[float], Exception],
) -> T:
    """동기 작업을 별도 스레드에서 실행하고 제한 시간 초과 시 예외를 발생시킨다."""
    if timeout_seconds <= 0:
        return operation()

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(operation)
    try:
        return future.result(timeout=timeout_seconds)
    except FuturesTimeoutError as exc:
        future.cancel()
        raise timeout_error_factory(timeout_seconds) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

