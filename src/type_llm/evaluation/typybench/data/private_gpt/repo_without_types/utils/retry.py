import logging
from collections.abc import Callable
from typing import Any

from retry_async import retry as retry_untyped  # type: ignore

retry_logger = logging.getLogger(__name__)


def retry(
    exceptions = Exception, *,
    is_async = False, tries = -1, delay = 0, max_delay = None, backoff = 1, jitter = 0, logger = retry_logger):
    wrapped = retry_untyped(
        exceptions=exceptions,
        is_async=is_async,
        tries=tries,
        delay=delay,
        max_delay=max_delay,
        backoff=backoff,
        jitter=jitter,
        logger=logger,
    )
    return wrapped  # type: ignore
