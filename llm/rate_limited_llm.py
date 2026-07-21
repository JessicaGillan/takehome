"""Rate-limited wrapper around any LLM implementation.

Three concerns are composed rather than tangled together:

  TokenBucket        smooths the request *rate* into a steady stream.
  Semaphore          caps how many requests are *in flight* at once.
  retry-with-backoff absorbs the 429/503 responses a provider returns when
                     its shared pool is momentarily saturated.

Nothing here provisions against a fixed quota, because gemini-2.5-flash on
Vertex uses Dynamic Shared Quota: the real ceiling is unobservable and
floats with global demand. So we pace conservatively and let backoff find
the true limit instead of sizing to a published number.

The limiter is async and lives on a single event loop, which is why none of
the state below needs a lock -- see TokenBucket.
"""

import asyncio
import random
import time
from typing import Optional

from llm.llm import LLM

class TokenBucket:
    """Token bucket rate limiter for a single event loop.

    The bucket holds up to `capacity` tokens and refills continuously at
    `refill_rate` tokens per second. Each unit of work consumes one token;
    when the bucket is empty, callers wait for it to refill. This is what
    converts bursty callers into a steady rate.

    No lock is needed: the methods that read-modify-write `_tokens` contain no
    `await`, so the event loop cannot interrupt one part-way. Adding an `await`
    inside any of them would break that and reintroduce the need for a lock.

    The algorithm has three moving parts, each its own method below:
      _refill()            -- add the tokens that accrued since last check
      _try_consume()       -- take a token if one is available
      _time_until_token()  -- how long until the next token is ready
    """

    def __init__(self, refill_rate: float, capacity: int):
        self._refill_rate = refill_rate
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()

    async def acquire(self) -> None:
        """Wait until a token is available, then consume it."""
        while not self._try_consume():
            await asyncio.sleep(self._time_until_token())

    def _try_consume(self) -> bool:
        """Consume one token if available. Return True on success."""
        self._refill()
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return True
        return False

    def _time_until_token(self) -> float:
        """Return the seconds until the bucket holds one full token."""
        self._refill()
        deficit = 1.0 - self._tokens
        return deficit / self._refill_rate

    def _refill(self) -> None:
        """Add tokens accrued since the last refill, capped at capacity.

        Refilling is time-based rather than on a timer task: we compute how
        many tokens *should* have arrived given the elapsed wall-clock time,
        which needs no background scheduling.
        """
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now

def status_code(error: BaseException) -> Optional[int]:
    """Best-effort HTTP status off an SDK exception.

    google-genai exposes `.code`; Together (and most httpx-based SDKs) expose
    `.status_code`. Reading both keeps this module free of any provider import.
    """
    code = getattr(error, "code", None)
    if code is None:
        code = getattr(error, "status_code", None)
    return code if isinstance(code, int) else None

class RateLimitedLLM(LLM):
    """Paces, bounds, and retries calls through any LLM implementation.

    Construct one and share it across the tasks that should contend for the
    same budget -- the limits apply per instance, not per call site.
    """

    RETRYABLE_STATUS = (429, 503)
    MAX_BACKOFF_SECONDS = 30

    def __init__(
        self,
        delegate: LLM,
        requests_per_second: float = 8.0,
        burst: int = 8,
        max_concurrent: int = 8,
        max_attempts: int = 5,
    ):
        self._delegate = delegate
        self._rate_limiter = TokenBucket(requests_per_second, burst)
        self._max_concurrent = max_concurrent
        self._in_flight = asyncio.Semaphore(max_concurrent)
        self._max_attempts = max_attempts

    def parallelism(self) -> int:
        return self._max_concurrent

    async def aclose(self) -> None:
        """Close the wrapped provider. The limiter itself holds no resources."""
        await self._delegate.aclose()

    async def ask_generic_question(
        self, system_prompt: str, question: str, temperature: float
    ) -> LLM.SimpleResponse:
        """Rate-limited, concurrency-capped question with retries.

        Each attempt takes a fresh token, since a retry is another real
        request against the shared pool and should be paced like any other.
        """
        for attempt in range(self._max_attempts):
            await self._rate_limiter.acquire()
            try:
                return await self._ask(system_prompt, question, temperature)
            except Exception as error:
                # Anything without a retryable status re-raises unchanged, so
                # catching broadly here never swallows a real bug.
                if not self._is_retryable(error, attempt):
                    raise
                await asyncio.sleep(self._backoff_seconds(attempt))

    async def _ask(self, system_prompt: str, question: str, temperature: float) -> LLM.SimpleResponse:
        """Issue one request, holding a concurrency slot for its duration."""
        async with self._in_flight:
            return await self._delegate.ask_generic_question(system_prompt, question, temperature)

    def _is_retryable(self, error: BaseException, attempt: int) -> bool:
        """Retry only transient overload errors, and only if attempts remain."""
        attempts_remain = attempt < self._max_attempts - 1
        return status_code(error) in self.RETRYABLE_STATUS and attempts_remain

    def _backoff_seconds(self, attempt: int) -> float:
        """Exponential backoff with jitter, capped to bound the wait."""
        exponential = min(2 ** attempt, self.MAX_BACKOFF_SECONDS)
        jitter = random.uniform(0, 1)
        return exponential + jitter
