"""Rate-limited wrapper around a single google-genai client.

Three concerns are composed rather than tangled together:

  TokenBucket        smooths the request *rate* into a steady stream.
  Semaphore          caps how many requests are *in flight* at once.
  retry-with-backoff absorbs the 429/503 responses that Vertex DSQ returns
                     when the shared pool is momentarily saturated.

Nothing here provisions against a fixed quota, because gemini-2.5-flash on
Vertex uses Dynamic Shared Quota: the real ceiling is unobservable and
floats with global demand. So we pace conservatively and let backoff find
the true limit instead of sizing to a published number.
"""

import random
import threading
import time

from google.genai import errors


class TokenBucket:
    """Thread-safe token bucket rate limiter.

    The bucket holds up to `capacity` tokens and refills continuously at
    `refill_rate` tokens per second. Each unit of work consumes one token;
    when the bucket is empty, callers wait for it to refill. This is what
    converts bursty callers into a steady rate.

    The algorithm has three moving parts, each its own method below:
      _refill()            -- add the tokens that accrued since last check
      _try_consume()       -- take a token if one is available
      _time_until_token()  -- how long until the next token is ready
    """

    def __init__(self, refill_rate, capacity):
        self._refill_rate = refill_rate
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self):
        """Block until a token is available, then consume it."""
        while not self._try_consume():
            time.sleep(self._time_until_token())

    def _try_consume(self):
        """Consume one token if available. Return True on success."""
        with self._lock:
            self._refill()
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return True
            return False

    def _time_until_token(self):
        """Return the seconds until the bucket holds one full token."""
        with self._lock:
            self._refill()
            deficit = 1.0 - self._tokens
            return deficit / self._refill_rate

    def _refill(self):
        """Add tokens accrued since the last refill, capped at capacity.

        Caller must hold the lock. Refilling is time-based rather than on a
        timer thread: we compute how many tokens *should* have arrived given
        the elapsed wall-clock time, which needs no background scheduling.
        """
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_rate)
        self._last_refill = now


class RateLimitedGemini:
    """Paces, bounds, and retries calls through one google-genai client.

    The underlying client is thread-safe, so construct a single instance
    and share it across workers.
    """

    RETRYABLE_STATUS = (429, 503)
    MAX_BACKOFF_SECONDS = 30

    def __init__(
        self,
        client,
        requests_per_second=8.0,
        burst=8,
        max_concurrent=8,
        max_attempts=5,
    ):
        self._client = client
        self._rate_limiter = TokenBucket(requests_per_second, burst)
        self._in_flight = threading.Semaphore(max_concurrent)
        self._max_attempts = max_attempts

    def generate_content(self, **kwargs):
        """Rate-limited, concurrency-capped generate_content with retries.

        Each attempt takes a fresh token, since a retry is another real
        request against the shared pool and should be paced like any other.
        """
        for attempt in range(self._max_attempts):
            self._rate_limiter.acquire()
            try:
                return self._send(**kwargs)
            except errors.APIError as error:
                if not self._is_retryable(error, attempt):
                    raise
                time.sleep(self._backoff_seconds(attempt))

    def _send(self, **kwargs):
        """Issue one request, holding a concurrency slot for its duration."""
        with self._in_flight:
            return self._client.models.generate_content(**kwargs)

    def _is_retryable(self, error, attempt):
        """Retry only transient overload errors, and only if attempts remain."""
        attempts_remain = attempt < self._max_attempts - 1
        return error.code in self.RETRYABLE_STATUS and attempts_remain

    def _backoff_seconds(self, attempt):
        """Exponential backoff with jitter, capped to bound the wait."""
        exponential = min(2 ** attempt, self.MAX_BACKOFF_SECONDS)
        jitter = random.uniform(0, 1)
        return exponential + jitter