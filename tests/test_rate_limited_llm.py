import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from google.genai.errors import APIError

from llm import rate_limited_llm
from llm.llm import LLM
from llm.rate_limited_llm import RateLimitedLLM, TokenBucket

RESPONSE = LLM.SimpleResponse(answer="hi", input_tokens=1, output_tokens=2)

class FakeClock:
    """Virtual clock: sleeping advances time instead of spending it.

    Keeps the pacing assertions exact (no tolerance for scheduler jitter) and
    instant.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += max(seconds, 0.0)

@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    """Install a fake clock into the module under test only.

    The `asyncio` *name* in rate_limited_llm is swapped for a shim rather than
    patching `asyncio.sleep` itself, which is process-wide and would hand the
    fake sleep to pytest-asyncio's own event loop machinery.
    """
    fake = FakeClock()
    monkeypatch.setattr(rate_limited_llm.time, "monotonic", fake.monotonic)
    monkeypatch.setattr(
        rate_limited_llm,
        "asyncio",
        SimpleNamespace(sleep=fake.sleep, Semaphore=asyncio.Semaphore),
    )
    return fake

class FakeLLM(LLM):
    """Scripts one outcome per call: an exception to raise or a response to
    return. The last element repeats once the script runs out."""

    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = outcomes
        self.calls = 0
        self.closed = False

    async def ask_generic_question(
        self, system_prompt: str, question: str, temperature: float
    ) -> LLM.SimpleResponse:
        outcome = self._outcomes[min(self.calls, len(self._outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def aclose(self) -> None:
        self.closed = True

def api_error(code: int) -> APIError:
    """A real google-genai error, exercising the `.code` attribute path."""
    return APIError(code, {})

class StatusCodeError(Exception):
    """An httpx/Together-shaped error, exercising the `.status_code` path."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code

async def ask(limiter: RateLimitedLLM) -> LLM.SimpleResponse:
    return await limiter.ask_generic_question("system", "question", 0.0)

@pytest.mark.parametrize("rate", [8.0, 2.0, 0.5])
async def test_rate_is_capped_at_requests_per_second(clock: FakeClock, rate: float) -> None:
    # capacity=1 removes the burst allowance, leaving the steady-state rate.
    bucket = TokenBucket(rate, capacity=1)

    for _ in range(10):
        await bucket.acquire()

    # The first acquire spends the starting token; the other nine each wait a
    # full refill interval.
    assert clock.now == pytest.approx(9 / rate)

async def test_burst_allows_capacity_requests_without_waiting(clock: FakeClock) -> None:
    bucket = TokenBucket(refill_rate=8.0, capacity=8)

    for _ in range(8):
        await bucket.acquire()

    assert clock.now == 0.0  # the whole burst is free

    await bucket.acquire()

    assert clock.now == pytest.approx(1 / 8)  # the next one is paced

async def test_max_concurrent_is_never_exceeded() -> None:
    # Real event loop and real sleeps: the semaphore bounds actual interleaving,
    # and a fake clock would collapse the overlap this test depends on.
    max_concurrent = 4
    requests = 20
    in_flight = 0
    peak = 0

    class TrackingLLM(LLM):
        async def ask_generic_question(
            self, system_prompt: str, question: str, temperature: float
        ) -> LLM.SimpleResponse:
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            # Hold the slot long enough that every admitted request overlaps, so
            # the peak reflects the cap rather than how fast calls happen to finish.
            await asyncio.sleep(0.01)
            in_flight -= 1
            return RESPONSE

    # Rate limiting is effectively disabled so it cannot serialize the requests.
    limiter = RateLimitedLLM(
        TrackingLLM(),
        requests_per_second=1e6,
        burst=requests,
        max_concurrent=max_concurrent,
    )

    await asyncio.gather(*(ask(limiter) for _ in range(requests)))

    assert peak <= max_concurrent
    assert peak == max_concurrent  # the cap was actually under pressure

@pytest.mark.parametrize("max_attempts", [1, 3, 5])
async def test_max_attempts_bounds_retries(clock: FakeClock, max_attempts: int) -> None:
    delegate = FakeLLM([api_error(429)])
    limiter = RateLimitedLLM(
        delegate,
        requests_per_second=1e6,  # pacing waits would only muddy the counts
        burst=max_attempts,
        max_attempts=max_attempts,
    )

    with pytest.raises(APIError):
        await ask(limiter)

    assert delegate.calls == max_attempts
    # One backoff between attempts, none after the last, and each grows.
    assert len(clock.sleeps) == max_attempts - 1
    assert clock.sleeps == sorted(clock.sleeps)

@pytest.mark.parametrize("code", [400, 403, 404, 500])
async def test_non_retryable_errors_are_raised_immediately(clock: FakeClock, code: int) -> None:
    delegate = FakeLLM([api_error(code)])
    limiter = RateLimitedLLM(delegate, requests_per_second=1e6, max_attempts=5)

    with pytest.raises(APIError):
        await ask(limiter)

    assert delegate.calls == 1

@pytest.mark.parametrize("code", RateLimitedLLM.RETRYABLE_STATUS)
@pytest.mark.parametrize("build_error", [api_error, StatusCodeError], ids=["code", "status_code"])
async def test_retryable_errors_are_retried_until_success(
    clock: FakeClock, code: int, build_error: Any
) -> None:
    # Parametrized over both attribute shapes, since the retry check duck-types
    # `.code` (google-genai) and `.status_code` (Together and other httpx SDKs).
    delegate = FakeLLM([build_error(code), RESPONSE])
    limiter = RateLimitedLLM(delegate, requests_per_second=1e6, max_attempts=5)

    assert await ask(limiter) is RESPONSE
    assert delegate.calls == 2

async def test_errors_without_a_status_are_not_retried(clock: FakeClock) -> None:
    delegate = FakeLLM([ValueError("bug in the delegate")])
    limiter = RateLimitedLLM(delegate, requests_per_second=1e6, max_attempts=5)

    with pytest.raises(ValueError):
        await ask(limiter)

    assert delegate.calls == 1  # a real bug surfaces immediately, not 5 times

async def test_aclose_closes_the_delegate() -> None:
    delegate = FakeLLM([RESPONSE])
    limiter = RateLimitedLLM(delegate)

    await limiter.aclose()

    assert delegate.closed is True

async def test_parallelism_reports_the_concurrency_cap() -> None:
    assert RateLimitedLLM(FakeLLM([RESPONSE]), max_concurrent=3).parallelism() == 3
