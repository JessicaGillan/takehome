import threading
from typing import Any

import pytest
from google.genai.errors import APIError

from llm import rate_limited_gemini
from llm.rate_limited_gemini import RateLimitedGemini, TokenBucket

# Sentinel returned by the fake client, so a test can tell "the call succeeded"
# apart from the implicit None a fallen-through retry loop would return.
RESPONSE = object()

class FakeClock:
    """Virtual clock: sleeping advances time instead of spending it.

    Substituted for the module's `time`, which keeps the pacing tests exact
    (no tolerance for scheduler jitter) and instant.
    """

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += max(seconds, 0.0)

@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(rate_limited_gemini.time, "monotonic", fake.monotonic)
    monkeypatch.setattr(rate_limited_gemini.time, "sleep", fake.sleep)
    return fake

class FakeModels:
    """Stands in for `client.models`, scripting one outcome per call.

    Each element of `outcomes` is either an exception to raise or a value to
    return; the last element repeats once the script runs out.
    """

    def __init__(self, outcomes: list[Any]) -> None:
        self._outcomes = outcomes
        self.calls = 0

    def generate_content(self, **kwargs: Any) -> Any:
        outcome = self._outcomes[min(self.calls, len(self._outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

class FakeClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.models = FakeModels(outcomes)

    @property
    def calls(self) -> int:
        return self.models.calls

def api_error(code: int) -> APIError:
    """A real SDK error, so the production `except`/`.code` path is exercised."""
    return APIError(code, {})

@pytest.mark.parametrize("rate", [8.0, 2.0, 0.5])
def test_rate_is_capped_at_requests_per_second(clock: FakeClock, rate: float) -> None:
    # capacity=1 removes the burst allowance, leaving the steady-state rate.
    bucket = TokenBucket(rate, capacity=1)

    for _ in range(10):
        bucket.acquire()

    # The first acquire spends the starting token; the other nine each wait a
    # full refill interval.
    assert clock.now == pytest.approx(9 / rate)

def test_burst_allows_capacity_requests_without_waiting(clock: FakeClock) -> None:
    bucket = TokenBucket(refill_rate=8.0, capacity=8)

    for _ in range(8):
        bucket.acquire()

    assert clock.now == 0.0  # the whole burst is free

    bucket.acquire()

    assert clock.now == pytest.approx(1 / 8)  # the next one is paced

def test_max_concurrent_is_never_exceeded() -> None:
    # Real threads and a real clock: the semaphore is a concurrency primitive,
    # and faking either side would prove nothing about it.
    max_concurrent = 4
    threads = 20
    lock = threading.Lock()
    started = threading.Barrier(threads, timeout=5)
    in_flight = 0
    peak = 0

    def record_and_yield(**kwargs: Any) -> Any:
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        # Hold the slot long enough that every admitted thread overlaps, so the
        # peak reflects the cap rather than how fast threads happen to finish.
        threading.Event().wait(0.02)
        with lock:
            in_flight -= 1
        return RESPONSE

    client = FakeClient([RESPONSE])
    client.models.generate_content = record_and_yield  # type: ignore[method-assign]
    # Rate limiting is effectively disabled so it cannot serialize the workers.
    limiter = RateLimitedGemini(
        client,
        requests_per_second=1e6,
        burst=threads,
        max_concurrent=max_concurrent,
    )

    def worker() -> None:
        started.wait()
        limiter.generate_content(model="fake")

    workers = [threading.Thread(target=worker) for _ in range(threads)]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert peak <= max_concurrent
    assert peak == max_concurrent  # the cap was actually under pressure

@pytest.mark.parametrize("max_attempts", [1, 3, 5])
def test_max_attempts_bounds_retries(clock: FakeClock, max_attempts: int) -> None:
    client = FakeClient([api_error(429)])
    limiter = RateLimitedGemini(
        client,
        requests_per_second=1e6,  # pacing waits would only muddy the counts
        burst=max_attempts,
        max_attempts=max_attempts,
    )

    with pytest.raises(APIError):
        limiter.generate_content(model="fake")

    assert client.calls == max_attempts
    # One backoff between attempts, none after the last, and each grows.
    assert len(clock.sleeps) == max_attempts - 1
    assert clock.sleeps == sorted(clock.sleeps)

@pytest.mark.parametrize("code", [400, 403, 404, 500])
def test_non_retryable_errors_are_raised_immediately(clock: FakeClock, code: int) -> None:
    client = FakeClient([api_error(code)])
    limiter = RateLimitedGemini(client, requests_per_second=1e6, max_attempts=5)

    with pytest.raises(APIError):
        limiter.generate_content(model="fake")

    assert client.calls == 1

@pytest.mark.parametrize("code", RateLimitedGemini.RETRYABLE_STATUS)
def test_retryable_errors_are_retried_until_success(clock: FakeClock, code: int) -> None:
    client = FakeClient([api_error(code), RESPONSE])
    limiter = RateLimitedGemini(client, requests_per_second=1e6, max_attempts=5)

    assert limiter.generate_content(model="fake") is RESPONSE
    assert client.calls == 2
