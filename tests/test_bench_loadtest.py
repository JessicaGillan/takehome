"""Live load-test scenarios for RateLimitedLLM against the real Gemini API.

Each scenario wraps a Gemini in a RateLimitedLLM with different pacing/
concurrency/retry limits, drives it through tests/bench/runner.run_benchmark,
then saves the analyze() summary alongside the raw JSONL.

Marked `loadtest`, NOT `integration`: this is sustained load costing up to
$1.50 per full run, an order of magnitude more than the single-request
integration tests, so it must never ride along with them. `pytest`,
`pytest -m integration`, and every other command exclude it; the only way to
run it is to ask for it by name:

    .venv/bin/python -m pytest -m loadtest

Real, billable calls against GOOGLE_CLOUD_PROJECT; requires application
default credentials.
"""

import time

import pytest

from llm import Gemini, RateLimitedLLM
from tests.bench.analyze import save_analysis
from tests.bench.pricing import Pricing
from tests.bench.runner import run_benchmark
from tests.bench.store import CheckpointStore

MAX_REQUESTS = 100
MAX_COST_USD = 0.50

@pytest.mark.loadtest
@pytest.mark.parametrize(
    "name,requests_per_second,burst,max_concurrent,max_attempts",
    [
        ("conservative", 2.0, 2, 2, 5),      # steady low-rate baseline
        ("bursty", 8.0, 16, 8, 5),           # burst capacity + higher concurrency
        ("thin-retry-budget", 4.0, 4, 4, 1), # no retries: 429/503 surface as failures
    ],
)
async def test_scenario(
    name: str, requests_per_second: float, burst: int, max_concurrent: int, max_attempts: int
) -> None:
    client = RateLimitedLLM(
        Gemini(),
        requests_per_second=requests_per_second,
        burst=burst,
        max_concurrent=max_concurrent,
        max_attempts=max_attempts,
    )
    path = f"tests/bench/results/{name}-{int(time.time())}.jsonl"
    store = CheckpointStore(path)

    try:
        budget = await run_benchmark(
            client, store, Pricing(),
            max_requests=MAX_REQUESTS, max_cost_usd=MAX_COST_USD, max_in_flight=max_concurrent,
        )
    finally:
        await client.aclose()

    summary = save_analysis(path)

    assert summary["requests"] > 0
    assert budget.spent_usd <= MAX_COST_USD + 0.05  # small in-flight-tail tolerance
