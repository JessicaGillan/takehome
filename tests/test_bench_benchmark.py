"""Production-rate benchmark tiers against the live Gemini API.

Verifies the provider withstands a stated request rate, tier by tier. Each
tier drives a 30-second window at its target rpm through RateLimitedLLM and
passes only if the harness kept pace (attempted rate >= 90% of target) AND the
provider absorbed it (goodput >= 90% of target). A failing tier is the
finding: it marks where Dynamic Shared Quota stops absorbing the load, and the
saved `.analysis.json` shows whether 429s, latency, or empty responses broke it.

Cost control: every tier caps GEMINI_MAX_OUTPUT_TOKENS at 256 (output tokens
dominate cost at $2.50/Mtok) and carries its own request cap and dollar
budget. Budgets are sized ~1.5x worst case so the cost cap is a safety
backstop, never the expected stopper -- if it fires, the run was truncated and
the rate assertions fail, which is the correct outcome.

| tier    | requests | budget  | worst-case cost |
| ------- | -------- | ------- | --------------- |
| 500rpm  | 250      | $0.25   | $0.17           |
| 1k      | 500      | $0.50   | $0.33           |
| 5k      | 2,500    | $2.50   | $1.63           |

Full sweep: 3,250 requests, budgets sum $3.25, worst-case actual ~$2.13,
typical (~150-token answers) ~$1.30. Tiers above 5k were removed after the
2026-07-21 run found the quota ceiling; see VERIFICATION.md "Measured results".

Marked `benchmark` -- excluded from `pytest`, `-m integration`, and
`-m loadtest`. Run explicitly:

    .venv/bin/python -m pytest -m benchmark              # all six tiers
    .venv/bin/python -m pytest -m benchmark -k 500rpm    # cheapest smoke (~$0.17)

Real, billable calls against GOOGLE_CLOUD_PROJECT; requires application
default credentials.
"""

import time
from dataclasses import dataclass

import pytest

from llm import Gemini, RateLimitedLLM
from tests.bench.analyze import save_analysis
from tests.bench.pricing import Pricing
from tests.bench.runner import run_benchmark
from tests.bench.store import CheckpointStore

# Cap on answer tokens for benchmark runs; see the cost table in the docstring.
BENCHMARK_MAX_OUTPUT_TOKENS = 256

# Retries stay bounded so a heavily throttled tier degrades the measured rate
# instead of stretching the run by minutes: 3 attempts caps per-request retry
# drag at ~4.5s of backoff, versus ~20s at the default 5.
BENCHMARK_MAX_ATTEMPTS = 3

# Passing bar for both attempted rate (did the harness keep pace?) and goodput
# (did the provider return useful answers at that pace?).
TARGET_FRACTION = 0.9

# The budget guard admits work until spent >= cap, so the requests already in
# flight at that moment can land just past it; one worst-case response of
# headroom per concurrent slot is far more than they can bill.
BUDGET_TOLERANCE_USD = 0.05

@dataclass(frozen=True)
class Tier:
    """One production-rate scenario: a 30s window at `rpm`, bounded two ways.

    max_requests is rpm/2 (the 30s window); max_concurrent is sized from
    rps x ~1.5s expected latency with ~2x headroom, and also sizes the
    Gemini connection pool via GEMINI_MAX_CONNECTIONS.
    """

    name: str
    rpm: int
    max_requests: int
    max_concurrent: int
    burst: int
    budget_usd: float

# No tier above 5k: the 2026-07-21 run measured this project's DSQ ceiling at
# ~505 useful rpm (68% of attempts 429-rejected at ~1,800 rpm attempted), so
# 10k-30k targets are unreachable until quota changes -- rerunning them only
# spends money re-confirming a known refusal. See VERIFICATION.md, "Measured
# results", before re-adding tiers: >160 concurrency also needs `ulimit -n`
# raised above the macOS default 256.
TIERS = [
    Tier("500rpm", 500, 250, 16, 9, 0.25),
    Tier("1k", 1_000, 500, 32, 17, 0.50),
    Tier("5k", 5_000, 2_500, 160, 84, 2.50),
]

@pytest.mark.benchmark
@pytest.mark.parametrize("tier", TIERS, ids=[tier.name for tier in TIERS])
async def test_sustains_target_rpm(tier: Tier, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", str(BENCHMARK_MAX_OUTPUT_TOKENS))
    monkeypatch.setenv("GEMINI_MAX_CONNECTIONS", str(tier.max_concurrent))
    client = RateLimitedLLM(
        Gemini(),
        requests_per_second=tier.rpm / 60,
        burst=tier.burst,
        max_concurrent=tier.max_concurrent,
        max_attempts=BENCHMARK_MAX_ATTEMPTS,
    )
    path = f"tests/bench/results/benchmark-{tier.name}-{int(time.time())}.jsonl"
    store = CheckpointStore(path)

    try:
        budget = await run_benchmark(
            client, store, Pricing(),
            max_requests=tier.max_requests,
            max_cost_usd=tier.budget_usd,
            max_in_flight=tier.max_concurrent,
        )
    finally:
        await client.aclose()

    summary = save_analysis(path)

    assert budget.spent_usd <= tier.budget_usd + BUDGET_TOLERANCE_USD
    assert summary["attempted_rps"] * 60 >= TARGET_FRACTION * tier.rpm  # harness kept pace
    assert summary["goodput_rpm"] >= TARGET_FRACTION * tier.rpm         # provider withstood it
