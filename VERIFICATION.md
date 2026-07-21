# Verification

Where the proof lives. Each section names the behaviour, the test that pins it,
and — more importantly — *why* that test would fail if the behaviour broke. A
test that passes on both the working and the broken implementation proves
nothing, so the reasoning below is written in terms of what each assertion
rules out.

```bash
.venv/bin/python -m pytest              # 57 offline tests, ~1s, no cost
.venv/bin/python -m pytest -m integration   # 1 live request
.venv/bin/python -m pytest -m loadtest      # 3 live load scenarios, up to $1.50
```

The two live markers are excluded from every other command, including a bare
`pytest` and an explicit path to their file. See `CLAUDE.md`.

---

## 1. Concurrency

**Claim:** `RateLimitedLLM` sends requests *concurrently*, and never has more
than `max_concurrent` of them in flight at once.

**Proof:** `tests/test_rate_limited_llm.py::test_max_concurrent_is_never_exceeded`

The limiter holds an `asyncio.Semaphore(max_concurrent)`
(`llm/rate_limited_llm.py:113`) and acquires it around the delegate call
(`llm/rate_limited_llm.py:144`). The test measures the real thing rather than
inspecting the semaphore: a fake provider increments a counter on entry,
records the high-water mark, awaits, then decrements. Whatever peak that
counter reaches *is* the true simultaneous in-flight count.

Two assertions, and it takes both:

| Assertion                    | Failure it rules out                                                                                   |
| ---------------------------- | ------------------------------------------------------------------------------------------------------ |
| `peak <= max_concurrent`     | The cap is missing or too loose — requests running unbounded.                                            |
| `peak == max_concurrent`     | The opposite bug: **accidental serialization**. If the limiter ran one request at a time, peak would be 1. |

`<=` alone is vacuous: code that never runs anything concurrently satisfies it
trivially. `==` is what proves the requests genuinely overlap. Together they
say "exactly this many, no more and no fewer".

Three details make the measurement trustworthy:

- **The counter is read inside the delegate**, which runs while the semaphore
  slot is held — so it counts admitted requests, not queued ones.
- **The delegate awaits `asyncio.sleep(0.01)`** while holding its slot. Without
  a real suspension point the calls would complete one after another and the
  peak would read 1 even with a working cap.
- **The rate limiter is disabled** (`requests_per_second=1e6`). Otherwise
  pacing, not the semaphore, could be what bounds the peak — and the test would
  pass while measuring the wrong mechanism.

**Parametrized over `[1, 4, 7]`.** This matters more than it looks. A single
hardcoded case can be satisfied by an implementation that ignores the argument
entirely and hardcodes the same number internally. Checking three values proves
the cap *tracks the argument*. `max_concurrent=1` pulls double duty as the
serialization guard — it is the one case where "correctly capped" and "not
concurrent at all" are indistinguishable, which is exactly why the `==`
assertion is needed alongside it.

Related: `test_parallelism_reports_the_concurrency_cap` pins that
`parallelism()` reports the same number it enforces.

---

## 2. Request rate

**Claim:** requests are paced to `requests_per_second`, with `burst` allowed up
front.

**Proof:** `test_rate_is_capped_at_requests_per_second`,
`test_burst_allows_capacity_requests_without_waiting`

These run on a virtual clock (`FakeClock`, `tests/test_rate_limited_llm.py:14`)
where sleeping advances time instead of spending it, so the assertions are
*exact equalities* rather than tolerances, and the suite stays fast. Ten
acquires at rate *r* with `capacity=1` must consume precisely `9/r` seconds —
the first token is free, the other nine each wait a full refill interval.
Parametrized over three rates so the arithmetic is shown to scale rather than
matching one lucky number.

The clock is installed by swapping the `asyncio` *name* inside the module under
test, never by patching `asyncio.sleep` globally — that is process-wide and
would hand a fake sleep to pytest-asyncio's own event loop.

`test_burst_allows_capacity_requests_without_waiting` pins the burst versus
steady-state split: the first 8 acquires cost zero time, the 9th costs `1/8`s.
Without it, a bucket with unbounded capacity could still pass the rate test.

---

## 3. Retries

**Claim:** transient overload (429/503) is retried with backoff; everything
else fails immediately; retries are bounded.

**Proof:**

| Test                                             | Pins                                                                             |
| ------------------------------------------------ | -------------------------------------------------------------------------------- |
| `test_max_attempts_bounds_retries`               | Exactly `max_attempts` calls, then the error propagates. Parametrized `[1, 3, 5]`, so the never-retry edge is covered. Also checks `max_attempts - 1` backoffs, monotonically increasing. |
| `test_retryable_errors_are_retried_until_success` | 429 and 503 recover on the next attempt.                                          |
| `test_non_retryable_errors_are_raised_immediately` | 400/403/404/500 → exactly 1 call.                                                |
| `test_errors_without_a_status_are_not_retried`   | A plain `ValueError` surfaces once, not five times.                              |

That last one is the guard on the broad `except Exception` in the retry loop
(`llm/rate_limited_llm.py:135`). Catching broadly is only safe because anything
without a retryable status re-raises unchanged; the test is what keeps a real
bug from being silently retried and reclassified.

Retryability is duck-typed off `.code` or `.status_code`
(`llm/rate_limited_llm.py:81`) so the limiter imports no provider SDK.
`test_retryable_errors_are_retried_until_success` is parametrized over **both
attribute shapes** — a real `google.genai` `APIError` and an httpx-style error
— so the abstraction is proven for both conventions rather than assumed.

---

## 4. Cost and budget safety

**Claim:** a load run stops at its request cap *and* its dollar cap.

**Proof:** `tests/test_bench_offline.py::test_run_benchmark_stops_at_max_requests`,
`::test_run_benchmark_stops_early_once_the_cost_cap_is_hit`

The cost cap is the one that matters, and it is the one that is easy to get
silently wrong. An earlier draft of `run_benchmark` launched every reserved
request before processing any result, so `spent_usd` never updated during
launch and the cap could never stop a run early — it only decorated the final
count. The current implementation interleaves launching and result collection
(`tests/bench/runner.py`), and the test pins the consequence: a fake provider
billed at \$1.00/request against a \$2.50 cap must stop after **3** calls.

This is only testable offline because `run_benchmark` accepts any `LLM`. Before
that refactor, exercising the budget guard would have required real spend.

`test_run_benchmark_records_failures_without_cost` pins that failed requests
add \$0 — otherwise a run of pure 429s could bill itself to a halt.

---

## 5. Durability and resume

**Claim:** a killed run resumes instead of repeating work, and a torn write
does not poison the record.

**Proof:** `test_checkpoint_store_resumes_and_appends`,
`test_checkpoint_store_tolerates_a_torn_final_line`,
`test_run_benchmark_skips_request_ids_already_recorded`

The torn-line test writes a deliberately truncated final JSONL line — what a
hard kill mid-write actually leaves behind — and asserts the store skips it
rather than raising. The resume test then proves the runner *acts* on that
state: with two IDs already recorded, only the new ones hit the provider.

---

## 6. Metrics

**Claim:** the numbers in the analysis output are correct.

**Proof:** `test_analyze_computes_goodput_error_rates_and_percentiles` builds a
fixture with known latencies and costs and asserts every derived figure against
hand-computed values — goodput, error-rate breakdown, p50/p95/p99, and cost per
1k useful requests. `test_analyze_with_no_useful_requests_reports_none_cost_per_1k`
covers the divide-by-zero path. `test_save_analysis_writes_summary_next_to_the_jsonl`
pins that the summary is actually persisted next to the raw data.

Classification into the outcome taxonomy is pinned separately by
`test_classify_finish_reason` (including the two Gemini-specific traps: a 200
with empty content, and `MAX_TOKENS` truncation) and `test_classify_exception`.

---

## 7. Provider contract

`tests/test_gemini.py` (17 cases) pins the `Gemini` → `SimpleResponse` mapping:
thinking-budget validation and defaults, output-token limits, and
`finish_reason` normalization to a plain string rather than the SDK enum.

---

## 8. Live checks

Offline tests prove the *logic*. These prove the wiring — credentials,
permissions, and the real Vertex path.

- `pytest -m integration` — one real request, asserts the `SimpleResponse`
  contract holds against the live API (not the wording, which is not
  deterministic).
- `pytest -m loadtest` — three scenarios with different pacing/concurrency/retry
  settings, each hard-capped at 100 requests / \$0.50. Writes one JSONL row per
  request plus an analysis summary to `tests/bench/results/` (gitignored).

The `thin-retry-budget` scenario (`max_attempts=1`) exists to make 429s visible
in the output instead of absorbed by retries — useful for finding where Dynamic
Shared Quota actually pushes back, since the real ceiling is unobservable.

---

## What this does not prove

Worth stating plainly:

- **Throughput at production scale.** The load scenarios cap at 100 requests.
  Nothing here demonstrates a sustained 20,000 rpm.
- **Time to first token.** `RateLimitedLLM` is non-streaming, so latency is
  measured end-to-end only.
- **Backoff timing against a real API.** Jitter and `MAX_BACKOFF_SECONDS`
  saturation are not pinned; the tests assert backoffs grow, not their values.
- **Multi-event-loop safety.** The limiter and `BudgetGuard` drop locks on the
  reasoning that their critical sections contain no `await`. That holds on one
  event loop and is commented at each site, but is not enforced by a test.
