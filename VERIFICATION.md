# Verification

Where the proof lives. Each section names the behavior, the test that pins it,
and — more importantly — _why_ that test would fail if the behavior broke. A
test that passes on both the working and the broken implementation proves
nothing, so the reasoning below is written in terms of what each assertion
rules out.

```bash
.venv/bin/python -m pytest              # 61 offline tests, ~1s, no cost
.venv/bin/python -m pytest -m integration   # 1 live request
.venv/bin/python -m pytest -m loadtest      # 3 live load scenarios, up to $1.50
.venv/bin/python -m pytest -m benchmark     # 3 production-rate tiers, budgets sum $3.25
```

The live markers are excluded from every other command, including a bare
`pytest` and an explicit path to their file. See `CLAUDE.md`.

---

## 1. Concurrency

**Claim:** `RateLimitedLLM` sends requests _concurrently_, and never has more
than `max_concurrent` of them in flight at once.

**Proof:** `tests/test_rate_limited_llm.py::test_max_concurrent_is_never_exceeded`

The limiter holds an `asyncio.Semaphore(max_concurrent)`
(`llm/rate_limited_llm.py:113`) and acquires it around the delegate call
(`llm/rate_limited_llm.py:144`). The test measures the real thing rather than
inspecting the semaphore: a fake provider increments a counter on entry,
records the high-water mark, awaits, then decrements. Whatever peak that
counter reaches _is_ the true simultaneous in-flight count.

Two assertions, and it takes both:

| Assertion                | Failure it rules out                                                                                       |
| ------------------------ | ---------------------------------------------------------------------------------------------------------- |
| `peak <= max_concurrent` | The cap is missing or too loose — requests running unbounded.                                              |
| `peak == max_concurrent` | The opposite bug: **accidental serialization**. If the limiter ran one request at a time, peak would be 1. |

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
the cap _tracks the argument_. `max_concurrent=1` pulls double duty as the
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
_exact equalities_ rather than tolerances, and the suite stays fast. Ten
acquires at rate _r_ with `capacity=1` must consume precisely `9/r` seconds —
the first token is free, the other nine each wait a full refill interval.
Parametrized over three rates so the arithmetic is shown to scale rather than
matching one lucky number.

The clock is installed by swapping the `asyncio` _name_ inside the module under
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

| Test                                               | Pins                                                                                                                                                                                      |
| -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `test_max_attempts_bounds_retries`                 | Exactly `max_attempts` calls, then the error propagates. Parametrized `[1, 3, 5]`, so the never-retry edge is covered. Also checks `max_attempts - 1` backoffs, monotonically increasing. |
| `test_retryable_errors_are_retried_until_success`  | 429 and 503 recover on the next attempt.                                                                                                                                                  |
| `test_non_retryable_errors_are_raised_immediately` | 400/403/404/500 → exactly 1 call.                                                                                                                                                         |
| `test_errors_without_a_status_are_not_retried`     | A plain `ValueError` surfaces once, not five times.                                                                                                                                       |

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

**Claim:** a load run stops at its request cap _and_ its dollar cap.

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
rather than raising. The resume test then proves the runner _acts_ on that
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

Offline tests prove the _logic_. These prove the wiring — credentials,
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

## 9. Production-rate benchmarks

**Claim:** the provider withstands a stated request rate.

**Proof:** `tests/test_bench_benchmark.py::test_sustains_target_rpm`, three
tiers, each a 30-second window at its target rate with its own request cap and
dollar budget:

| tier   | target rpm | requests | budget | worst-case cost |
| ------ | ---------- | -------- | ------ | --------------- |
| 500rpm | 500        | 250      | $0.25  | $0.17           |
| 1k     | 1,000      | 500      | $0.50  | $0.33           |
| 5k     | 5,000      | 2,500    | $2.50  | $1.63           |

Full sweep: 3,250 requests, budgets summing **$3.25**, worst-case actual
**~$2.13**, typical **~$1.30** (all tiers cap output at 256 tokens — output
tokens dominate cost at $2.50/Mtok). Budgets are a safety backstop sized above
worst case: if a cost cap fires, the run was truncated and the rate assertions
fail, which is the correct verdict.

**Why there is no tier above 5k.** Tiers at 10k/20k/30k rpm existed and were
removed after the 2026-07-21 run (section 10): the provider's quota ceiling
measured ~505 useful rpm, with 68% of attempts rejected at ~1,800 rpm — the
removed targets are unreachable by 10-60× until quota changes, so rerunning
them only spends money re-confirming a known refusal. Getting there is a
quota / provisioned-throughput conversation with Google, not a harness
setting. For whoever re-adds them: those tiers also need 320-960 concurrent
sockets, which exceeds the default macOS file-descriptor soft limit of 256 —
they crashed with `OSError: Too many open files` before sending a single
request. Raise `ulimit -n` (or `resource.setrlimit`) first.

Three assertions per tier, and the split matters:

| Assertion                       | Failure it rules out                                                    |
| ------------------------------- | ----------------------------------------------------------------------- |
| `attempted_rpm >= 0.9 × target` | The _harness_ fell behind — the tier never applied its load.            |
| `goodput_rpm >= 0.9 × target`   | The _provider_ buckled — 429s, empties, or latency ate the useful rate. |
| `spent <= budget + tolerance`   | The budget guard failed open.                                           |

Without the first assertion, a passing goodput number would be meaningless —
you can't fail to withstand load that was never applied. A failing tier is the
deliverable: it marks where DSQ stops absorbing, and the saved
`*.analysis.json` shows the error mix and latency percentiles that broke it.

Supporting production change, pinned offline: high tiers need more in-flight
requests than the SDK's httpx pool default (100 connections) allows, so
`GEMINI_MAX_CONNECTIONS` sizes the pool
(`test_max_connections_env_reaches_the_connection_pool` asserts the
`httpx.Limits` object actually reaches the client, not just that the env var
is read).

---

## 10. Measured results — 2026-07-21 snapshot

What actually happened when the live suites ran, in plain language. All six
runs against one Google Cloud project on a single afternoon; every number
below comes from the saved `*.analysis.json` files (the raw
`tests/bench/results/` files are gitignored, so these tables are the durable
record). Latency is end-to-end **including retries**, so a row that succeeded
on its second attempt carries its first attempt and the backoff in between.

| run          | limiter (rps / burst / conc / attempts) | attempted rpm | useful rpm | ok %  | 429 %    | truncated % | p50 / p95 latency | cost   |
| ------------ | --------------------------------------- | ------------- | ---------- | ----- | -------- | ----------- | ----------------- | ------ |
| conservative | 2 / 2 / 2 / 5                           | 36*           | 39*        | 100   | 0        | 0           | 1.63s / 9.79s     | $0.035 |
| bursty       | 8 / 16 / 8 / 5                          | 139           | 139        | 100   | 0        | 0           | 1.39s / 9.21s     | $0.030 |
| thin-retry   | 4 / 4 / 4 / 1                           | 66            | 62         | 93    | **6**    | 1           | 1.50s / 8.85s     | $0.031 |
| 500rpm       | 8.3 / 9 / 16 / 3                        | 270           | 239        | 88    | 0        | 12          | 1.48s / 9.68s     | $0.056 |
| 1k           | 16.7 / 17 / 32 / 3                      | 348           | 239        | 68.4  | 21.8     | 9.8         | 2.46s / 12.58s    | $0.091 |
| 5k           | 83.3 / 84 / 160 / 3                     | 1,812         | **505**    | 27.9  | **68.2** | 3.9         | 2.37s / 10.99s    | $0.180 |

(*conservative's 36 vs 39 is rounding in the saved file — every request
succeeded, so attempted and useful are both really ~38.5 rpm.)

### Reading the table

**The light-load baseline (first three rows).** At up to ~139 rpm the provider
was flawless: 200/200 requests succeeded, zero throttling, median latency
~1.4-1.6s. The `thin-retry` row is the revealing one — with retries disabled
it exposes the *raw* rejection rate that retries normally hide: **6% of
attempts drew a 429 even at 66 rpm**. Dynamic Shared Quota sheds a few percent
of load at all times; a production client must retry as a matter of course,
not as an exception.

**The ceiling discovery (last three rows).** Pushing harder bought almost
nothing. The 500rpm tier (270 rpm attempted) was still 429-free. The 1k tier
attempted only ~28% more and got 21.8% of attempts rejected. The 5k tier
attempted 5× more and got 68.2% rejected. Useful throughput topped out at
**505 rpm — the most this project's quota delivered all afternoon** — and
paying for it meant two of every three requests bouncing.

**Throttling is time-structured, not uniform.** 429 share by quarter of each
run:

| run | Q1    | Q2    | Q3    | Q4        |
| --- | ----- | ----- | ----- | --------- |
| 1k  | 1.8%  | 3.3%  | 71.4% | 12.2%     |
| 5k  | 54.9% | 81.5% | 51.2% | **98.7%** |

The 1k run sailed for ~40 seconds — including an opening burst of ~768 rpm
instantaneous — and then DSQ slammed to 71% rejection before partially
recovering. The 5k run (~5,160 rpm instantaneous at open) was throttled from
the first seconds and ended at near-total rejection. Same offered load, very
different treatment minutes apart: the quota is a moving target.

**Concurrency does not buy throughput.** Achieved concurrency was pinned at
the cap for the entire run in all three benchmark tiers, and throughput obeyed
Little's law (throughput = slots ÷ mean latency) almost exactly: 16 ÷ 3.08s
predicts 5.2 rps, measured 4.5; 160 ÷ 5.10s predicts 31.4, measured 30.2.
Doubling slots 16 → 32 bought **zero** additional useful rpm (239 → 239): the
extra pressure converted entirely into 429s and higher latency (ok-row mean
2.77s → 4.42s). Going 32 → 160 bought 2.1× the goodput for 5× the slots, with
per-slot yield collapsing from 0.25 to 0.05 useful requests/second.

### The takeaway

Under this project's quota as measured that afternoon, gemini-2.5-flash **can**
support:

- **~270 rpm sustained** (0% throttling at 16 concurrent), and short bursts to
  ~750 rpm instantaneous;
- **~505 useful rpm absolute maximum**, and only by attempting 1,800 rpm and
  eating a 68% rejection rate;
- median latency **1.4-1.6s** at light load, **~2.4s** under pressure, with a
  heavy tail: p95 ~9-13s, p99 16-19.5s, worst observed 24.3s;
- useful concurrency up to **~16 in-flight**; beyond that, added slots mostly
  buy rejections.

It **cannot**:

- sustain **350-450 rpm for longer than ~40 seconds** without mass throttling
  (71% rejection);
- deliver fully clean service at any measured rate without retries (6% raw
  429 floor at 66 rpm);
- approach the original production targets: 5k rpm useful would need ~10× the
  best observed goodput, 20k ~40×, 30k ~60×. **That gap is quota, not code** —
  closing it means provisioned throughput or a quota increase from Google, and
  only then re-adding the removed tiers.

### Limitations of this measurement

1. **Snapshot, not soak.** Windows were 30-156 seconds on one afternoon.
   Nothing here covers hours-long stability, time-of-day or day-of-week quota
   variation, connection churn, or client memory under sustained load.
2. **The rate limit itself is variable.** DSQ capacity floats with global
   demand — the 1k run alone went from 2% to 71% rejection at constant offered
   load. These numbers are a weather report, not a contract.
3. **The harness undersized concurrency.** Caps were sized for an assumed
   1.5s latency; measured means were 3-5s, so attempted rates reached only
   35-55% of their targets. "500 rpm sustained" was never actually applied —
   270 rpm was. The 270-1,800 rpm band is coarsely sampled.
4. **Truncation is an artifact of the cost lever.** The 256-token benchmark
   cap turned 4-12% of real answers into `truncated` failures; production's
   1000-token default would shift cost, latency, and goodput.
5. **Nothing above 1,812 rpm attempted was ever observed.** The 10k-30k tiers
   died on the macOS 256-fd soft limit before their first request.
6. **One project, one region.** DSQ is per-project-per-region; a second region
   or project would have its own ceiling.
7. **Synthetic uniform traffic.** Sixteen short, listy prompts cycled
   repeatedly — no long context, no production mix, and repeated prompts may
   benefit from server-side caching.
8. **Fixed retry policy, no adaptive rate control.** Three attempts with
   exponential backoff; a production client that lowered its *send* rate on
   429s (AIMD-style) would see a different goodput/rejection mix at the same
   quota.
9. **Client-side measurement only.** Latencies include local event-loop
   scheduling under up-to-160-way concurrency; there is no server-side truth
   to separate provider time from client time.

---

## What this does not prove

Worth stating plainly:

- **Sustained production throughput on every run.** The benchmark tiers reach
  for 500-5k rpm, but they are opt-in and billable — a green offline suite says
  nothing about live behavior, and what the live runs did measure (section 10)
  is a dated snapshot with the limitations listed there.
- **Time to first token.** `RateLimitedLLM` is non-streaming, so latency is
  measured end-to-end only.
- **Backoff timing against a real API.** Jitter and `MAX_BACKOFF_SECONDS`
  saturation are not pinned; the tests assert backoffs grow, not their values.
- **Multi-event-loop safety.** The limiter and `BudgetGuard` drop locks on the
  reasoning that their critical sections contain no `await`. That holds on one
  event loop and is commented at each site, but is not enforced by a test.
- **Hung-request resilience.** There is no per-request timeout, so a request
  that never returns holds a concurrency slot indefinitely.
