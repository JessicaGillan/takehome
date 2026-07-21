# Gemini 2.5 Flash on Vertex AI: integration and production-readiness findings

The task: add Gemini 2.5 Flash on Vertex as a provider and demonstrably prove
it holds up at production scale. All numbers below come from live runs on
2026-07-21 against one Google Cloud project; raw per-request data is committed
under `tests/bench/results/`, and [VERIFICATION.md](VERIFICATION.md) maps each
claim to the test or data that pins it (§10 for measured results).

## What I built

- [llm/gemini.py](llm/gemini.py) — `Gemini` provider behind the existing
  `LLM` interface, same contract as `Together`.
- [llm/rate_limited_llm.py](llm/rate_limited_llm.py) — provider-agnostic
  wrapper: token-bucket pacing, concurrency cap, bounded retries on 429/503.
- [tests/bench/](tests/bench/) — load harness driving any `LLM` at a target
  rate; classifies every outcome (ok / throttled / truncated / empty /
  safety), enforces a hard dollar budget, checkpoints to JSONL for resume.
- 61 offline tests (~1s, no cost) pin the logic; opt-in pytest markers
  exercise the live API at individually budgeted spend.

## Headline findings

- **~505 useful requests/minute is this project's absolute ceiling** — at the
  price of a 68% rejection rate while attempting 1,800 rpm. Clean,
  throttle-free service tops out around **~270 rpm sustained**.
- **Vertex's Dynamic Shared Quota (DSQ) — a shared regional capacity pool,
  not a fixed per-project limit — is weather, not a contract.** The same
  offered load went from 2% to 71% rejection inside 40 seconds, and there is
  a **6% baseline 429 rate even at 66 rpm** with retries off. Retries are
  mandatory.
- **Latency: fast middle, heavy tail.** p50 1.4–2.5s in every run; p95 9–13s;
  p99 10–11s at light load, 15–19.5s under throttling (worst request 24.3s).
- **Cost:** ~$0.25–0.35 per 1,000 useful answers at a 256-token output cap.

## How it behaves under load

Six live scenarios: three limiter-config load tests plus benchmark tiers
pushing toward 500 / 1,000 / 5,000 rpm (full table: VERIFICATION §10).

- **Clean up to ~139 rpm** — zero throttling, p50 ~1.4–1.6s. But with retries
  disabled, 6% of raw attempts drew a 429 even at 66 rpm: DSQ always sheds a
  few percent, and retry logic silently absorbs it.
- **Capacity curve (concurrency → useful rpm): 16 → 239, 32 → 239,
  160 → 505.** The knee is ~16 concurrent. Doubling 16→32 bought _zero_
  goodput — pressure converted entirely to 429s (21.8%) and latency (success
  mean 2.77s → 4.42s). 32→160 bought 2.1× goodput for 5× the slots; per-slot
  yield collapsed 0.25 → 0.05 ok-requests/sec.
- Throughput matched Little's law (slots ÷ mean latency) within ~10% in every
  run — the curve reflects real capacity, not a harness artifact.
- **Headroom is a range, not a number: plan for 200–350 rpm sustained**
  (bursts to ~750 rpm instantaneous succeed), treat >500 rpm as unavailable
  without quota work, and re-measure at your traffic's time of day.

## Model quirks and parameters that mattered

- **Thinking is on by default and bills like output.** Dynamic thinking
  tokens are invisible in the response but bill at $2.50/Mtok and add
  latency. This integration defaults `thinking_budget=0`, env-overridable.
- **The MAX_TOKENS empty-response trap:** with thinking on, thoughts can eat
  the whole output budget → HTTP 200, `finish_reason=MAX_TOKENS`, no text.
  Even with thinking off, 4–12% of answers truncated at the 256-token cap.
- **Safety/recitation blocks also arrive as 200-with-no-text**, not errors.
  The harness gives them their own failure buckets so they can't masquerade
  as successes (unit-tested; never triggered live by these prompts).
- **Parameters that mattered:** `thinking_budget`, `max_output_tokens`, and
  `GEMINI_MAX_CONNECTIONS` — the SDK's httpx pool silently queues everything
  past its 100-connection default.

## Compared to other LLM providers

Qualitative — the live head-to-head is the Together control run in next steps.

- **The quota model is the biggest difference.** OpenAI/Anthropic/Together
  publish fixed rate limits you provision against; DSQ gives no number at
  all. Fixed quotas mean capacity planning once; DSQ means measuring
  continuously, and the measurement expires.
- **The failure surface is quieter.** Most providers surface content
  suppression as an explicit error or finish reason; Gemini returns 200 with
  no text. A status-code-only integration silently counts these as successes.
- **Billing has a hidden half.** Counting only visible candidate tokens can
  understate the invoice by more than the visible answer's cost when dynamic
  thinking is on — a trap non-reasoning models don't have.
- **Untested hypothesis: temperature is not portable.** The same value need
  not produce the same variance across providers, threatening any cross-model
  comparison at "the same" temperature (determinism experiment below).

## Decisions and tradeoffs

- **Native `system_instruction`** instead of Together's system-as-chat-message
  — same intent, different delivery; a comparability caveat, not a bug.
- **Extend `SimpleResponse`, don't rewrite `LLM`:** `finish_reason` and
  `model_version` added as optionals; other providers untouched.
- **Token accounting sums the invisible half:** `output_tokens` includes
  `thoughts_token_count`, so downstream cost math can't get it wrong.
- **Retries are bounded and pay full fare** — each retry takes a fresh
  rate-limiter token; it's another real request against the shared pool.
- **No circuit breaker on 429:** 429 means _slow down_, a breaker means
  _stop_ — surrendering throughput DSQ was still granting. Breakers belong on
  401/403 and sustained 5xx.
- **Failover never means a substitute model:** the answers _are_ the data.
  Failover = same model in another region, or checkpoint-and-retry.

What didn't work:

- The first runner draft submitted all work before collecting results, so the
  cost cap could never stop a run early. Caught by an offline test.
- The ≥10k-rpm tiers died on macOS's 256-fd soft limit before sending one
  request. Lesson: the client machine is part of the system under test.
- Concurrency was sized on an assumed 1.5s latency; reality was 3–5s, so
  attempted rates hit only 35–55% of target. Lesson: size load generators
  from measured latency, and check attempted rate before trusting goodput.

## Next steps for production (timeboxed, in priority order)

- **Error handling:** add a per-request timeout (a hung request currently
  holds a slot forever); live-exercise the empty/safety/recitation paths.
- **Capacity curve (20 min):** ramp concurrency 1→256, plot goodput and p99,
  set `parallelism()` to the knee.
- **Thinking ablation (15 min):** `thinking_budget ∈ {0, 128, 1024, dynamic}`
  — latency, true cost incl. thoughts, and does the answer set change?
- **Determinism (10 min):** 100 identical calls at temp 0 and 1; tests the
  temperature-portability concern.
- **Together control run (5 min):** same harness, one constructor swap —
  turns every number here into a comparison. Also capture whether Vertex 429s
  carry a `Retry-After` hint; the client currently backs off blindly.
- **Soak (attended):** ~10 min at 80% of the knee, watching 429 drift.

**Adaptive concurrency (AIMD) is the right production controller** — grow the
in-flight limit additively while clean, halve on 429, keep the token bucket
as a hard ceiling. Left out because it needs live tuning and a soak; the
sketch is ~10 lines in `RateLimitedLLM`'s semaphore and retry hook.

Beyond load:

- **Pin, log, and alert on `model_version`** (already recorded per row). A
  silent model bump is a confound for longitudinal measurement.
- **Vertex Batch Prediction is likely the real answer** if traffic is
  batch-shaped — roughly half the price, no rate fight. The entire 429
  struggle above is an artifact of choosing the online API.
- **Can't know from outside:** the real quota ceiling vs DSQ weather; whether
  logprobs are load-bearing for scoring (top-1 requested to match Together);
  provisioned-throughput pricing; and the production traffic shape.

## Bottom line

I'd sign off at **200–350 rpm sustained per project per region**, with
retries as mandatory plumbing and the empty-response buckets monitored — that
envelope was clean in every measurement. Above ~500 useful rpm the gap is
quota, not code: a provisioned-throughput conversation with Google. And if
the traffic is batch-shaped, Batch Prediction sidesteps the fight entirely.

## If this wasn't a code test I'd...

- Use smaller commits
- Set up skills
- Verify and review all work by separate agents with a fresh context
