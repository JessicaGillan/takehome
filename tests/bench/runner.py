"""Drives requests through a RateLimitedLLM and records one JSONL row each.

Pacing, concurrency, and retries are RateLimitedLLM's job (see
llm/rate_limited_llm.py); this module only decides *when to stop*
(BudgetGuard), turns each response into a RequestResult, and persists it.
"""

import asyncio
import time
from typing import Optional

from llm.llm import LLM

from .pricing import Pricing
from .prompts import PROMPTS
from .results import RequestResult, classify_exception, classify_finish_reason
from .store import CheckpointStore

# Load shape held fixed across scenarios so runs stay comparable: only the
# rate-limit settings vary between them.
SYSTEM_PROMPT = "Answer concisely."
TEMPERATURE = 0.7

class BudgetGuard:
    """Hard stop for a run: refuses new work once either cap is hit.

    No lock: `try_reserve` and `record_cost` contain no `await`, so on a single
    event loop neither can be interrupted part-way. Worst-case overshoot is
    bounded by however many requests are in flight when a cap trips.
    """

    def __init__(self, max_requests: int, max_cost_usd: float):
        self._max_requests = max_requests
        self._max_cost = max_cost_usd
        self.requests_started = 0
        self.spent_usd = 0.0

    def try_reserve(self) -> bool:
        if self.requests_started >= self._max_requests or self.spent_usd >= self._max_cost:
            return False
        self.requests_started += 1
        return True

    def record_cost(self, cost_usd: float) -> None:
        self.spent_usd += cost_usd

async def _run_one(
    request_id: str,
    prompt_id: str,
    prompt: str,
    client: LLM,
    pricing: Pricing,
) -> RequestResult:
    started_unix = time.time()
    started = time.monotonic()

    try:
        response = await client.ask_generic_question(SYSTEM_PROMPT, prompt, TEMPERATURE)
    except Exception as error:
        return RequestResult(
            request_id=request_id, prompt_id=prompt_id, error_class=classify_exception(error).value,
            model_version=None, finish_reason=None, input_tokens=0, output_tokens=0,
            total_latency_s=time.monotonic() - started, cost_usd=0.0,
            start_unix=started_unix, end_unix=time.time(),
        )

    return RequestResult(
        request_id=request_id, prompt_id=prompt_id,
        error_class=classify_finish_reason(response.finish_reason, bool(response.answer)).value,
        model_version=response.model_version, finish_reason=response.finish_reason,
        input_tokens=response.input_tokens, output_tokens=response.output_tokens,
        total_latency_s=time.monotonic() - started,
        cost_usd=pricing.cost_usd(response.input_tokens, response.output_tokens),
        start_unix=started_unix, end_unix=time.time(),
    )

async def run_benchmark(
    client: LLM,
    store: CheckpointStore,
    pricing: Pricing,
    max_requests: int,
    max_cost_usd: float,
    max_in_flight: int = 8,
) -> BudgetGuard:
    """Drive up to `max_requests` requests, stopping early if `max_cost_usd` is hit.

    Launching and result processing are interleaved (a sliding window of at
    most `max_in_flight` tasks), not "launch everything, then collect" — the
    cost cap can only stop a run early if `spent_usd` is updated *while* new
    requests are still being considered.
    """
    budget = BudgetGuard(max_requests, max_cost_usd)

    def start_next() -> Optional[asyncio.Task]:
        while budget.try_reserve():
            index = budget.requests_started - 1
            request_id = f"req-{index:04d}"
            if request_id in store.completed_ids:
                continue  # resume: already recorded in a prior run
            prompt_id = f"p{index % len(PROMPTS):02d}"
            return asyncio.create_task(
                _run_one(request_id, prompt_id, PROMPTS[index % len(PROMPTS)], client, pricing)
            )
        return None

    in_flight = {task for task in (start_next() for _ in range(max_in_flight)) if task}
    while in_flight:
        done, in_flight = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            result = task.result()
            budget.record_cost(result.cost_usd)
            store.append(result)
            next_task = start_next()
            if next_task:
                in_flight.add(next_task)

    return budget
