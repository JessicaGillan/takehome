"""Offline tests for the tests/bench package: no network, no cost.

Because run_benchmark takes any LLM, the runner's budget caps and resume logic
are covered here with a fake provider rather than real spend. The live
scenarios are in test_bench_loadtest.py, gated behind `-m loadtest`.
"""

import json
from pathlib import Path

import pytest
from google.genai.errors import APIError

from llm.llm import LLM
from tests.bench.analyze import analyze, save_analysis
from tests.bench.pricing import Pricing
from tests.bench.results import (
    ErrorClass,
    RequestResult,
    classify_exception,
    classify_finish_reason,
)
from tests.bench.runner import run_benchmark
from tests.bench.store import CheckpointStore

def _result(request_id: str, error_class: str, latency_s: float, cost_usd: float, end_unix: float) -> RequestResult:
    return RequestResult(
        request_id=request_id, prompt_id="p00", error_class=error_class,
        model_version="gemini-2.5-flash-001", finish_reason="STOP",
        input_tokens=10, output_tokens=20,
        total_latency_s=latency_s, cost_usd=cost_usd, start_unix=0.0, end_unix=end_unix,
    )

class FakeLLM(LLM):
    """Returns a fixed-size response, or raises `error` on every call."""

    def __init__(self, output_tokens: int = 20, error: BaseException | None = None) -> None:
        self._output_tokens = output_tokens
        self._error = error
        self.calls = 0

    async def ask_generic_question(
        self, system_prompt: str, question: str, temperature: float
    ) -> LLM.SimpleResponse:
        self.calls += 1
        if self._error is not None:
            raise self._error
        return LLM.SimpleResponse(
            answer="an answer", input_tokens=10, output_tokens=self._output_tokens,
            finish_reason="STOP", model_version="fake-1.0",
        )

@pytest.mark.parametrize(
    "reason,has_text,expected",
    [
        ("STOP", True, ErrorClass.OK),
        ("STOP", False, ErrorClass.EMPTY),        # 200 with no text — a swallowed block
        ("SAFETY", False, ErrorClass.SAFETY),
        ("PROHIBITED_CONTENT", False, ErrorClass.SAFETY),
        ("RECITATION", False, ErrorClass.RECITATION),
        ("MAX_TOKENS", True, ErrorClass.TRUNCATED),
        (None, False, ErrorClass.EMPTY),          # prompt blocked before any candidate
    ],
)
def test_classify_finish_reason(reason: str | None, has_text: bool, expected: ErrorClass) -> None:
    assert classify_finish_reason(reason, has_text) is expected

def test_checkpoint_store_resumes_and_appends(tmp_path: Path) -> None:
    path = str(tmp_path / "run.jsonl")
    first = CheckpointStore(path)
    assert first.completed_ids == set()  # nothing written yet

    first.append(_result("req-0000", ErrorClass.OK.value, 1.0, 0.001, 1.0))
    first.append(_result("req-0001", ErrorClass.OK.value, 2.0, 0.002, 2.0))

    resumed = CheckpointStore(path)  # simulates restarting after a kill

    assert resumed.completed_ids == {"req-0000", "req-0001"}
    resumed.append(_result("req-0002", ErrorClass.OK.value, 1.5, 0.001, 3.0))
    assert resumed.completed_ids == {"req-0000", "req-0001", "req-0002"}
    assert sum(1 for _ in open(path)) == 3

def test_checkpoint_store_tolerates_a_torn_final_line(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text(_result("req-0000", ErrorClass.OK.value, 1.0, 0.001, 1.0).to_jsonl() + "\n" + '{"request_id": "req-000')

    store = CheckpointStore(str(path))

    assert store.completed_ids == {"req-0000"}  # the torn line is skipped, not fatal

def test_analyze_computes_goodput_error_rates_and_percentiles(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    rows = [
        _result("req-0000", ErrorClass.OK.value, 1.0, 0.001, 1.0),
        _result("req-0001", ErrorClass.OK.value, 3.0, 0.002, 3.0),
        _result("req-0002", ErrorClass.RATE_LIMIT.value, 0.1, 0.0, 2.0),
        _result("req-0003", ErrorClass.SAFETY.value, 0.1, 0.0, 2.0),
    ]
    path.write_text("\n".join(r.to_jsonl() for r in rows) + "\n")

    summary = analyze(str(path))

    assert summary["requests"] == 4
    assert summary["useful"] == 2
    assert summary["wall_seconds"] == pytest.approx(3.0)
    assert summary["goodput_rpm"] == pytest.approx(40.0)  # 2 useful / 3s * 60
    assert summary["error_rates"] == {"ok": 0.5, "429": 0.25, "safety": 0.25}
    assert summary["latency_s"]["p50"] == pytest.approx(2.0)   # midpoint of [1.0, 3.0]
    assert summary["latency_s"]["p99"] == pytest.approx(2.98)
    assert summary["total_cost_usd"] == pytest.approx(0.003)
    assert summary["cost_per_1k_useful_usd"] == pytest.approx(1.5)

def test_analyze_with_no_useful_requests_reports_none_cost_per_1k(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text(_result("req-0000", ErrorClass.SERVER.value, 0.1, 0.0, 1.0).to_jsonl() + "\n")

    summary = analyze(str(path))

    assert summary["useful"] == 0
    assert summary["cost_per_1k_useful_usd"] is None

def test_save_analysis_writes_summary_next_to_the_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    path.write_text(_result("req-0000", ErrorClass.OK.value, 1.0, 0.001, 1.0).to_jsonl() + "\n")

    summary = save_analysis(str(path))

    saved = json.loads((tmp_path / "run.analysis.json").read_text())
    assert saved == summary

@pytest.mark.parametrize(
    "error,expected",
    [
        (APIError(429, {}), ErrorClass.RATE_LIMIT),        # google-genai `.code`
        (APIError(503, {}), ErrorClass.SERVER),
        (TimeoutError("request timed out"), ErrorClass.TIMEOUT),  # no status at all
    ],
)
def test_classify_exception(error: BaseException, expected: ErrorClass) -> None:
    assert classify_exception(error) is expected

async def test_run_benchmark_stops_at_max_requests(tmp_path: Path) -> None:
    delegate = FakeLLM()
    store = CheckpointStore(str(tmp_path / "run.jsonl"))

    budget = await run_benchmark(
        delegate, store, Pricing(), max_requests=5, max_cost_usd=1.0, max_in_flight=2,
    )

    assert delegate.calls == 5
    assert budget.requests_started == 5
    assert len(store.completed_ids) == 5

async def test_run_benchmark_stops_early_once_the_cost_cap_is_hit(tmp_path: Path) -> None:
    # 400k output tokens at $2.50/Mtok = $1.00 per request, so a $2.50 cap must
    # stop after the third — proving cost is reconciled *while* new requests are
    # still being launched, not only after every request has been submitted.
    delegate = FakeLLM(output_tokens=400_000)
    store = CheckpointStore(str(tmp_path / "run.jsonl"))

    budget = await run_benchmark(
        delegate, store, Pricing(), max_requests=100, max_cost_usd=2.50, max_in_flight=1,
    )

    assert delegate.calls == 3
    assert budget.spent_usd == pytest.approx(3 * Pricing().cost_usd(10, 400_000))

async def test_run_benchmark_skips_request_ids_already_recorded(tmp_path: Path) -> None:
    store = CheckpointStore(str(tmp_path / "run.jsonl"))
    for request_id in ("req-0000", "req-0001"):
        store.append(_result(request_id, ErrorClass.OK.value, 1.0, 0.001, 1.0))
    delegate = FakeLLM()

    await run_benchmark(
        delegate, store, Pricing(), max_requests=4, max_cost_usd=1.0, max_in_flight=2,
    )

    assert delegate.calls == 2  # only req-0002 and req-0003 are new
    assert store.completed_ids == {"req-0000", "req-0001", "req-0002", "req-0003"}

async def test_run_benchmark_records_failures_without_cost(tmp_path: Path) -> None:
    delegate = FakeLLM(error=APIError(429, {}))
    store = CheckpointStore(str(tmp_path / "run.jsonl"))

    budget = await run_benchmark(
        delegate, store, Pricing(), max_requests=3, max_cost_usd=1.0, max_in_flight=1,
    )

    assert budget.spent_usd == 0.0
    summary = analyze(str(tmp_path / "run.jsonl"))
    assert summary["error_rates"] == {"429": 1.0}
    assert summary["useful"] == 0
