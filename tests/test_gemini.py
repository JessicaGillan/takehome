from types import SimpleNamespace

import pytest
from google.genai.types import Candidate, FinishReason, GenerateContentResponse

from llm import LLM, Gemini
from llm import gemini as gemini_module
from llm.gemini import (
    DEFAULT_MAX_CONNECTIONS,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_THINKING_BUDGET,
    MAX_THINKING_BUDGET,
    THINKING_BUDGET_DISABLED,
    THINKING_BUDGET_DYNAMIC,
)

@pytest.mark.parametrize(
    "budget",
    [
        str(THINKING_BUDGET_DYNAMIC),
        str(THINKING_BUDGET_DISABLED),
        str(MAX_THINKING_BUDGET),
    ],
)
def test_valid_budgets_reach_thinking_config(monkeypatch: pytest.MonkeyPatch, budget: str) -> None:
    monkeypatch.setenv("GEMINI_THINKING_BUDGET", budget)

    client = Gemini()

    config = client._Gemini__thinking_config
    assert config.thinking_budget == int(budget)
    assert config.include_thoughts is False

@pytest.mark.parametrize("budget", [str(MAX_THINKING_BUDGET + 1), "-2"])
def test_out_of_range_budget_is_rejected(monkeypatch: pytest.MonkeyPatch, budget: str) -> None:
    monkeypatch.setenv("GEMINI_THINKING_BUDGET", budget)

    with pytest.raises(ValueError, match=r"outside 2\.5 Flash range"):
        Gemini()

def test_budget_defaults_to_disabled_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_THINKING_BUDGET", raising=False)

    client = Gemini()

    assert client._Gemini__thinking_config.thinking_budget == DEFAULT_THINKING_BUDGET

def test_max_output_tokens_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "512")

    assert Gemini()._Gemini__max_output_tokens == 512

def test_max_output_tokens_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_MAX_OUTPUT_TOKENS", raising=False)

    assert Gemini()._Gemini__max_output_tokens == DEFAULT_MAX_OUTPUT_TOKENS

@pytest.mark.parametrize("value", ["0", "-1"])
def test_non_positive_max_output_tokens_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", value)

    with pytest.raises(ValueError, match="must be positive"):
        Gemini()

@pytest.mark.parametrize(
    "reason,expected",
    [
        (FinishReason.STOP, "STOP"),
        (FinishReason.MAX_TOKENS, "MAX_TOKENS"),  # response hit the output cap
        (FinishReason.SAFETY, "SAFETY"),
    ],
)
def test_finish_reason_is_returned_as_plain_string(reason: FinishReason, expected: str) -> None:
    response = GenerateContentResponse(candidates=[Candidate(finish_reason=reason)])

    result = Gemini._Gemini__finish_reason(response)

    assert result == expected
    assert type(result) is str  # not the SDK enum, which subclasses str

@pytest.mark.parametrize(
    "candidates",
    [
        None,                                # prompt blocked before generation
        [],                                  # no candidate returned
        [Candidate(finish_reason=None)],     # candidate without a reason
    ],
)
def test_missing_finish_reason_is_none(candidates: list[Candidate] | None) -> None:
    response = GenerateContentResponse(candidates=candidates)

    assert Gemini._Gemini__finish_reason(response) is None

def test_simple_response_finish_reason_defaults_to_none() -> None:
    # Together and any other provider construct SimpleResponse without it.
    response = LLM.SimpleResponse(answer="hi", input_tokens=1, output_tokens=2)

    assert response.finish_reason is None

def test_max_connections_defaults_and_reaches_parallelism(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_MAX_CONNECTIONS", raising=False)

    assert Gemini().parallelism() == DEFAULT_MAX_CONNECTIONS

def test_max_connections_env_reaches_the_connection_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_MAX_CONNECTIONS", "640")
    captured: dict = {}

    def recording_client(**kwargs) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(aio="aio-half")

    monkeypatch.setattr(gemini_module, "Client", recording_client)

    client = Gemini()

    limits = captured["http_options"].async_client_args["limits"]
    assert limits.max_connections == 640
    assert limits.max_keepalive_connections == 640
    assert client.parallelism() == 640

@pytest.mark.parametrize("value", ["0", "-1"])
def test_non_positive_max_connections_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("GEMINI_MAX_CONNECTIONS", value)

    with pytest.raises(ValueError, match="must be positive"):
        Gemini()
