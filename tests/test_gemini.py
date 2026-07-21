import pytest

from llm import Gemini

@pytest.mark.parametrize(
    "budget",
    [
        "-1",     # dynamic thinking
        "0",      # thinking disabled
        "24576",  # upper bound for 2.5 Flash
    ],
)
def test_valid_budgets_reach_thinking_config(monkeypatch: pytest.MonkeyPatch, budget: str) -> None:
    monkeypatch.setenv("GEMINI_THINKING_BUDGET", budget)

    client = Gemini()

    config = client._Gemini__thinking_config
    assert config.thinking_budget == int(budget)
    assert config.include_thoughts is False

@pytest.mark.parametrize("budget", ["24577", "-2"])
def test_out_of_range_budget_is_rejected(monkeypatch: pytest.MonkeyPatch, budget: str) -> None:
    monkeypatch.setenv("GEMINI_THINKING_BUDGET", budget)

    with pytest.raises(ValueError, match=r"outside 2\.5 Flash range"):
        Gemini()

def test_budget_defaults_to_disabled_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_THINKING_BUDGET", raising=False)

    client = Gemini()

    assert client._Gemini__thinking_config.thinking_budget == 0

def test_max_output_tokens_read_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "512")

    assert Gemini()._Gemini__max_output_tokens == 512

def test_max_output_tokens_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_MAX_OUTPUT_TOKENS", raising=False)

    assert Gemini()._Gemini__max_output_tokens == 10

@pytest.mark.parametrize("value", ["0", "-1"])
def test_non_positive_max_output_tokens_is_rejected(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", value)

    with pytest.raises(ValueError, match="must be positive"):
        Gemini()
