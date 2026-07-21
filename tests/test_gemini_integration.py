"""Live API tests. Deselected by default; run with `pytest -m integration`.

These make real, billable calls to the project in GOOGLE_CLOUD_PROJECT and
require application default credentials.
"""
import pytest

from llm import Gemini

@pytest.mark.integration
async def test_ask_generic_question_against_live_api() -> None:
    client = Gemini()

    try:
        response = await client.ask_generic_question(
            system_prompt="Answer with a single word.",
            question="What is the capital of France?",
            temperature=0.0,
        )
    finally:
        await client.aclose()

    # Assert the SimpleResponse contract, not the wording, which is not deterministic.
    assert isinstance(response.answer, str)
    assert response.answer.strip()
    assert response.input_tokens > 0
    assert response.output_tokens > 0
