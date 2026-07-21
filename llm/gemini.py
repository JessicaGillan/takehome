import os

import httpx
from google.genai import Client
from google.genai.types import (
    GenerateContentConfig,
    GenerateContentResponse,
    GenerateContentResponseUsageMetadata,
    HttpOptions,
    ThinkingConfig,
)

from llm import LLM

# Cap on the tokens generated per response when GEMINI_MAX_OUTPUT_TOKENS is unset.
DEFAULT_MAX_OUTPUT_TOKENS = 1000
MIN_MAX_OUTPUT_TOKENS = 1

# thinking_budget adjusts the model's "thinking" capabilities: DISABLED spends no
# thinking tokens, DYNAMIC lets the model size the budget by request complexity, and
# any value in between caps the thinking tokens. The upper bound is 2.5 Flash specific.
THINKING_BUDGET_DISABLED = 0
THINKING_BUDGET_DYNAMIC = -1
MIN_THINKING_BUDGET = 0
MAX_THINKING_BUDGET = 24576
DEFAULT_THINKING_BUDGET = THINKING_BUDGET_DISABLED

# Ceiling on concurrent in-flight requests, reported to callers via parallelism().
# This sizes the underlying httpx connection pool, which is the real client-side
# limit: httpx defaults to 100 connections and quietly queues everything beyond
# that, so a higher target rate needs a bigger pool, not just more callers.
DEFAULT_MAX_CONNECTIONS = 100
MIN_MAX_CONNECTIONS = 1

NUMBER_ALTERNATIVE_TOKEN_OPTIONS = 1

class Gemini(LLM):
    @staticmethod
    def __total_output_tokens(usage: GenerateContentResponseUsageMetadata) -> int:
        """Billed output tokens: visible candidates plus any hidden thinking tokens.

        Both counts are omitted by the API rather than zeroed when not applicable.
        """
        return (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)

    @staticmethod
    def __finish_reason(response: GenerateContentResponse) -> str | None:
        """Why generation stopped, e.g. STOP or MAX_TOKENS.

        Absent when the prompt was blocked before any candidate was produced.
        """
        if not response.candidates:
            return None

        reason = response.candidates[0].finish_reason
        return reason.value if reason is not None else None

    def __init__(self):
        # see the connection-pool constants at the top of the file
        self.__max_connections = int(os.getenv("GEMINI_MAX_CONNECTIONS", DEFAULT_MAX_CONNECTIONS))
        if self.__max_connections < MIN_MAX_CONNECTIONS:
            raise ValueError(f"max_connections {self.__max_connections} must be positive")

        self.__client = Client(
            enterprise=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION"),
            http_options=HttpOptions(
                async_client_args={
                    "limits": httpx.Limits(
                        max_connections=self.__max_connections,
                        max_keepalive_connections=self.__max_connections,
                    ),
                },
            ),
        ).aio
        self.__model = os.getenv("GEMINI_MODEL")

        # limits the maximum number of tokens the model generates in its response
        self.__max_output_tokens = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS))
        if self.__max_output_tokens < MIN_MAX_OUTPUT_TOKENS:
            raise ValueError(f"max_output_tokens {self.__max_output_tokens} must be positive")

        # see the thinking budget constants at the top of the file
        budget = int(os.getenv("GEMINI_THINKING_BUDGET", DEFAULT_THINKING_BUDGET))
        if budget != THINKING_BUDGET_DYNAMIC and not (MIN_THINKING_BUDGET <= budget <= MAX_THINKING_BUDGET):
            raise ValueError(
                f"thinking_budget {budget} outside 2.5 Flash range "
                f"[{MIN_THINKING_BUDGET}, {MAX_THINKING_BUDGET}] (or {THINKING_BUDGET_DYNAMIC})"
            )
        self.__thinking_config = ThinkingConfig(
            thinking_budget=budget,
            include_thoughts=False, # don't return thought summaries as parts
        )

    def parallelism(self) -> int:
        return self.__max_connections

    async def aclose(self) -> None:
        """Release the async client's connection pool. Callers own the lifetime.

        Only the async half is held (see __init__), so Client.close() is not needed.
        """
        await self.__client.aclose()

    async def ask_generic_question(self, system_prompt: str, question: str, temperature: float) -> LLM.SimpleResponse:
        response = await self.__client.models.generate_content(
            model=self.__model,
            contents=question,
            config=GenerateContentConfig(
                system_instruction=system_prompt,
                # controls creativity (0.0 to 2.0): lower is more deterministic,
                # higher is more creative
                temperature=temperature,
                max_output_tokens=self.__max_output_tokens,
                # see the thinking budget rules in __init__
                thinking_config=self.__thinking_config,
                response_logprobs=True,  # Enables the logprobs feature
                logprobs=NUMBER_ALTERNATIVE_TOKEN_OPTIONS # Returns 1 top alternative option (matching Together functionality)
            ),
        )

        usage = response.usage_metadata

        return LLM.SimpleResponse(
            answer=response.text,
            input_tokens=usage.prompt_token_count,
            output_tokens=self.__total_output_tokens(usage),
            finish_reason=self.__finish_reason(response),
            model_version=response.model_version,
        )
