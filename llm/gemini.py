import os

from dotenv import load_dotenv
from google.genai import Client
from google.genai.types import (
    GenerateContentConfig,
    GenerateContentResponseUsageMetadata,
    ThinkingConfig,
)

from llm import LLM

load_dotenv()  # reads .env into os.environ; existing env vars win by default

class Gemini(LLM):
    def __init__(self):
        self.__client = Client(
            enterprise=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION"),
        ).aio
        self.__model = os.getenv("GEMINI_MODEL")

        # limits the maximum number of tokens the model generates in its response
        self.__max_output_tokens = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", 10))
        if self.__max_output_tokens < 1:
            raise ValueError(f"max_output_tokens {self.__max_output_tokens} must be positive")

        # thinking_budget adjusts the model's "thinking" capabilities: 0 disables thinking,
        # -1 turns on dynamic thinking where the model adjusts the budget based on request
        # complexity, and a positive number caps the thinking tokens. For 2.5 Flash
        # specifically, the valid range is 0 to 24,576.
        budget = int(os.getenv("GEMINI_THINKING_BUDGET", 0))
        if budget != -1 and not (0 <= budget <= 24576):
            raise ValueError(f"thinking_budget {budget} outside 2.5 Flash range [0, 24576] (or -1)")
        self.__thinking_config = ThinkingConfig(
            thinking_budget=budget,
            include_thoughts=False, # don't return thought summaries as parts
        )

    def parallelism(self) -> int:
        return 100

    async def aclose(self) -> None:
        """Release the async client's connection pool. Callers own the lifetime.

        Only the async half is held (see __init__), so Client.close() is not needed.
        """
        await self.__client.aclose()

    @staticmethod
    def __total_output_tokens(usage: GenerateContentResponseUsageMetadata) -> int:
        """Billed output tokens: visible candidates plus any hidden thinking tokens.

        Both counts are omitted by the API rather than zeroed when not applicable.
        """
        return (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)

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
            ),
        )

        usage = response.usage_metadata

        return LLM.SimpleResponse(
            answer=response.text,
            input_tokens=usage.prompt_token_count,
            output_tokens=self.__total_output_tokens(usage),
        )
