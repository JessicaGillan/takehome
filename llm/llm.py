from dataclasses import dataclass

class LLM:
    @dataclass
    class SimpleResponse:
        answer: str
        input_tokens: int
        output_tokens: int
        # why generation stopped, e.g. STOP or MAX_TOKENS; None if the provider
        # does not report one
        finish_reason: str | None = None
        # the provider's served model build; None if it does not report one
        model_version: str | None = None

    async def ask_generic_question(self, system_prompt: str, question: str, temperature: float) -> SimpleResponse:
        raise NotImplementedError()

    def parallelism(self):
        raise NotImplementedError()

    async def aclose(self) -> None:
        """Release any connection pool. Providers holding none need not override."""
        return None