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

    async def ask_generic_question(self, system_prompt: str, question: str, temperature: float) -> SimpleResponse:
        raise NotImplementedError()

    def parallelism(self):
        raise NotImplementedError()