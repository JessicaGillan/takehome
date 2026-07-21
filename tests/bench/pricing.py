"""Vertex pricing for gemini-2.5-flash. Defaults verified mid-2026; volatile.

Thinking tokens bill at the output rate, and LLM.SimpleResponse.output_tokens
already includes them (see Gemini.__total_output_tokens), so a single output
count prices the whole response.
"""

from dataclasses import dataclass

@dataclass(frozen=True)
class Pricing:
    input_per_mtok_usd: float = 0.30
    output_per_mtok_usd: float = 2.50  # candidates + thoughts are billed here

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_per_mtok_usd
            + output_tokens * self.output_per_mtok_usd
        ) / 1_000_000
