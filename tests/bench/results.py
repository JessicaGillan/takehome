"""Outcome taxonomy and the per-request record.

One request maps to exactly one `ErrorClass`. Only OK is "useful" — everything
else is a way throughput can be lost, which is the whole point of measuring it.
"""

import json
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

from llm.rate_limited_llm import status_code

class ErrorClass(str, Enum):
    OK = "ok"
    RATE_LIMIT = "429"
    SERVER = "5xx"
    TIMEOUT = "timeout"
    SAFETY = "safety"
    RECITATION = "recitation"
    EMPTY = "empty"          # 200 with no usable text — a swallowed block
    TRUNCATED = "truncated"  # hit the output token ceiling

def classify_finish_reason(finish_reason: Optional[str], has_text: bool) -> ErrorClass:
    """Map a finish reason (+ whether any text came back) to a class."""
    reason = (finish_reason or "").upper()
    if reason in ("SAFETY", "BLOCKLIST", "PROHIBITED_CONTENT"):
        return ErrorClass.SAFETY
    if reason == "RECITATION":
        return ErrorClass.RECITATION
    if reason == "MAX_TOKENS":
        return ErrorClass.TRUNCATED
    if not has_text:
        return ErrorClass.EMPTY
    return ErrorClass.OK

def classify_exception(error: BaseException) -> ErrorClass:
    """Map an error that survived RateLimitedLLM's retries to a class."""
    code = status_code(error)
    if code == 429:
        return ErrorClass.RATE_LIMIT
    if code is not None and 500 <= code < 600:
        return ErrorClass.SERVER
    text = str(error).lower()  # fall back to string sniffing for anything odd
    if "timeout" in text or "timed out" in text:
        return ErrorClass.TIMEOUT
    return ErrorClass.SERVER

@dataclass
class RequestResult:
    request_id: str
    prompt_id: str
    error_class: str
    model_version: Optional[str]
    finish_reason: Optional[str]
    input_tokens: int
    output_tokens: int
    total_latency_s: float
    cost_usd: float
    start_unix: float
    end_unix: float

    def to_jsonl(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))
