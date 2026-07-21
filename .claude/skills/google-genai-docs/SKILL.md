---
name: google-genai-docs
description: Consult the official python-genai SDK reference at https://googleapis.github.io/python-genai/ before writing or changing Google Gemini / google-genai code. Use whenever the task involves the `google-genai` package, `from google import genai`, `google.genai.types`, `genai.Client`, `generate_content`, Gemini models, or files under `llm/` that talk to Gemini — including debugging SDK errors, choosing model IDs, or picking parameter/type names.
---

# Google GenAI SDK documentation

The `google-genai` Python SDK changes shape between releases: constructor
arguments, `types.*` names, and streaming/async method names have all moved.
Do not write `google-genai` code from memory.

## Rule

Before generating or editing any Gemini / `google-genai` code, fetch the
relevant page from <https://googleapis.github.io/python-genai/> with WebFetch
and match your code to what it documents.

Useful entry points:

- `https://googleapis.github.io/python-genai/` — overview, client setup,
  auth, quickstart examples for sync/async/streaming.
- `https://googleapis.github.io/python-genai/genai.html` — full API reference
  (`genai.Client`, `client.models`, `client.chats`, `client.files`, …).
- `https://googleapis.github.io/python-genai/genai.html#module-genai.types` —
  the `types` module: `HttpOptions`, `GenerateContentConfig`, `Part`,
  `Content`, `Tool`, `Schema`, and friends.

If the fetch fails, say so explicitly and flag that the code is
memory-based and unverified rather than presenting it as confirmed.

## Checks before finishing

- Every `google.genai.types` symbol used appears in the reference.
- Method names and their sync/async/streaming variants match the reference
  (e.g. `generate_content` vs `generate_content_stream` vs
  `aio.models.generate_content`).
- Config/keyword arguments are real parameters, not plausible-looking guesses.
- The installed version is what you documented against — check the pin in
  `requirements.txt` (currently `google-genai==2.12.1`) and note any
  mismatch with the docs, which track the latest release.

## Project conventions

- Client construction follows `llm/gemini.py`: `genai.Client(...)` with model
  IDs read from the environment (`GEMINI_MODEL`) rather than hardcoded.
  Production code never loads `.env` itself — `tests/conftest.py` calls
  `load_dotenv()` for test runs; deployments inject real env vars.
- Credentials and model names live in `.env` — never inline an API key.
