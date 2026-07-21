# Local Setup

## Prerequisites

- **Python 3.13** (`python3.13 --version` to check; `brew install python@3.13` on macOS)
- **Google Cloud CLI** (`gcloud --version`; `brew install --cask google-cloud-sdk` on macOS)
- **Access to the Vertex AI project.** Your Google account must have `roles/aiplatform.user` on the project. There is no API key — auth is tied to your identity via Application Default Credentials.

## 1. Create the virtual environment (in the project root)

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Confirm you're in the venv on the right interpreter:

```bash
python --version   # Python 3.13.x
which python       # .../.venv/bin/python
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Authenticate to Google Cloud

Two separate credentials — you need both. The first authenticates the CLI;
the second writes Application Default Credentials, which is what the SDK
actually reads.

```bash
gcloud auth login
gcloud auth application-default login
```

## 4. Configure environment

Copy the committed template, then edit your copy:

```bash
cp .env.example .env
```

Open `.env` and set the three required values — everything else already has a
working default:

| Variable                | Set it to                                          |
| ----------------------- | -------------------------------------------------- |
| `GOOGLE_CLOUD_PROJECT`  | your Vertex AI project ID                          |
| `GOOGLE_CLOUD_LOCATION` | the region to call, e.g. `us-central1`             |
| `GEMINI_MODEL`          | the model ID, e.g. `gemini-2.5-flash`              |

`.env` is gitignored; `.env.example` is committed, so keep real values out of
the template. `llm/gemini.py` calls `load_dotenv()` at import, so no manual
`export` is needed — though a variable already exported in your shell takes
precedence over the same key in `.env`.

Optional overrides, with the defaults baked into `llm/gemini.py`:

| Variable                   | Default | Purpose                                                |
| -------------------------- | ------- | ------------------------------------------------------ |
| `GEMINI_THINKING_BUDGET`   | `0`     | `0` off, `-1` dynamic, `N` hard cap on thinking tokens |
| `GEMINI_MAX_OUTPUT_TOKENS` | `1000`  | Cap on answer tokens; must be positive                 |
| `GEMINI_MAX_CONNECTIONS`   | `100`   | httpx connection-pool size — the real client-side concurrency ceiling; raise for high-rate benchmarks |

`TOGETHER_API_KEY` and `TOGETHER_MODEL` are only needed if you use the Together
provider; they are commented out in the template.

## 5. Verify

Confirm ADC is live:

```bash
gcloud auth application-default print-access-token
```

Then a one-shot end-to-end check against Vertex:

```bash
python -c "from google import genai; \
print(genai.Client(vertexai=True, project='$GOOGLE_CLOUD_PROJECT', location='$GOOGLE_CLOUD_LOCATION')\
.models.generate_content(model='gemini-2.5-flash', contents='ping').text)"
```

A printed reply means auth, permissions, and the Vertex path all work.
