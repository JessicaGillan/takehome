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

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Authenticate to Google Cloud

Two separate credentials — you need both. The first authenticates the CLI;
the second writes Application Default Credentials, which is what the SDK
actually reads.

```bash
gcloud auth login
gcloud auth application-default login
```

## 5. Configure environment

The Gemini provider selects Vertex AI when `GOOGLE_CLOUD_PROJECT` is set, and
defaults the region to `us-central1`.

```bash
export GOOGLE_CLOUD_PROJECT=PROJECT_ID
export GOOGLE_CLOUD_LOCATION=us-central1
```

Optional overrides (sensible defaults are baked in):

| Variable                   | Default            | Purpose                                                |
| -------------------------- | ------------------ | ------------------------------------------------------ |
| `GEMINI_MODEL`             | `gemini-2.5-flash` | Model ID                                               |
| `GEMINI_THINKING_BUDGET`   | `0`                | `0` off, `-1` dynamic, `N` hard cap on thinking tokens |
| `GEMINI_MAX_OUTPUT_TOKENS` | unset              | Cap on answer tokens                                   |
| `GEMINI_PARALLELISM`       | `100`              | Client-side concurrency cap                            |
| `GEMINI_MAX_RETRIES`       | `5`                | Retry attempts on 429/5xx (Dynamic Shared Quota)       |

## 6. Verify

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
