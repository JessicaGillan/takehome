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

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Authenticate to Google Cloud

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

| Variable               | Set it to                 |
| ---------------------- | ------------------------- |
| `GOOGLE_CLOUD_PROJECT` | your Vertex AI project ID |

## 5. Verify

Confirm ADC is live:

```bash
gcloud auth application-default print-access-token
```

Confirm the gemini integration works and run load tests:

```bash
.venv/bin/python -m pytest                   # offline suite first: free, no credentials
.venv/bin/python -m pytest -m integration    # 1 live request — proves auth + the Vertex path
.venv/bin/python -m pytest -m loadtest       # 3 load scenarios, up to $1.50
.venv/bin/python -m pytest -m benchmark      # 3 production-rate tiers, budgets sum $3.25
```

Each live marker bills real requests to `GOOGLE_CLOUD_PROJECT` and must be
named explicitly — no other command selects them. Load runs write one JSONL
row per request plus an `.analysis.json` summary to `tests/bench/results/`,
which is committed so results can be reviewed.

> **Reading the results:** see [VERIFICATION.md](VERIFICATION.md) — section 9
> explains the benchmark pass criteria, and section 10 analyzes the measured
> results (throughput ceiling, latency, error mix, and the limitations of the
> setup).
