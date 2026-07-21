from pathlib import Path

from dotenv import load_dotenv

# Test runs read the local .env for convenience; production injects real
# environment variables and never touches this path. Existing env vars win.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")
