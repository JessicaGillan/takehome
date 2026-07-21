import os

from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions

load_dotenv()  # reads .env into os.environ; existing env vars win by default

client = genai.Client(http_options=HttpOptions(api_version="v1"))
response = client.models.generate_content(
    model=os.getenv("GEMINI_MODEL"),
    contents="How does AI work?",
)
print(response.text)