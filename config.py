import os
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

