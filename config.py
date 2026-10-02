import os
from dotenv import load_dotenv

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

AGENT_MAX_TOKENS = os.getenv("AGENT_MAX_TOKENS", "0")

AGENT_REASONING_EFFORT = os.getenv("AGENT_REASONING_EFFORT")