import os
from getpass import getpass
from pathlib import Path
import requests


def _load_env():
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if env_path.exists():
        with env_path.open() as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


def _get_api_key():
    _load_env()
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        try:
            key = getpass("Enter your OpenRouter API key: ")
        except Exception:
            key = None
        if key:
            os.environ["OPENROUTER_API_KEY"] = key
    return key


def get_llm_response(question: str, *, debug: bool = False) -> str:
    """Call OpenRouter / DeepSeek and return the assistant text.

    The API key is read from `OPENROUTER_API_KEY` or prompted once via stdin.
    """
    key = _get_api_key()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek/deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a factual assistant. Give short, accurate answers."},
            {"role": "user", "content": question},
        ],
        "temperature": 0.2,
    }

    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    if not resp.ok:
        if debug:
            print("LLM request failed:", resp.status_code, resp.text)
        resp.raise_for_status()

    data = resp.json()
    if debug:
        print("LLM full response:", data)

    # defensive access into chat response
    choice = None
    if isinstance(data, dict):
        choices = data.get("choices") or data.get("outputs")
        if choices and isinstance(choices, list) and len(choices) > 0:
            choice = choices[0]

    if not choice:
        raise RuntimeError("Unexpected LLM response format")

    # support both `{message: {content: ...}}` and `{text: ...}` shapes
    if isinstance(choice, dict):
        msg = choice.get("message") or choice
        if isinstance(msg, dict) and "content" in msg:
            return msg["content"]
        if "text" in choice:
            return choice["text"]

    raise RuntimeError("Could not extract assistant text from LLM response")
