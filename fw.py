"""Thin Fireworks client: OpenAI-compatible chat with timing + token accounting."""
import os, time, base64, json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).with_name(".env"))
API_KEY = os.environ["FIREWORKS_API_KEY"]
ACCOUNT = os.environ.get("FIREWORKS_ACCOUNT", "accounts/sissiwang-xlorceau0c")
client = OpenAI(api_key=API_KEY, base_url="https://api.fireworks.ai/inference/v1")

def model_path(name: str) -> str:
    return name if name.startswith("accounts/") else f"accounts/fireworks/models/{name}"

def img_part(path: str) -> dict:
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}

def chat(model: str, messages: list, max_tokens: int = 512, temperature: float = 0.0, **kw) -> dict:
    """Returns {text, latency_s, prompt_tokens, completion_tokens, error}."""
    t0 = time.perf_counter()
    try:
        r = client.chat.completions.create(model=model_path(model), messages=messages,
                                           max_tokens=max_tokens, temperature=temperature, **kw)
        dt = time.perf_counter() - t0
        u = r.usage
        msg = r.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None) or ""
        return {"text": msg.content or "", "reasoning_chars": len(reasoning), "finish": r.choices[0].finish_reason, "latency_s": dt,
                "prompt_tokens": u.prompt_tokens if u else None,
                "completion_tokens": u.completion_tokens if u else None, "error": None}
    except Exception as e:
        return {"text": "", "latency_s": time.perf_counter() - t0, "prompt_tokens": None,
                "completion_tokens": None, "error": f"{type(e).__name__}: {str(e)[:200]}"}

def extract_json(text: str):
    """Best-effort: pull the first {...} or [...] block out of a reply."""
    import re
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|```$", "", text, flags=re.M).strip()
    for opener, closer in (("{", "}"), ("[", "]")):
        s = text.find(opener); e = text.rfind(closer)
        if s != -1 and e > s:
            try: return json.loads(text[s:e+1])
            except Exception: pass
    return None
