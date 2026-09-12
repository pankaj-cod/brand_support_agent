"""One LLM entry point for the whole project, with a disk cache.

Provider-agnostic: everything speaks the OpenAI chat-completions wire format, so
this works against OpenAI, OpenRouter, Groq, Together, DeepInfra, or Anthropic's
OpenAI-compatible endpoint by changing two env vars.

Every call is cached on a hash of (model, messages, temperature, json_mode). This
matters for an eval harness: a rerun of the full evaluation must be free and must
return byte-identical results, otherwise "did my change help?" is unanswerable.

Two provider quirks cost real debugging time and are handled here:

* REASONING MODELS. openai/gpt-oss-* spend the token budget on hidden reasoning
  before writing anything. With max_tokens=150 the visible content came back
  EMPTY on 5 of 6 calls. We pass reasoning_effort="low" to these models and keep
  a generous max_tokens.
* STRICT JSON MODE. Groq validates response_format={"type":"json_object"} server
  side and returns 400 when the model's output does not parse -- which, for a
  reasoning model, was about half the time. We therefore ask for JSON in the
  prompt and parse tolerantly instead of relying on the provider's enforcement.
"""
import os, json, hashlib, time, pathlib, threading, collections, re

CACHE_DIR = pathlib.Path("data/llm_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
_lock = threading.Lock()

def _load_dotenv(path=".env"):
    if not os.path.exists(path): return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

_load_dotenv()

MODEL       = os.environ.get("LLM_MODEL", "gpt-4o-mini")
BASE_URL    = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
API_KEY     = os.environ.get("LLM_API_KEY", "")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", MODEL)
TPM         = int(os.environ.get("LLM_TPM", "8000"))
# Offline mode: serve only what is already cached and raise on a miss. Lets us
# work on partial results while a provider's daily quota is exhausted, without
# burning the remaining budget or accidentally mixing models mid-run.
CACHE_ONLY  = os.environ.get("LLM_CACHE_ONLY", "0") == "1"

class CacheMiss(RuntimeError):
    pass

# ---------------------------------------------------------------------------
# Rate limiting. Groq's free tier caps TOKENS per minute, and its bucket refills
# continuously (observed: ~700 tokens returned every 3s) rather than in discrete
# 60s windows. An earlier version reserved a pessimistic worst case against a
# rolling 60s ledger and throttled throughput roughly 10x below what the account
# actually allowed. We now reserve a realistic estimate, reconcile it against the
# usage the API reports, and otherwise let the provider's own 429 + retry-after
# do the work.
# ---------------------------------------------------------------------------
_ledger = collections.defaultdict(collections.deque)   # model -> [(ts, tokens)]
_rl = threading.Lock()

def _est_tokens(messages, max_tokens):
    chars = sum(len(m.get("content") or "") for m in messages)
    return int(chars / 4) + 300            # 300 covers reasoning + answer

def _prune(dq, now):
    while dq and now - dq[0][0] > 60: dq.popleft()

def _reserve(model, want):
    while True:
        with _rl:
            now = time.time(); dq = _ledger[model]; _prune(dq, now)
            used = sum(t for _, t in dq)
            if used + want <= TPM * 0.9:
                dq.append([now, want]); return
            sleep = max(0.5, 60 - (now - dq[0][0]))
        time.sleep(min(sleep, 30))

def _reconcile(model, want, actual):
    """Swap the estimate for what the call really cost."""
    with _rl:
        dq = _ledger[model]
        for e in reversed(dq):
            if e[1] == want: e[1] = actual; break

_client = None
def client():
    global _client
    if _client is None:
        from openai import OpenAI
        if not API_KEY:
            raise RuntimeError("No LLM_API_KEY. Copy .env.example to .env and fill it in.")
        _client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120.0, max_retries=0)
    return _client

def _is_reasoning(model): return "gpt-oss" in (model or "")

def _retry_after(err):
    m = re.search(r"try again in ([0-9.]+)s", str(err), re.I)
    if m: return float(m.group(1)) + 0.5
    return None

def _key(model, messages, temperature, json_mode):
    blob = json.dumps([model, messages, temperature, json_mode], sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]

def chat(messages, model=None, temperature=0.0, json_mode=False, max_tokens=700,
         retries=6, use_cache=True):
    """Return the assistant message text. Cached on disk."""
    model = model or MODEL
    k = _key(model, messages, temperature, json_mode)
    fp = CACHE_DIR / f"{k}.json"
    if use_cache and fp.exists():
        return json.loads(fp.read_text(encoding="utf-8"))["text"]
    if CACHE_ONLY:
        raise CacheMiss("cache-only mode: no cached response for this call")

    kwargs = dict(model=model, messages=messages, temperature=temperature,
                  max_tokens=max_tokens)
    if _is_reasoning(model):
        kwargs["reasoning_effort"] = "low"
    # NOTE: json_mode deliberately does NOT set response_format -- see module docstring.

    want = _est_tokens(messages, max_tokens)
    last = None
    for a in range(retries):
        _reserve(model, want)
        try:
            r = client().chat.completions.create(**kwargs)
            try: _reconcile(model, want, r.usage.total_tokens)
            except Exception: pass
            text = (r.choices[0].message.content or "").strip()
            if not text:
                raise RuntimeError("empty completion (reasoning consumed the budget?)")
            with _lock:
                fp.write_text(json.dumps({"text": text, "model": model}), encoding="utf-8")
            return text
        except Exception as e:
            last = e
            wait = _retry_after(e)
            time.sleep(min(wait if wait is not None else min(2 ** a, 20), 65))
    raise RuntimeError(f"LLM call failed after {retries} tries: {last}")

def chat_json(messages, **kw):
    """Chat returning parsed JSON, tolerating fenced or prefixed output."""
    txt = chat(messages, json_mode=kw.pop("json_mode", True), **kw)
    return parse_json(txt)

def parse_json(txt):
    t = (txt or "").strip()
    if t.startswith("```"):
        t = t.split("```")[1]
        t = t[4:] if t.lower().startswith("json") else t
    t = t.strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        i, j = t.find("{"), t.rfind("}")
        if i >= 0 and j > i:
            return json.loads(t[i:j+1])
        raise

def cache_stats():
    files = list(CACHE_DIR.glob("*.json"))
    return {"cached_calls": len(files),
            "mb": round(sum(f.stat().st_size for f in files)/1e6, 2)}
