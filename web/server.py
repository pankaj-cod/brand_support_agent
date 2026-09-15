"""Local web server for the AppleSupport AI agent.

Run from the project root:
    python3 web/server.py

Then open:
    http://localhost:5001
"""
import sys, os, json, time, threading

# ── Resolve the project root so every relative path in src/ works correctly ──
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_ROOT)                       # data/, golden/, .env all live here
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from flask import Flask, request, jsonify, send_from_directory
import taxonomy as T, retrieval, agent as ag, llm

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app = Flask(__name__, static_folder=STATIC_DIR)

# ── Build retrieval index once at startup ────────────────────────────────────
_ix = None
_ix_lock = threading.Lock()


def get_index():
    global _ix
    if _ix is None:
        with _ix_lock:
            if _ix is None:
                _ix = retrieval.build()
                print(f"✅ Retrieval index ready — {len(_ix.cases)} corpus cases loaded",
                      flush=True)
    return _ix


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/classify", methods=["POST"])
def api_classify():
    """Stage 1 only: classify a tweet."""
    data = request.get_json(force=True)
    opener = (data.get("tweet") or "").strip()
    if not opener:
        return jsonify({"error": "No tweet provided"}), 400

    t0 = time.time()
    result = ag.classify(opener)
    result["latency_ms"] = int((time.time() - t0) * 1000)
    return jsonify(result)


@app.route("/api/retrieve", methods=["POST"])
def api_retrieve():
    """Retrieve similar historical cases."""
    data = request.get_json(force=True)
    opener = (data.get("tweet") or "").strip()
    k = data.get("k", 4)
    if not opener:
        return jsonify({"error": "No tweet provided"}), 400

    ix = get_index()
    t0 = time.time()
    hits = ix.search(opener, k=k, min_substantive=0.8)
    latency = int((time.time() - t0) * 1000)
    return jsonify({"hits": hits, "latency_ms": latency})


@app.route("/api/run", methods=["POST"])
def api_run():
    """Full pipeline: classify → retrieve → draft."""
    data = request.get_json(force=True)
    opener = (data.get("tweet") or "").strip()
    if not opener:
        return jsonify({"error": "No tweet provided"}), 400

    ix = get_index()
    t0 = time.time()
    result = ag.run_case(opener, ix, use_retrieval=True, k=4)
    result["latency_ms"] = int((time.time() - t0) * 1000)

    # Enrich with taxonomy info for the UI
    result["intent_description"] = T.INTENTS.get(result.get("intent"), "")
    result["default_route"] = T.INTENT_DEFAULT_ROUTE.get(result.get("intent"), "")
    if result.get("escalate_reason"):
        result["reason_description"] = T.ESCALATE_REASONS.get(
            result["escalate_reason"], "")
    return jsonify(result)


@app.route("/api/taxonomy", methods=["GET"])
def api_taxonomy():
    """Return the full taxonomy for display."""
    return jsonify({
        "intents": {
            k: {"description": v, "default_route": T.INTENT_DEFAULT_ROUTE[k]}
            for k, v in T.INTENTS.items()
        },
        "escalation_reasons": T.ESCALATE_REASONS,
        "model": llm.MODEL,
        "judge_model": llm.JUDGE_MODEL,
        "base_url": llm.BASE_URL,
        "cache_stats": llm.cache_stats(),
    })


@app.route("/api/examples", methods=["GET"])
def api_examples():
    """Return sample tweets from the golden set (one per intent)."""
    golden_path = os.path.join(PROJECT_ROOT, "golden", "golden.jsonl")
    examples = []
    if os.path.exists(golden_path):
        rows = [json.loads(l) for l in open(golden_path, encoding="utf-8")]
        seen_intents = set()
        for r in rows:
            if r["gold_intent"] not in seen_intents:
                seen_intents.add(r["gold_intent"])
                examples.append({
                    "tweet": r["opener"],
                    "gold_intent": r["gold_intent"],
                    "gold_route": r["gold_route"],
                })
            if len(examples) >= 10:
                break
    return jsonify(examples)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    threading.Thread(target=get_index, daemon=True).start()
    print("\n🚀 AppleSupport Agent running at http://localhost:5001\n", flush=True)
    app.run(host="0.0.0.0", port=5001, debug=False)
