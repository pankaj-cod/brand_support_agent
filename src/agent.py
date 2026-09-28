"""The AppleSupport agent: classify -> ground -> draft -> route.

Two LLM calls per case, deliberately separated so each stage can be measured on
its own. A single blended call is cheaper but makes it impossible to tell whether
a bad reply came from a bad intent or from bad grounding.

Stage 1  classify   intent + route + escalation reason, taxonomy in the prompt.
Stage 2  draft      reply grounded in retrieved real AppleSupport replies.

--no-retrieval runs stage 2 with the grounding block removed. That ablation is
how we show retrieval is doing work rather than decorating the prompt.
"""
import json, sys, argparse, concurrent.futures as cf
sys.path.insert(0, "src")
import taxonomy as T, llm, retrieval

CLASSIFY_SYS = """You triage incoming customer tweets for Apple Support.

Assign exactly one INTENT:
{intents}

Then decide ROUTE:
- "auto": a first-line public reply can help. Reversible, no personal data needed.
- "escalate": hand to a human agent.

Escalate only for one of these reasons:
{reasons}

Rules that matter:
- Escalate anything needing account, order, or identity data.
- Escalate anything where a wrong instruction destroys data or voids a repair.
- Escalate physical hazards (overheating, burns, swelling) immediately.
- Escalate abuse, legal threats, or a customer already failed by support.
- Venting with no answerable question is NOT an escalation; it is auto.
- Do not escalate merely because the problem is hard.

Return JSON only:
{{"intent": "...", "route": "auto"|"escalate", "escalate_reason": "..." or "",
  "confidence": 0.0-1.0, "rationale": "one short sentence"}}"""

DRAFT_SYS = """You draft public reply tweets for Apple Support.

House style, learned from how this brand actually replies:
- Under 280 characters. Warm, plain, never sarcastic back.
- Acknowledge the problem in one short clause, then give the next concrete step.
- Ask at most ONE diagnostic question if you genuinely need it.
- Never invent version numbers, prices, dates, or policies.
- Never promise a fix, a refund, a timeline, or compensation.
- If the route is "escalate", do NOT try to solve it. Acknowledge, say a
  specialist will take it, and ask them to move to DM. Do not ask for personal
  details in public.

Return JSON only: {"reply": "...", "grounded_in": [indices you used, may be empty]}"""

def classify(opener, model=None):
    msgs = [{"role": "system", "content": CLASSIFY_SYS.format(
                intents=T.taxonomy_prompt_block(), reasons=T.escalation_prompt_block())},
            {"role": "user", "content": f"Customer tweet:\n{opener}"}]
    try:
        d = llm.chat_json(msgs, model=model, temperature=0.0, max_tokens=600)
    except Exception as e:
        # Explicit failure. Scoring a fabricated label would be worse than a gap.
        return {"intent": None, "route": None, "escalate_reason": None,
                "confidence": 0.0, "rationale": "", "failed": f"classify: {e}"}
    intent = d.get("intent", "unclear")
    if intent not in T.INTENTS: intent = "unclear"
    route = d.get("route", "escalate")
    if route not in T.ROUTES: route = "escalate"
    reason = d.get("escalate_reason", "") or ""
    if route == "auto": reason = ""
    elif reason not in T.ESCALATE_REASONS: reason = "insufficient_info"
    return {"intent": intent, "route": route, "escalate_reason": reason,
            "confidence": float(d.get("confidence", 0.5) or 0.5),
            "rationale": d.get("rationale", "")}

def draft(opener, cls, hits, model=None, use_retrieval=True):
    blocks = ""
    if use_retrieval and hits:
        blocks = "\n\nHow Apple Support has answered similar tweets before:\n" + "\n".join(
            f"[{i}] customer: {h['opener'][:180]}\n    apple: {h['reply'][:240]}"
            for i, h in enumerate(hits))
    user = (f"Customer tweet:\n{opener}\n\n"
            f"Triage: intent={cls['intent']} route={cls['route']} "
            f"reason={cls['escalate_reason'] or 'n/a'}{blocks}")
    try:
        d = llm.chat_json([{"role": "system", "content": DRAFT_SYS},
                           {"role": "user", "content": user}],
                          model=model, temperature=0.0, max_tokens=700)
        reply = (d.get("reply") or "").strip()
    except Exception as e:
        return {"reply": "", "grounded_in": [], "failed": f"draft: {e}"}
    return {"reply": reply, "grounded_in": d.get("grounded_in", [])}

def run_case(opener, ix, model=None, use_retrieval=True, k=4):
    cls = classify(opener, model=model)
    if cls.get("failed"):
        return {**cls, "reply": "", "grounded_in": [], "retrieved": []}
    hits = ix.search(opener, k=k, min_substantive=0.8) if use_retrieval else []
    d = draft(opener, cls, hits, model=model, use_retrieval=use_retrieval)
    out = {**cls, "reply": d["reply"], "grounded_in": d.get("grounded_in", [])}
    if d.get("failed"): out["failed"] = d["failed"]
    return {**out,
            "retrieved": [{"opener": h["opener"], "reply": h["reply"],
                           "score": h["score"]} for h in hits]}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="golden/golden.jsonl")
    ap.add_argument("--out", default="data/processed/agent_preds.jsonl")
    ap.add_argument("--no-retrieval", action="store_true")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--tweet", default=None, help="run on one tweet and print the result")
    a = ap.parse_args()

    if a.tweet:
        r = run_case(a.tweet, retrieval.build(), model=a.model,
                     use_retrieval=not a.no_retrieval)
        print(json.dumps(r, ensure_ascii=False, indent=2))
        return

    import os
    rows = [json.loads(l) for l in open(a.input, encoding="utf-8")]
    if a.limit: rows = rows[:a.limit]
    ix = retrieval.build()

    # Resume: keep previously successful rows, redo only failures. A run that dies
    # at case 180 on a rate limit should not cost the previous 179.
    prev = {}
    if os.path.exists(a.out):
        for l in open(a.out, encoding="utf-8"):
            r = json.loads(l)
            if not r.get("failed") and r.get("intent"):
                prev[r["case_id"]] = r
        print("  resuming: %d already done" % len(prev), flush=True)

    todo = [r for r in rows if r["case_id"] not in prev]
    print("  to process: %d" % len(todo), flush=True)

    def work(r):
        return {"case_id": r["case_id"], "idx": r.get("idx"), "opener": r["opener"],
                **run_case(r["opener"], ix, model=a.model,
                           use_retrieval=not a.no_retrieval)}

    fresh = {}
    if todo:
        with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(work, r): r["case_id"] for r in todo}
            done = 0
            for f in cf.as_completed(futs):
                fresh[futs[f]] = f.result(); done += 1
                if done % 10 == 0:
                    nf = sum(1 for v in fresh.values() if v.get("failed"))
                    print("  %d/%d  failed=%d" % (done, len(todo), nf), flush=True)

    merged = {**prev, **fresh}
    out = [merged[r["case_id"]] for r in rows if r["case_id"] in merged]
    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + chr(10))
    failed = [r for r in out if r.get("failed")]
    print("wrote %s (%d/%d)  failed=%d  cache=%s"
          % (a.out, len(out), len(rows), len(failed), llm.cache_stats()))
    if failed:
        print("  !! rerun the same command to retry the %d failures" % len(failed))
        print("     example:", str(failed[0].get("failed"))[:160])

if __name__ == "__main__":
    main()
