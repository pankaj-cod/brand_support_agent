"""LLM-as-judge for reply quality, plus the human-agreement harness.

Design choices worth defending:

1. REFERENCE-FREE. The judge does not see Apple's actual reply. Apple's reply is
   a deflection ~half the time, so scoring similarity to it would reward
   deflection and punish a genuinely better answer.

2. The headline is a BINARY decision -- would a support lead send this unedited?
   Mean rubric scores drift upward and are hard to act on; "send-worthy" maps
   directly to whether the thing can be deployed.

3. Rubric dimensions are scored BEFORE the verdict, so the verdict is forced to
   follow stated evidence rather than a vibe.

4. The judge should be a DIFFERENT model from the drafter (JUDGE_MODEL in .env),
   otherwise it is grading its own prose style. We measure that bias in
   src/judge_agreement.py.
"""
import json, sys, argparse, concurrent.futures as cf
sys.path.insert(0, "src")
import llm

RUBRIC_SYS = """You are a senior customer-support quality lead at Apple. You are
reviewing draft reply tweets before they go out publicly.

Score each dimension on 1-3 (1 bad, 2 borderline, 3 good):

- grounded: Does it avoid inventing facts? Any made-up version number, price,
  date, policy, or promise of a fix/refund/timeline scores 1.
- actionable: Does it give a concrete next step, or ask the one question that is
  genuinely needed? Pure sympathy with no step scores 1.
- tone: Warm, plain, non-defensive, not sarcastic, not corporate word-salad.
- policy: Under 280 chars; asks for NO personal data in public; if the case was
  escalated it hands off rather than attempting a fix.

Then decide:
- send_worthy: true only if a support lead would send this UNEDITED right now.
  If you would change a single word for correctness or safety, it is false.
- failure_mode: if not send_worthy, one of: "invented_fact", "no_next_step",
  "wrong_tone", "unsafe_advice", "asks_pii_publicly", "too_long",
  "ignores_escalation", "generic_deflection", "misread_intent". Else "".

Return JSON only:
{"grounded":1-3,"actionable":1-3,"tone":1-3,"policy":1-3,
 "send_worthy":true|false,"failure_mode":"...","why":"one short sentence"}"""

def judge_one(opener, reply, route, model=None):
    if not reply:
        return {"grounded":1,"actionable":1,"tone":1,"policy":1,"send_worthy":False,
                "failure_mode":"no_next_step","why":"empty reply"}
    user = (f"Customer tweet:\n{opener}\n\n"
            f"This case was routed as: {route}\n\n"
            f"Draft reply:\n{reply}")
    try:
        d = llm.chat_json([{"role":"system","content":RUBRIC_SYS},
                           {"role":"user","content":user}],
                          model=model or llm.JUDGE_MODEL, temperature=0.0, max_tokens=600)
    except Exception as e:
        return {"grounded":0,"actionable":0,"tone":0,"policy":0,"send_worthy":False,
                "failure_mode":"judge_error","why":str(e)}
    for k in ("grounded","actionable","tone","policy"):
        try: d[k] = int(d.get(k, 1))
        except Exception: d[k] = 1
    d["send_worthy"] = bool(d.get("send_worthy", False))
    if d["send_worthy"]: d["failure_mode"] = ""
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default="data/processed/agent_preds.jsonl")
    ap.add_argument("--out", default="data/processed/judged.jsonl")
    ap.add_argument("--model", default=None)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.preds, encoding="utf-8")]
    out = [None]*len(rows)
    def work(r): return {**r, "judge": judge_one(r["opener"], r.get("reply",""),
                                                 r.get("route","auto"), model=a.model)}
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, r): i for i, r in enumerate(rows)}
        done = 0
        for f in cf.as_completed(futs):
            out[futs[f]] = f.result(); done += 1
            if done % 20 == 0: print(f"  {done}/{len(rows)}", flush=True)

    with open(a.out, "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    sw = sum(1 for r in out if r["judge"]["send_worthy"])
    print(f"wrote {a.out}  send-worthy {sw}/{len(out)} = {100*sw/len(out):.1f}%")

if __name__ == "__main__":
    main()
