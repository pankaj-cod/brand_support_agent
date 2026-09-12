"""Enumerate real failures for the report. No LLM needed."""
import json, sys, collections
sys.path.insert(0,"src")

gold = {r["case_id"]: r for r in
        (json.loads(l) for l in open("golden/golden.jsonl",encoding="utf-8"))}
pred = [r for r in (json.loads(l) for l in
        open("data/processed/agent_preds.jsonl",encoding="utf-8"))
        if not r.get("failed") and r.get("intent")]

print("scored on %d cases\n" % len(pred))

print("="*100); print("MISSED ESCALATIONS  (gold=escalate, agent said auto) -- the expensive error")
print("="*100)
missed = [r for r in pred if gold[r["case_id"]]["gold_route"]=="escalate" and r["route"]=="auto"]
for r in missed:
    g = gold[r["case_id"]]
    print("- [%s -> %s | reason %s]" % (g["gold_intent"], r["intent"], g["gold_escalate_reason"]))
    print("  CUST :", r["opener"][:155])
    print("  REPLY:", (r["reply"] or "")[:155])
print("\ntotal missed escalations:", len(missed))

print("\n"+"="*100); print("OVER-ESCALATIONS (gold=auto, agent escalated)")
print("="*100)
over = [r for r in pred if gold[r["case_id"]]["gold_route"]=="auto" and r["route"]=="escalate"]
for r in over:
    g = gold[r["case_id"]]
    print("- [%s -> %s | agent reason %s]" % (g["gold_intent"], r["intent"], r["escalate_reason"]))
    print("  CUST :", r["opener"][:155])
print("\ntotal over-escalations:", len(over))

print("\n"+"="*100); print("INTENT CONFUSIONS (most frequent first)")
print("="*100)
conf = collections.Counter((gold[r["case_id"]]["gold_intent"], r["intent"])
                           for r in pred if gold[r["case_id"]]["gold_intent"]!=r["intent"])
for (g_, p_), n in conf.most_common(12):
    print("  %-20s -> %-20s  x%d" % (g_, p_, n))

print("\n"+"="*100); print("EXAMPLES PER TOP CONFUSION")
print("="*100)
for (g_, p_), n in conf.most_common(4):
    print("\n### gold=%s  predicted=%s  (n=%d)" % (g_, p_, n))
    for r in pred:
        if gold[r["case_id"]]["gold_intent"]==g_ and r["intent"]==p_:
            print("   CUST:", r["opener"][:150])
            print("   WHY :", str(r.get("rationale",""))[:130])
