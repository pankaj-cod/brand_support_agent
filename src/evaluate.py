"""Score every system on the golden set and emit the results table."""
import json, sys, argparse, collections
sys.path.insert(0, "src")
import metrics as M, taxonomy as T

GOLDEN = "golden/golden.jsonl"

def load_golden(): return [json.loads(l) for l in open(GOLDEN, encoding="utf-8")]

def load_jsonl(p): return [json.loads(l) for l in open(p, encoding="utf-8")]

def eval_system(name, gold, pi, pr, judged=None):
    gi = [r["gold_intent"] for r in gold]; gr = [r["gold_route"] for r in gold]
    im, rm = M.intent_metrics(gi, pi), M.route_metrics(gr, pr)
    M.print_report(name, im, rm)
    row = {"system": name, "intent_acc": im["accuracy"], "intent_acc_ci": im["accuracy_ci"],
           "macro_f1": im["macro_f1"], "macro_f1_ci": im["macro_f1_ci"],
           "esc_recall": rm["escalate_recall"], "esc_recall_ci": rm["escalate_recall_ci"],
           "esc_precision": rm["escalate_precision"],
           "missed_escalations": rm["missed_escalations"],
           "over_escalations": rm["over_escalations"]}
    if judged:
        sw = [j["judge"]["send_worthy"] for j in judged]
        row["send_worthy"] = sum(sw)/len(sw)
        lo, hi = M.bootstrap_ci([1]*len(sw), sw, lambda a,b: sum(b)/len(b))
        row["send_worthy_ci"] = (lo, hi)
        print(f"  SEND-WORTHY      {row['send_worthy']*100:5.1f}%  [{lo*100:.1f}, {hi*100:.1f}]")
        dims = collections.defaultdict(list)
        for j in judged:
            for k in ("grounded","actionable","tone","policy"): dims[k].append(j["judge"][k])
        print("  rubric means:", {k: round(sum(v)/len(v),2) for k,v in dims.items()})
        fm = collections.Counter(j["judge"]["failure_mode"] for j in judged
                                 if not j["judge"]["send_worthy"])
        if fm: print("  failure modes:", dict(fm.most_common()))
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="data/processed/judged.jsonl")
    ap.add_argument("--ablation", default=None)
    ap.add_argument("--baselines", default="data/processed/baseline_preds.json")
    ap.add_argument("--out", default="reports/results.json")
    a = ap.parse_args()

    gold = load_golden()
    table = []

    import os
    # Fair comparison: if any LLM system covers only part of the golden set (a
    # provider quota can stop a run mid-way), every system is scored on the SAME
    # subset. Comparing a partial agent against full-set baselines would be a
    # silent apples-to-oranges bug.
    common = {g["case_id"] for g in gold}
    for pth in [a.agent, "data/processed/agent_preds.jsonl", a.ablation]:
        if pth and os.path.exists(pth):
            ids = {r["case_id"] for r in load_jsonl(pth)
                   if not r.get("failed") and r.get("intent")}
            if ids: common &= ids
    if len(common) < len(gold):
        print("[scope] all systems scored on the %d/%d cases every system covers"
              % (len(common), len(gold)))
    gold = [g for g in gold if g["case_id"] in common]
    if os.path.exists(a.baselines):
        b = json.load(open(a.baselines, encoding="utf-8"))
        full = load_golden()
        keep = [i for i, g in enumerate(full) if g["case_id"] in common]
        for name, d in b.items():
            table.append(eval_system(name, gold,
                                     [d["intent"][i] for i in keep],
                                     [d["route"][i] for i in keep]))

    def add(label, path, judged_ok=True):
        by = {r["case_id"]: r for r in load_jsonl(path)
               if not r.get("failed") and r.get("intent")}
        sub = [g for g in gold if g["case_id"] in by]          # align on case_id
        if len(sub) < len(gold):
            print("[warn] %s: PARTIAL -- scoring on %d/%d golden cases" % (label, len(sub), len(gold)))
        ordered = [by[g["case_id"]] for g in sub]
        has_judge = bool(ordered) and "judge" in ordered[0]
        table.append(eval_system(label, sub,
                                 [r["intent"] for r in ordered],
                                 [r["route"] for r in ordered],
                                 judged=ordered if (judged_ok and has_judge) else None))

    if os.path.exists(a.agent):
        add("AGENT (LLM + retrieval)", a.agent)
    elif os.path.exists("data/processed/agent_preds.jsonl"):
        add("AGENT (LLM + retrieval, UNJUDGED)", "data/processed/agent_preds.jsonl")
    if a.ablation and os.path.exists(a.ablation):
        add("ABLATION (LLM, no retrieval)", a.ablation)

    os.makedirs("reports", exist_ok=True)
    json.dump(table, open(a.out, "w", encoding="utf-8"), indent=2)
    print(f"\n{'='*74}\nHEADLINE TABLE\n{'='*74}")
    print(f"{'system':44s} {'int.acc':>8} {'macroF1':>8} {'escRec':>8} {'sendOK':>8}")
    for r in table:
        sw = f"{r.get('send_worthy',float('nan'))*100:7.1f}%" if "send_worthy" in r else "      --"
        print(f"{r['system'][:44]:44s} {r['intent_acc']*100:7.1f}% "
              f"{r['macro_f1']*100:7.1f}% {r['esc_recall']*100:7.1f}% {sw}")
    print(f"\nwrote {a.out}")

if __name__ == "__main__":
    main()
