"""Merge the hand-labelled batches into the golden evaluation set."""
import json, csv, collections, sys

POOL = "data/processed/golden_pool.jsonl"
OUT  = "golden/golden.jsonl"

def main():
    rows = [json.loads(l) for l in open(POOL, encoding="utf-8")]
    labels = {}
    for b in range(1, 5):
        for line in open(f"golden/labels_batch{b}.tsv", encoding="utf-8"):
            line = line.rstrip("\n")
            if not line or line.startswith("idx"): continue
            parts = line.split("\t")
            idx, intent, route = int(parts[0]), parts[1].strip(), parts[2].strip()
            reason = parts[3].strip() if len(parts) > 3 else ""
            labels[idx] = (intent, route, reason)

    sys.path.insert(0, "src"); import taxonomy as T
    missing = [i for i in range(len(rows)) if i not in labels]
    assert not missing, f"unlabelled indices: {missing}"

    out = []
    for i, r in enumerate(rows):
        intent, route, reason = labels[i]
        assert intent in T.INTENTS, f"bad intent {intent} at {i}"
        assert route in T.ROUTES, f"bad route {route} at {i}"
        if route == "escalate":
            assert reason in T.ESCALATE_REASONS, f"bad reason {reason} at {i}"
        else:
            assert reason == "", f"auto rows must have no reason (idx {i})"
        out.append({
            "case_id": r["case_id"], "idx": i, "opener": r["opener"],
            "stratum": r["stratum"],
            "gold_intent": intent, "gold_route": route, "gold_escalate_reason": reason,
            # Apple's own reply is kept as a REFERENCE for reply grading, never as a label.
            "reference_reply": r["first_brand_reply"],
            "n_turns": r["n_turns"],
        })

    with open(OUT, "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {OUT} ({len(out)})")

    ci = collections.Counter(r["gold_intent"] for r in out)
    cr = collections.Counter(r["gold_route"] for r in out)
    ce = collections.Counter(r["gold_escalate_reason"] for r in out if r["gold_route"]=="escalate")
    print("\n=== intent distribution ===")
    for k, v in ci.most_common(): print(f"  {k:22s} {v:4d}  {100*v/len(out):5.1f}%")
    print("\n=== route distribution ===")
    for k, v in cr.most_common(): print(f"  {k:22s} {v:4d}  {100*v/len(out):5.1f}%")
    print("\n=== escalation reasons ===")
    for k, v in ce.most_common(): print(f"  {k:22s} {v:4d}")

if __name__ == "__main__":
    main()
