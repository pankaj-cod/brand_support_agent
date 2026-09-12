"""Retrieve historically similar AppleSupport cases to ground a draft reply.

Grounding is the point: the agent should answer the way this brand actually
answered, not the way a generic assistant would. We embed the customer opener of
every corpus case and do cosine nearest-neighbour lookup.

We also score each historical reply for how SUBSTANTIVE it is. Roughly half of
AppleSupport's replies are "DM us" deflections, which are useless as grounding
for an auto-reply -- retrieving five of them teaches the agent to deflect.
"""
import json, re, os, numpy as np

CORPUS = "data/processed/corpus.jsonl"
EMB    = "data/processed/corpus_emb.npy"
MODEL  = "all-MiniLM-L6-v2"

DEFLECT = re.compile(r"\b(dm us|send us a dm|meet us in dm|continue in dm|in dm|"
                     r"direct message|private message|call us|contact us here|"
                     r"reach out to (us|our))\b", re.I)
STEPS   = re.compile(r"\b(try|tap|go to|settings|restart|reset|uninstall|reinstall|"
                     r"toggle|update|back up|check|make sure|ensure|swipe|hold|"
                     r"press|sign out|log out|force)\b", re.I)

def substantive_score(reply: str) -> float:
    """0..1. High = the reply actually told the customer what to do."""
    s = 0.0
    if STEPS.search(reply): s += 0.6
    if len(reply.split()) >= 15: s += 0.2
    if not DEFLECT.search(reply): s += 0.2
    return round(min(s, 1.0), 2)

class Index:
    def __init__(self, cases, emb, model):
        self.cases, self.emb, self.model = cases, emb, model

    def search(self, query, k=5, min_substantive=0.0):
        q = self.model.encode([query], normalize_embeddings=True)[0]
        sims = self.emb @ q
        order = np.argsort(-sims)
        out = []
        for i in order:
            c = self.cases[i]
            if c["substantive"] < min_substantive: continue
            out.append({"score": float(sims[i]), "opener": c["opener"],
                        "reply": c["reply"], "substantive": c["substantive"]})
            if len(out) >= k: break
        return out

def build(limit=None, rebuild=False):
    rows = [json.loads(l) for l in open(CORPUS, encoding="utf-8")]
    if limit: rows = rows[:limit]
    cases = [{"opener": r["opener"], "reply": r["first_brand_reply"],
              "substantive": substantive_score(r["first_brand_reply"])} for r in rows]

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL)
    if os.path.exists(EMB) and not rebuild:
        emb = np.load(EMB)
        if len(emb) == len(cases):
            return Index(cases, emb, model)
    emb = model.encode([c["opener"] for c in cases], batch_size=256,
                       show_progress_bar=True, normalize_embeddings=True)
    np.save(EMB, emb)
    return Index(cases, emb, model)

if __name__ == "__main__":
    ix = build()
    subs = np.array([c["substantive"] for c in ix.cases])
    print(f"corpus={len(ix.cases)}  mean substantive={subs.mean():.2f}  "
          f">=0.8: {(subs>=0.8).mean()*100:.1f}%")
    for q in ["my battery dies in 2 hours since the update",
              "I can't type the letter I",
              "my screen is cracked how much to fix"]:
        print("\n" + "="*80); print("QUERY:", q)
        for h in ix.search(q, k=3, min_substantive=0.8):
            print(f"  [{h['score']:.2f} sub={h['substantive']}] {h['opener'][:80]}")
            print(f"      -> {h['reply'][:150]}")
