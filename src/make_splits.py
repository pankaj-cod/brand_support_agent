"""Build corpus / dev / golden splits with no leakage between them.

Leakage matters here: the agent grounds its replies in historical AppleSupport
replies via retrieval. If a golden case sat in the retrieval corpus, the agent
could retrieve the very reply it is being scored against. So we partition the
cases FIRST, then sample the golden set only from the held-out side.

Golden sampling is stratified over embedding clusters rather than uniform. Uniform
sampling of this dataset yields a golden set that is ~35% "iOS 11 / the I-glitch"
complaints, because that is what October 2017 Twitter was shouting about. That
would let a majority-class model look good and would leave rare-but-expensive
intents (data_loss, hardware_repair) with too few examples to measure. Once
golden/golden.jsonl exists, the pool is rebuilt from its case ids instead of
re-sampled, so the hand labels stay attached to the cases they were written for.

--intent_sampler switches to src/sample_golden.py, which samples to per-intent
quotas using keyword rules. Those rules also assign the labels, so its output is
not a hand-labelled set and is not what REPORT.md is scored on.
"""
import json, numpy as np, os, argparse, sys
from sklearn.cluster import KMeans

CASES = "data/processed/applesupport_cases.jsonl"
OUT   = "data/processed"
GOLDEN = "golden/golden.jsonl"

def main(n_golden=220, n_corpus=30000, k=28, seed=13, stratified=True):
    rows = [json.loads(l) for l in open(CASES, encoding="utf-8")]
    rng = np.random.default_rng(seed)
    n = len(rows)
    cut = int(n * 0.75)
    corpus_pool, eval_pool = rows[:cut], rows[cut:]
    print(f"[splits] corpus pool={len(corpus_pool)}  eval pool={len(eval_pool)}")

    corpus = corpus_pool[:n_corpus]
    dev = [r for r in eval_pool[12000:24000]]

    os.makedirs(OUT, exist_ok=True)
    def dump(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for r in data: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[splits] wrote {path} ({len(data)})")

    dump(f"{OUT}/corpus.jsonl", corpus)
    dump(f"{OUT}/dev.jsonl", dev)

    if stratified:
        print("[splits] Running intentional stratified sampling (10 intents + 15 edge cases, semantic diversity)...")
        sys.path.insert(0, "src")
        import sample_golden
        sample_golden.sample_golden_dataset(cases_path=CASES, corpus_path=f"{OUT}/corpus.jsonl", seed=seed)
    elif os.path.exists(GOLDEN):
        # The hand labels in golden/labels_batch*.tsv are keyed by position in this
        # pool. Embeddings and KMeans drift across library versions, so re-sampling
        # does not reproduce the same 220 cases; pin the pool to the committed set.
        by_id = {r["case_id"]: r for r in eval_pool}
        gold = [json.loads(l) for l in open(GOLDEN, encoding="utf-8")]
        missing = [g["case_id"] for g in gold if g["case_id"] not in by_id]
        assert not missing, f"{len(missing)} golden cases not in the eval pool, e.g. {missing[:3]}"
        dump(f"{OUT}/golden_pool.jsonl",
             [dict(by_id[g["case_id"]], stratum=g["stratum"]) for g in gold])
    else:
        # Legacy square-root allocation over raw clusters
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("all-MiniLM-L6-v2")
        sub = eval_pool[:12000]
        X = m.encode([r["opener"] for r in sub], batch_size=256,
                     show_progress_bar=True, normalize_embeddings=True)
        lab = KMeans(n_clusters=k, random_state=seed, n_init=5).fit_predict(X)

        sizes = np.array([(lab == c).sum() for c in range(k)], dtype=float)
        alloc = np.sqrt(sizes); alloc = alloc / alloc.sum() * n_golden
        alloc = np.maximum(np.round(alloc).astype(int), 3)
        while alloc.sum() > n_golden: alloc[np.argmax(alloc)] -= 1
        while alloc.sum() < n_golden: alloc[np.argmin(alloc)] += 1

        picked = []
        for c in range(k):
            idx = np.where(lab == c)[0]
            take = rng.choice(idx, size=min(alloc[c], len(idx)), replace=False)
            for i in take:
                r = dict(sub[i]); r["stratum"] = int(c); picked.append(r)
        rng.shuffle(picked)
        dump(f"{OUT}/golden_pool.jsonl", picked)

    golden_rows = [json.loads(l) for l in open(f"{OUT}/golden_pool.jsonl", encoding="utf-8")]
    golden_ids = {r["case_id"] for r in golden_rows}
    corpus_ids = {r["case_id"] for r in corpus}
    assert not (corpus_ids & golden_ids), f"LEAK: {len(corpus_ids & golden_ids)} golden cases found in corpus"
    print(f"[splits] strict zero-leakage check passed ({len(golden_ids)} golden cases, {len(corpus_ids)} corpus cases)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_golden", type=int, default=220)
    ap.add_argument("--n_corpus", type=int, default=30000)
    ap.add_argument("--intent_sampler", action="store_true",
                    help="Use the keyword-based intent sampler (src/sample_golden.py). Its labels "
                         "are rule-assigned, not hand-labelled, and it overwrites golden/golden.jsonl")
    args = ap.parse_args()
    main(n_golden=args.n_golden, n_corpus=args.n_corpus, stratified=args.intent_sampler)
