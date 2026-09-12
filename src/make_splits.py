"""Build corpus / dev / golden splits with no leakage between them.

Leakage matters here: the agent grounds its replies in historical AppleSupport
replies via retrieval. If a golden case sat in the retrieval corpus, the agent
could retrieve the very reply it is being scored against. So we partition the
cases FIRST, then sample the golden set only from the held-out side.

Golden sampling is stratified over embedding clusters rather than uniform. Uniform
sampling of this dataset yields a golden set that is ~35% "iOS 11 / the I-glitch"
complaints, because that is what October 2017 Twitter was shouting about. That
would let a majority-class model look good and would leave rare-but-expensive
intents (data_loss, hardware_repair) with too few examples to measure.
"""
import json, numpy as np, os, argparse
from sklearn.cluster import KMeans

CASES = "data/processed/applesupport_cases.jsonl"
OUT   = "data/processed"

def main(n_golden=220, n_corpus=30000, k=28, seed=13):
    rows = [json.loads(l) for l in open(CASES, encoding="utf-8")]
    rng = np.random.default_rng(seed)
    # rows were already shuffled deterministically in prepare_data.py
    n = len(rows)
    cut = int(n * 0.75)
    corpus_pool, eval_pool = rows[:cut], rows[cut:]
    print(f"[splits] corpus pool={len(corpus_pool)}  eval pool={len(eval_pool)}")

    corpus = corpus_pool[:n_corpus]

    # --- stratified golden sample over the EVAL pool ---
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    sub = eval_pool[:12000]
    X = m.encode([r["opener"] for r in sub], batch_size=256,
                 show_progress_bar=True, normalize_embeddings=True)
    lab = KMeans(n_clusters=k, random_state=seed, n_init=5).fit_predict(X)

    # Square-root allocation: damps the giant clusters, guarantees the small ones
    # a seat, without pretending the true distribution is uniform.
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
    print(f"[splits] golden sample: {len(picked)} across {k} strata "
          f"(alloc min={alloc.min()} max={alloc.max()})")

    golden_ids = {r["case_id"] for r in picked}
    dev = [r for r in eval_pool[12000:24000]]

    os.makedirs(OUT, exist_ok=True)
    def dump(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for r in data: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[splits] wrote {path} ({len(data)})")

    dump(f"{OUT}/corpus.jsonl", corpus)
    dump(f"{OUT}/dev.jsonl", dev)
    dump(f"{OUT}/golden_pool.jsonl", picked)
    assert not ({r["case_id"] for r in corpus} & golden_ids), "LEAK: golden case in corpus"
    print("[splits] leakage check passed")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_golden", type=int, default=220)
    ap.add_argument("--n_corpus", type=int, default=30000)
    main(**vars(ap.parse_args()))
