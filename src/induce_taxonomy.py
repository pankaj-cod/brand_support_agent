"""Induce a candidate intent taxonomy from the data (no LLM needed).

We embed a sample of customer openers, cluster them, and print exemplars +
distinctive terms per cluster. A human then consolidates these clusters into a
small taxonomy. Clusters are an INPUT to taxonomy design, not the taxonomy.
"""
import json, numpy as np, argparse
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

def main(n=12000, k=28, seed=13):
    rows = [json.loads(l) for l in open("data/processed/applesupport_cases.jsonl",encoding="utf-8")]
    rows = rows[:n]
    texts = [r["opener"] for r in rows]

    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("all-MiniLM-L6-v2")
    X = m.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    np.save("data/processed/opener_emb_sample.npy", X)

    km = KMeans(n_clusters=k, random_state=seed, n_init=5).fit(X)
    lab = km.labels_

    tf = TfidfVectorizer(max_features=20000, stop_words="english", ngram_range=(1,2))
    T = tf.fit_transform(texts); vocab = np.array(tf.get_feature_names_out())

    out = []
    for c in range(k):
        idx = np.where(lab==c)[0]
        centroid = np.asarray(T[idx].mean(axis=0)).ravel()
        top = vocab[np.argsort(-centroid)[:10]]
        # exemplars closest to cluster centre
        d = X[idx] @ km.cluster_centers_[c]
        ex = [texts[idx[i]] for i in np.argsort(-d)[:4]]
        out.append((c, len(idx), list(top), ex))

    for c, n_, top, ex in sorted(out, key=lambda r:-r[1]):
        print("="*100)
        print(f"CLUSTER {c}  n={n_}  ({100*n_/len(texts):.1f}%)")
        print("  terms:", ", ".join(top))
        for e in ex: print("   -", e[:170])

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=12000); ap.add_argument("--k",type=int,default=28)
    main(**vars(ap.parse_args()))
