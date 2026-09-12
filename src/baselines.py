"""Two baselines the agent must beat.

B0 TRIVIAL  - always predict the majority intent and the majority route, and send
              one fixed canned reply. This exists to expose how much of any
              headline accuracy is just class imbalance.
B1 SIMPLE   - TF-IDF + logistic regression for intent and route, and for the reply,
              the top-1 retrieved historical AppleSupport reply with no generation
              at all. Cheap, fast, no LLM.

Note an asymmetry that flatters B1: it is TRAINED on the golden labels (scored
out-of-fold via 5-fold CV) whereas the LLM agent never sees a single labelled
example. The comparison is therefore conservative against the agent.
"""
import json, sys, numpy as np, collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
sys.path.insert(0, "src")
import taxonomy as T, metrics as M

GOLDEN = "golden/golden.jsonl"
CANNED = ("We're happy to help! Send us a DM with more details and we'll take a "
          "look. <URL>")

def load():
    return [json.loads(l) for l in open(GOLDEN, encoding="utf-8")]

def trivial(rows):
    maj_i = collections.Counter(r["gold_intent"] for r in rows).most_common(1)[0][0]
    maj_r = collections.Counter(r["gold_route"] for r in rows).most_common(1)[0][0]
    return ([maj_i]*len(rows), [maj_r]*len(rows), [CANNED]*len(rows))

def simple(rows, seed=13):
    X = [r["opener"] for r in rows]
    yi = [r["gold_intent"] for r in rows]
    yr = [r["gold_route"] for r in rows]

    def cv(y):
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True,
                            strip_accents="unicode"),
            LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced"))
        cnt = collections.Counter(y); k = min(5, min(cnt.values()))
        skf = StratifiedKFold(n_splits=max(k,2), shuffle=True, random_state=seed)
        return list(cross_val_predict(pipe, X, y, cv=skf))

    # Reply baseline: nearest historical reply, no generation.
    import retrieval
    ix = retrieval.build()
    replies = [ix.search(x, k=1, min_substantive=0.8)[0]["reply"] for x in X]
    return cv(yi), cv(yr), replies

def main():
    rows = load()
    gi = [r["gold_intent"] for r in rows]; gr = [r["gold_route"] for r in rows]
    results = {}
    for name, fn in [("B0 trivial (majority class + canned reply)", trivial),
                     ("B1 simple (TF-IDF+LogReg + retrieved reply)", simple)]:
        pi, pr, rep = fn(rows)
        im, rm = M.intent_metrics(gi, pi), M.route_metrics(gr, pr)
        M.print_report(name, im, rm)
        results[name] = {"intent": pi, "route": pr, "reply": rep}
    with open("data/processed/baseline_preds.json", "w", encoding="utf-8") as f:
        json.dump(results, f)
    print("\nwrote data/processed/baseline_preds.json")

if __name__ == "__main__":
    main()
