"""How much should we trust the LLM judge?

An LLM-judged 'send-worthy' number is worthless without knowing whether the judge
agrees with a person. We hand-grade a stratified subset of the SAME replies on the
SAME binary question, then report raw agreement, Cohen's kappa, and -- most
usefully -- where the judge is biased (does it wave through replies a human would
reject, or the reverse?).

Kappa, not raw agreement, is the number to read: if 80% of replies are send-worthy,
a judge that says "yes" every time scores 80% agreement and is useless.
"""
import json, sys, argparse, collections
from sklearn.metrics import cohen_kappa_score, confusion_matrix

def load_human(path):
    """TSV: case_id \t send_worthy(1/0) \t optional note"""
    h = {}
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line or line.startswith("case_id"): continue
        p = line.split("\t")
        h[p[0]] = int(p[1])
    return h

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judged", default="data/processed/judged.jsonl")
    ap.add_argument("--human", default="golden/human_reply_grades.tsv")
    a = ap.parse_args()

    judged = {r["case_id"]: r for r in
              (json.loads(l) for l in open(a.judged, encoding="utf-8"))}
    human = load_human(a.human)
    ids = [i for i in human if i in judged]
    hy = [human[i] for i in ids]
    jy = [1 if judged[i]["judge"]["send_worthy"] else 0 for i in ids]

    agree = sum(int(x==y) for x, y in zip(hy, jy))/len(ids)
    kappa = cohen_kappa_score(hy, jy)
    tn, fp, fn, tp = confusion_matrix(hy, jy, labels=[0,1]).ravel()
    print(f"n graded by hand      : {len(ids)}")
    print(f"human send-worthy rate: {sum(hy)/len(hy)*100:.1f}%")
    print(f"judge send-worthy rate: {sum(jy)/len(jy)*100:.1f}%")
    print(f"raw agreement         : {agree*100:.1f}%")
    print(f"Cohen's kappa         : {kappa:.3f}")
    print(f"\n  judge says SEND, human says NO (judge too lenient): {fp}")
    print(f"  judge says NO, human says SEND (judge too harsh)   : {fn}")
    print(f"  both SEND {tp}   both NO {tn}")
    verdict = ("substantial" if kappa >= .61 else "moderate" if kappa >= .41
               else "fair" if kappa >= .21 else "poor")
    print(f"\nAgreement is {verdict} (Landis & Koch bands).")
    json.dump({"n": len(ids), "raw_agreement": agree, "kappa": kappa,
               "judge_lenient_errors": int(fp), "judge_harsh_errors": int(fn),
               "human_rate": sum(hy)/len(hy), "judge_rate": sum(jy)/len(jy)},
              open("reports/judge_agreement.json","w",encoding="utf-8"), indent=2)
    print("wrote reports/judge_agreement.json")

if __name__ == "__main__":
    main()
