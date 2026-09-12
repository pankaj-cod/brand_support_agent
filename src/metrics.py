"""Metrics shared by every system under test."""
import numpy as np, collections, sys
from sklearn.metrics import (accuracy_score, f1_score, precision_recall_fscore_support,
                             confusion_matrix, cohen_kappa_score)
sys.path.insert(0, "src")
import taxonomy as T

def bootstrap_ci(y_true, y_pred, fn, n=2000, seed=13):
    """Percentile bootstrap CI. With n=220 the point estimate alone is not
    informative -- a 5-point difference between systems is often noise."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    stats = []
    for _ in range(n):
        i = rng.integers(0, len(y_true), len(y_true))
        try: stats.append(fn(y_true[i], y_pred[i]))
        except Exception: pass
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))

def intent_metrics(gold, pred):
    acc = accuracy_score(gold, pred)
    mf1 = f1_score(gold, pred, average="macro", labels=T.INTENT_NAMES, zero_division=0)
    lo, hi = bootstrap_ci(gold, pred, accuracy_score)
    mlo, mhi = bootstrap_ci(gold, pred,
        lambda a, b: f1_score(a, b, average="macro", labels=T.INTENT_NAMES, zero_division=0))
    p, r, f, s = precision_recall_fscore_support(
        gold, pred, labels=T.INTENT_NAMES, zero_division=0)
    return {"accuracy": acc, "accuracy_ci": (lo, hi),
            "macro_f1": mf1, "macro_f1_ci": (mlo, mhi),
            "per_class": {n: {"p": p[i], "r": r[i], "f1": f[i], "support": int(s[i])}
                          for i, n in enumerate(T.INTENT_NAMES)}}

def route_metrics(gold, pred):
    """Escalation is asymmetric: a missed escalation (routed a risky case to the
    bot) is far more costly than an unnecessary escalation. So we report recall
    on the ESCALATE class as the headline, not accuracy."""
    g = [1 if x == "escalate" else 0 for x in gold]
    p_ = [1 if x == "escalate" else 0 for x in pred]
    p, r, f, _ = precision_recall_fscore_support(g, p_, average="binary", zero_division=0)
    acc = accuracy_score(g, p_)
    rlo, rhi = bootstrap_ci(g, p_, lambda a, b: precision_recall_fscore_support(
        a, b, average="binary", zero_division=0)[1])
    tn, fp, fn_, tp = confusion_matrix(g, p_, labels=[0, 1]).ravel()
    return {"accuracy": acc, "escalate_precision": p, "escalate_recall": r,
            "escalate_recall_ci": (rlo, rhi), "escalate_f1": f,
            "missed_escalations": int(fn_), "over_escalations": int(fp),
            "tn": int(tn), "tp": int(tp)}

def print_report(name, im, rm):
    print(f"\n{'='*74}\n{name}\n{'='*74}")
    print(f"  intent accuracy  {im['accuracy']*100:5.1f}%  "
          f"[{im['accuracy_ci'][0]*100:.1f}, {im['accuracy_ci'][1]*100:.1f}]")
    print(f"  intent macro-F1  {im['macro_f1']*100:5.1f}%  "
          f"[{im['macro_f1_ci'][0]*100:.1f}, {im['macro_f1_ci'][1]*100:.1f}]")
    print(f"  route accuracy   {rm['accuracy']*100:5.1f}%")
    print(f"  ESCALATE recall  {rm['escalate_recall']*100:5.1f}%  "
          f"[{rm['escalate_recall_ci'][0]*100:.1f}, {rm['escalate_recall_ci'][1]*100:.1f}]"
          f"   precision {rm['escalate_precision']*100:.1f}%")
    print(f"  missed escalations {rm['missed_escalations']}   "
          f"over-escalations {rm['over_escalations']}")
    print(f"  {'per-intent':22s} {'P':>6} {'R':>6} {'F1':>6} {'n':>5}")
    for n, d in im["per_class"].items():
        print(f"  {n:22s} {d['p']*100:6.1f} {d['r']*100:6.1f} {d['f1']*100:6.1f} {d['support']:5d}")
