"""Reconstruct AppleSupport conversation threads from the flat TWCS tweet table.

TWCS gives us a flat table of tweets with parent/child links. We rebuild threads,
then reduce each thread to a 'case': the customer's opening message plus the
brand's response chain. The case is the unit our agent acts on.
"""
import pandas as pd, numpy as np, json, re, html, os, argparse

RAW = "data/raw/twcs.csv"
OUT = "data/processed"
BRAND = "AppleSupport"

MENTION = re.compile(r"@\w+")
URL     = re.compile(r"https?://\S+")

def clean(t: str) -> str:
    """Normalise tweet text. We keep content words, drop handles and URLs, which
    carry no intent signal and leak thread ids into the model."""
    t = html.unescape(str(t))
    t = URL.sub(" <URL> ", t)
    t = MENTION.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def build(brand=BRAND, seed=13):
    df = pd.read_csv(RAW, usecols=["tweet_id","author_id","inbound","created_at","text",
                                   "response_tweet_id","in_response_to_tweet_id"])
    df["tweet_id"] = df.tweet_id.astype("int64")

    # Restrict to the neighbourhood of the brand: brand tweets + everything they
    # reply to + everything that replies to them.
    brand_tw = df[df.author_id == brand]
    parent_ids = set(brand_tw.in_response_to_tweet_id.dropna().astype("int64"))
    brand_ids  = set(brand_tw.tweet_id)
    child_ids  = set()
    for s in brand_tw.response_tweet_id.dropna():
        for p in str(s).split(","):
            p = p.strip()
            if p.isdigit(): child_ids.add(int(p))
    keep = brand_ids | parent_ids | child_ids
    sub = df[df.tweet_id.isin(keep)].copy()
    print(f"[prepare] brand-neighbourhood tweets: {len(sub)}")

    by_id = {int(r.tweet_id): r for r in sub.itertuples()}

    # Thread roots: customer tweets whose parent is not in our subset.
    roots = [i for i, r in by_id.items()
             if r.inbound and (pd.isna(r.in_response_to_tweet_id)
                               or int(r.in_response_to_tweet_id) not in by_id)]
    print(f"[prepare] thread roots: {len(roots)}")

    cases = []
    for rid in roots:
        turns, cur, seen = [], rid, set()
        while cur is not None and cur in by_id and cur not in seen:
            seen.add(cur)
            r = by_id[cur]
            turns.append({"tweet_id": int(r.tweet_id),
                          "author": "customer" if r.inbound else str(r.author_id),
                          "inbound": bool(r.inbound),
                          "created_at": str(r.created_at),
                          "text": clean(r.text)})
            nxt = None
            if pd.notna(r.response_tweet_id):
                for p in str(r.response_tweet_id).split(","):
                    p = p.strip()
                    if p.isdigit() and int(p) in by_id and int(p) not in seen:
                        nxt = int(p); break
            cur = nxt
        # A usable case needs a customer opener AND at least one brand reply.
        brand_turns = [t for t in turns if t["author"] == brand]
        if not brand_turns or not turns[0]["inbound"]:
            continue
        opener = turns[0]["text"]
        if len(opener.split()) < 3:      # junk / emoji-only openers
            continue
        cases.append({
            "case_id": f"{brand.lower()}-{turns[0]['tweet_id']}",
            "opener": opener,
            "created_at": turns[0]["created_at"],
            "n_turns": len(turns),
            "first_brand_reply": brand_turns[0]["text"],
            "all_brand_replies": [t["text"] for t in brand_turns],
            "turns": turns,
        })

    print(f"[prepare] usable cases: {len(cases)}")
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(cases))
    cases = [cases[i] for i in idx]

    with open(f"{OUT}/{brand.lower()}_cases.jsonl","w",encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False)+"\n")
    print(f"[prepare] wrote {OUT}/{brand.lower()}_cases.jsonl")

    lens = [c["n_turns"] for c in cases]
    print(f"[prepare] turns/case: median={np.median(lens):.0f} mean={np.mean(lens):.1f} max={max(lens)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--brand", default=BRAND)
    build(**vars(ap.parse_args()))
