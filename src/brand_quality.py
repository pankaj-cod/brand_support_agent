"""Pick a brand by RESOLUTION RICHNESS, not raw volume.

A brand is only useful for grounded reply-drafting if its historical outbound
tweets contain real answers. Many big brands just deflect everything to DM.
"""
import pandas as pd, numpy as np, re

RAW = "data/raw/twcs.csv"
CANDIDATES = ["AmazonHelp","AppleSupport","Uber_Support","SpotifyCares","Delta",
              "AmericanAir","TMobileHelp","XboxSupport","hulu_support","AskPlayStation",
              "comcastcares","British_Airways","SouthwestAir","Ask_Spectrum","UPSHelp"]

DEFLECT = re.compile(r"\b(dm|direct message|private message|pm us|send us a (dm|message)|"
                     r"click the link|follow.{0,10}and dm|call us|give us a call|"
                     r"reach out to us at|email us)\b", re.I)
LINKOUT = re.compile(r"https?://|\bt\.co\b", re.I)
STEPS   = re.compile(r"\b(try|tap|go to|settings|restart|reset|uninstall|reinstall|"
                     r"toggle|update|check|make sure|ensure|swipe|hold|press|sign out|log out)\b", re.I)

def main():
    df = pd.read_csv(RAW, usecols=["tweet_id","author_id","inbound","text"])
    out = df[(~df.inbound) & (df.author_id.isin(CANDIDATES))].copy()
    out["text"] = out["text"].fillna("")
    out["n_words"]  = out.text.str.split().str.len()
    out["deflect"]  = out.text.str.contains(DEFLECT)
    out["linkout"]  = out.text.str.contains(LINKOUT)
    out["steps"]    = out.text.str.contains(STEPS)
    # "substantive" = gives actionable guidance and does NOT just punt to another channel
    out["substantive"] = out.steps & ~out.deflect

    g = out.groupby("author_id").agg(
        n=("text","size"),
        median_words=("n_words","median"),
        pct_deflect=("deflect","mean"),
        pct_linkout=("linkout","mean"),
        pct_steps=("steps","mean"),
        pct_substantive=("substantive","mean"),
    ).sort_values("pct_substantive", ascending=False)
    for c in ["pct_deflect","pct_linkout","pct_steps","pct_substantive"]:
        g[c] = (g[c]*100).round(1)
    print(g.to_string())

if __name__ == "__main__":
    main()
