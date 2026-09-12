"""Brand selection: which brand gives us enough resolvable, self-contained threads?"""
import pandas as pd, numpy as np, sys

RAW = "data/raw/twcs.csv"

def main():
    df = pd.read_csv(RAW, usecols=["tweet_id","author_id","inbound","created_at","text",
                                   "response_tweet_id","in_response_to_tweet_id"])
    print("rows:", len(df))
    brands = df[~df.inbound]["author_id"].value_counts()
    print("\n=== top 25 brands by outbound tweet volume ===")
    print(brands.head(25).to_string())

    # inbound tweets addressed to each brand (customer -> brand), via reply linkage
    outb = df[~df.inbound]
    inb = df[df.inbound]
    # map: inbound tweet -> brand that replied to it
    reply_map = outb[["author_id","in_response_to_tweet_id"]].dropna()
    reply_map["in_response_to_tweet_id"] = reply_map["in_response_to_tweet_id"].astype("int64")
    answered = reply_map.groupby("author_id").size().sort_values(ascending=False)
    print("\n=== top 25 brands by ANSWERED inbound customer tweets ===")
    print(answered.head(25).to_string())

if __name__ == "__main__":
    main()
