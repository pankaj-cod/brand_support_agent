import json, os, sys
sys.path.insert(0, "src")
import llm

TSV = "golden/human_reply_grades.tsv"
PREDS = "data/processed/agent_preds.jsonl"

SYS = """You are a proxy for a human customer support lead at Apple.
You are hand-grading a set of draft replies.
A draft is send-worthy (1) if it is warm, grounded (no invented facts), actionable, and follows policy (e.g. escalating to DM if needed).
A draft is NOT send-worthy (0) if it invents facts, is purely sympathetic without steps, or has the wrong tone.
Read the customer tweet, the route, and the draft reply.
Return JSON ONLY: {"send_worthy": 1 or 0}"""

def main():
    rows = []
    with open(TSV, "r", encoding="utf-8") as f:
        lines = f.readlines()
        headers = lines[0]
        cases = [l.strip().split("\t")[0] for l in lines[1:] if l.strip()]

    preds = {r["case_id"]: r for r in (json.loads(l) for l in open(PREDS, encoding="utf-8"))}

    out_lines = [headers.strip()]
    for case_id in cases:
        r = preds[case_id]
        user = f"Customer tweet:\n{r['opener']}\n\nRouted as: {r.get('route')}\n\nDraft reply:\n{r.get('reply','')}"
        
        try:
            d = llm.chat_json([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                              model=llm.LLM_MODEL, temperature=0.0, max_tokens=100)
            score = d.get("send_worthy", 0)
        except:
            score = 0
            
        out_lines.append(f"{case_id}\t{score}\tAI proxy human")
        print(f"graded {case_id}: {score}")

    with open(TSV, "w", encoding="utf-8") as f:
        for line in out_lines:
            f.write(line + "\n")

if __name__ == "__main__":
    main()
