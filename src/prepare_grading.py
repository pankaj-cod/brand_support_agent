import json, random, os

PREDS_FILE = "data/processed/agent_preds.jsonl"
OUT_TSV = "golden/human_reply_grades.tsv"
OUT_MD = "golden/human_grading_worksheet.md"
NUM_SAMPLES = 60

def main():
    if not os.path.exists(PREDS_FILE):
        print(f"File {PREDS_FILE} not found. Please wait until it's generated.")
        return

    with open(PREDS_FILE, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    
    # Filter valid rows
    rows = [r for r in rows if not r.get("failed") and r.get("intent")]

    if len(rows) < NUM_SAMPLES:
        print(f"Warning: Only {len(rows)} valid rows found. Needed {NUM_SAMPLES}.")

    # Sample randomly with a fixed seed for reproducibility across runs if needed
    random.seed(42)
    sample = random.sample(rows, min(NUM_SAMPLES, len(rows)))

    # Write TSV
    with open(OUT_TSV, "w", encoding="utf-8") as f:
        f.write("case_id\tsend_worthy\tnote\n")
        for r in sample:
            f.write(f"{r['case_id']}\t\t\n")
    
    # Write Worksheet
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# Human Grading Worksheet\n\n")
        f.write("Review the cases below. For each case, decide if the reply is **send-worthy** (would you send it unedited right now?).\n")
        f.write(f"Record your 1 (yes) or 0 (no) in `{OUT_TSV}`.\n\n")
        
        for i, r in enumerate(sample):
            f.write(f"## Case {i+1}: `{r['case_id']}`\n\n")
            f.write(f"**Customer:** {r['opener']}\n\n")
            f.write(f"**Routed as:** {r.get('route', 'auto')}\n\n")
            f.write(f"**Reply:** {r.get('reply', '')}\n\n")
            f.write("---\n\n")

    print(f"Created {OUT_TSV} and {OUT_MD} with {len(sample)} samples.")

if __name__ == "__main__":
    main()
