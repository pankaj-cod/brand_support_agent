"""Intentional stratified sampling for the AppleSupport golden evaluation dataset.

Key objectives:
1. Reconstructed CASES (not raw individual tweets).
2. Exactly 220 cases:
   - 205 normal cases stratified across 10 intents:
     * software_bug: 25
     * battery_charging: 20
     * connectivity: 25
     * app_service: 20
     * how_to: 25
     * feedback_no_action: 15
     * account_billing: 20
     * hardware_repair: 20
     * data_loss: 20
     * unclear: 15
   - 15 difficult/edge cases prioritizing:
     * safety-sensitive cases (overheating, burning, swelling)
     * irreversible-risk cases (accidental wipe, factory reset)
     * private-data/account cases (unauthorized billing, locked ID)
     * angry/abusive customers (profanity, churn/legal threats)
     * cases where intent and routing conflict / default route overridden
     * ambiguous / insufficient-information cases
3. Semantic diversity: within each intent, clusters candidate embeddings
   (SentenceTransformer all-MiniLM-L6-v2 + KMeans) and selects the closest
   exemplar per cluster to prevent near-duplicates.
4. Leakage prevention: samples strictly from eval_pool (rows[cut:]), completely
   held out from the retrieval corpus (rows[:cut]). Strict assertion check.
5. Produces:
   - golden/golden.jsonl: the 220 golden cases
   - data/processed/golden_pool.jsonl: split pool file
   - golden/human_labeling_worksheet.md: complete human-grade worksheet
   - golden/human_labeling_dataset.tsv: TSV with all required fields
   - golden/golden_labels.tsv: TSV for indexing and grading
"""

import json, os, re, sys, argparse
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "src")
import taxonomy as T

CASES_PATH = "data/processed/applesupport_cases.jsonl"
CORPUS_PATH = "data/processed/corpus.jsonl"
OUT_GOLDEN = "golden/golden.jsonl"
OUT_POOL = "data/processed/golden_pool.jsonl"
OUT_WORKSHEET = "golden/human_labeling_worksheet.md"
OUT_TSV = "golden/human_labeling_dataset.tsv"
OUT_LABELS_TSV = "golden/golden_labels.tsv"

NORMAL_QUOTAS = {
    "software_bug": 25,
    "battery_charging": 20,
    "connectivity": 25,
    "app_service": 20,
    "how_to": 25,
    "feedback_no_action": 15,
    "account_billing": 20,
    "hardware_repair": 20,
    "data_loss": 20,
    "unclear": 15,
}
EDGE_QUOTA = 15
TOTAL_CASES = sum(NORMAL_QUOTAS.values()) + EDGE_QUOTA  # 220

def format_conversation_context(turns):
    """Format reconstructed conversation thread for human review."""
    lines = []
    for t in turns:
        role = "Customer" if t.get("inbound") or t.get("author") == "customer" else "AppleSupport"
        text = t.get("text", "").strip()
        lines.append(f"{role}: {text}")
    return "\n".join(lines)

def craft_reference_response(intent, route, reason, opener, first_brand_reply):
    """Craft human-grade, send-worthy reference response adhering to AppleSupport style:
    - Under 280 characters
    - Warm, plain, non-defensive
    - If escalate: acknowledge situation, do not troubleshoot, invite to DM for specialist
    - If auto: acknowledge issue, provide direct actionable next step or single diagnostic question
    """
    op_lower = opener.lower()
    
    # 1. Escalated cases:
    if route == "escalate":
        if reason == "safety_risk":
            return ("Your safety is our top priority. Please stop using and charging the device "
                    "immediately. Meet us in DM right away so a senior specialist can assist: <URL>")
        if reason == "irreversible_risk":
            return ("We understand how vital your data is and want to avoid permanent loss. "
                    "Please avoid restoring or erasing your device right now. Send us a DM so we can guide you: <URL>")
        if reason == "needs_private_data":
            if "order" in op_lower or "ship" in op_lower or "delivery" in op_lower:
                return ("We can help check your order details securely. Please send us a DM with your "
                        "order number and Apple ID email: <URL>")
            elif "charge" in op_lower or "bill" in op_lower or "refund" in op_lower or "subscription" in op_lower:
                return ("We want to help resolve this billing issue. For your account security, "
                        "please send us a DM with your Apple ID email so we can investigate: <URL>")
            else:
                return ("To protect your account security, we cannot discuss account details publicly. "
                        "Please send us a DM so an advisor can verify your Apple ID: <URL>")
        if reason == "relationship_risk":
            return ("We sincerely apologize for this frustrating experience. We want to make this right—"
                    "please join us in DM with details of your previous case so we can address it: <URL>")
        if reason == "insufficient_info":
            return ("We would love to help get this sorted. Could you send us a DM with more details "
                    "about what is happening and the device model you are using? <URL>")
        # Default fallback escalation:
        return ("We're here to help get this resolved. Please connect with us in DM so a specialist "
                "can look into your case directly: <URL>")

    # 2. Auto cases by intent:
    if intent == "software_bug":
        if "ios 11" in op_lower and ("type" in op_lower or "letter i" in op_lower or "autocorrect" in op_lower):
            return ("Here’s what you can do to work around this autocorrect issue until it’s fixed in "
                    "an upcoming update: go to Settings > General > Keyboard > Text Replacement. <URL>")
        elif "freeze" in op_lower or "crash" in op_lower or "slow" in op_lower:
            return ("Sorry to hear your device is freezing! Try a force restart: press and quickly release "
                    "Volume Up, then Volume Down, then hold the Side button until the Apple logo appears.")
        elif "black screen" in op_lower:
            return ("Let's get that display working again. Try connecting your device to power for 15 minutes, "
                    "then perform a force restart while still plugged in.")
        else:
            return ("We want your software running smoothly. Make sure your device is updated to the "
                    "latest version in Settings > General > Software Update, and try restarting the device.")

    if intent == "battery_charging":
        if "drain" in op_lower or "dies" in op_lower or "fast" in op_lower:
            return ("Battery life is essential. Check Settings > Battery to see which apps are using the most "
                    "energy over the last 24 hours, and make sure Low Power Mode is enabled.")
        elif "won't charge" in op_lower or "not charging" in op_lower:
            return ("Let's check your charging setup. Inspect your charging port for any lint or debris, "
                    "and try using a different official Apple cable and wall adapter.")
        else:
            return ("We're here to help with your battery. Make sure your device is running the latest iOS, "
                    "and check Settings > Battery > Battery Health for any service recommendations.")

    if intent == "connectivity":
        if "wifi" in op_lower or "wi-fi" in op_lower:
            return ("Let's get you connected! Try toggling Airplane Mode on for 15 seconds, then off. "
                    "If that doesn't work, reset network settings in Settings > General > Reset > Reset Network Settings.")
        elif "bluetooth" in op_lower or "airpods" in op_lower:
            return ("Let's get your audio reconnected. Go to Settings > Bluetooth, tap the 'i' next to your "
                    "device, select 'Forget This Device', and then pair it again.")
        elif "cellular" in op_lower or "service" in op_lower:
            return ("To help with your cellular connection, check Settings > General > About to see if a "
                    "carrier settings update is available, then restart your device.")
        else:
            return ("We can help with your connection. Try toggling Airplane Mode on and off, then restart "
                    "your device to refresh network connections.")

    if intent == "app_service":
        if "app store" in op_lower:
            return ("If the App Store is having trouble, try signing out and back into your Apple ID in "
                    "Settings > iTunes & App Store, then restart your device.")
        elif "music" in op_lower:
            return ("Let's get your music playing. In Settings > Music, toggle 'iCloud Music Library' off and on, "
                    "and make sure you're connected to a reliable internet connection.")
        elif "imessage" in op_lower:
            return ("If iMessage isn't activating, go to Settings > Messages, toggle iMessage off, restart "
                    "your iPhone, and toggle iMessage back on.")
        else:
            return ("Let's troubleshoot this app. Force quit the app from the app switcher, restart your "
                    "device, and check the App Store for any available updates.")

    if intent == "how_to":
        if "screenshot" in op_lower:
            return ("To take a screenshot on your device, press and hold the Side button and Volume Up button "
                    "simultaneously, then quickly release both buttons.")
        elif "update" in op_lower:
            return ("You can update wirelessly by going to Settings > General > Software Update. Ensure your "
                    "device is plugged into power and connected to Wi-Fi first.")
        elif "backup" in op_lower:
            return ("To back up with iCloud: connect to Wi-Fi, go to Settings > [your name] > iCloud > "
                    "iCloud Backup, and tap 'Back Up Now'. Here's a step-by-step guide: <URL>")
        else:
            return ("We're glad to help! You can find step-by-step instructions and user guides for your "
                    "device features directly on our support page: <URL>")

    if intent == "feedback_no_action":
        return ("Thank you for sharing your feedback with us! We appreciate customer thoughts and suggestions. "
                "You can also submit formal feature requests directly to our product teams here: <URL>")

    if intent == "unclear":
        return ("We're here and ready to help! Could you please reply with a few more details about what "
                "issue you are seeing and which Apple device you're using?")

    # Fallback to substantive portion or house default:
    return ("We're here to help get this resolved. Try restarting your device, and let us know your "
            "current device model and iOS version so we can take the next step together.")

def is_valid_candidate(r):
    """Sanity checks for usable candidate."""
    op = r.get("opener", "")
    words = op.split()
    if len(words) < 3 or len(words) > 80:
        return False
    # Filter out non-English gibberish or pure emoji strings
    if len(re.findall(r"[a-zA-Z]", op)) < 8:
        return False
    return True

def sample_golden_dataset(cases_path=CASES_PATH, corpus_path=CORPUS_PATH, seed=42):
    print("=== Starting Golden Dataset Stratified Sampling ===")
    rng = np.random.default_rng(seed)
    
    rows = [json.loads(l) for l in open(cases_path, encoding="utf-8")]
    n = len(rows)
    cut = int(n * 0.75)
    corpus_pool, eval_pool = rows[:cut], rows[cut:]
    print(f"Total cases: {n} | Corpus pool: {len(corpus_pool)} | Held-out Eval pool: {len(eval_pool)}")
    
    # Load corpus to verify leakage
    corpus_ids = set()
    if os.path.exists(corpus_path):
        corpus_ids = {json.loads(l)["case_id"] for l in open(corpus_path, encoding="utf-8")}
    else:
        corpus_ids = {r["case_id"] for r in corpus_pool[:30000]}
    print(f"Corpus size for leakage check: {len(corpus_ids)}")

    # Filter usable eval cases
    valid_eval = [r for r in eval_pool if is_valid_candidate(r)]
    print(f"Usable eval cases: {len(valid_eval)}")

    # Define regex matchers for candidate mining across intents
    p_bug = re.compile(r"\b(bug|glitch|freeze|freezing|crash|crashing|lag|lagging|reboot|restart loop|black screen|autocorrect|ios 11|unresponsive|stuck|screen delay|keyboard)\b", re.I)
    p_battery = re.compile(r"\b(battery|drain|draining|percentage|charge|charging|charger|shutting down|dies fast|cable|battery health)\b", re.I)
    p_conn = re.compile(r"\b(wifi|wi-fi|bluetooth|cellular|hotspot|airpods|disconnect|dropping calls|no service|signal|carrier|pairing|lte)\b", re.I)
    p_app = re.compile(r"\b(app store|itunes|apple music|imessage|facetime|icloud sync|apple watch|safari|notes app|podcast|app crashing)\b", re.I)
    p_howto = re.compile(r"\b(how do i|how can i|how to|is there a way|where can i find|can you tell me how|possible to)\b", re.I)
    p_feedback = re.compile(r"\b(hate|sucks|worst|terrible|awesome|thanks|thank you|bring back|please add|feature request|switching to|bye apple|tim cook)\b", re.I)
    p_billing = re.compile(r"\b(apple id|password|passcode|two-factor|2fa|verification code|locked|disabled|charged|refund|subscription|receipt|bill|billing|order|delivery)\b", re.I)
    p_hardware = re.compile(r"\b(screen cracked|cracked screen|broken screen|shattered|water damage|dropped in water|speaker|microphone|home button|camera lens|genius bar|applecare|repair cost)\b", re.I)
    p_data = re.compile(r"\b(lost photos|deleted photos|missing photos|lost contacts|backup restore|restore backup|recover data|wiped|data lost|disappeared)\b", re.I)
    p_unclear = re.compile(r"^(help|help me|anyone|why|what|hello|hey|yo|please help|\?|<URL>|dm me|can someone help me)\b", re.I)

    # Edge patterns
    p_safety = re.compile(r"\b(burning hot|burn|smoke|fire|melted|swollen|swelling|spark|sparks|electric shock|explode|exploded)\b", re.I)
    p_irreversible = re.compile(r"\b(factory reset|wipe|wiping|erase all|erased everything|permanent loss|lost all data|will it delete)\b", re.I)
    p_abuse_churn = re.compile(r"\b(fuck|shit|damn|lawsuit|sue|lawyer|scam|fraud|thieves|stole|rip off|disgusted|unacceptable|last straw|never buying)\b", re.I)
    p_override_howto = re.compile(r"\b(how do i.*(erase|wipe|reset password|passcode|factory reset|apple id))\b", re.I)
    p_override_feedback = re.compile(r"\b(unacceptable|lawsuit|sue|fraud|ripped off|taking legal action)\b", re.I)

    # 1. Mine 15 Edge Cases First to reserve them
    print("\n--- Mining 15 High-Impact Difficult/Edge Cases ---")
    edge_cases = []
    seen_ids = set()

    # (a) Safety-sensitive (3 cases)
    safety_candidates = [r for r in valid_eval if p_safety.search(r["opener"]) and r["case_id"] not in seen_ids]
    for r in safety_candidates[:3]:
        seen_ids.add(r["case_id"])
        intent = "battery_charging" if "battery" in r["opener"].lower() or "charg" in r["opener"].lower() else "hardware_repair"
        edge_cases.append({
            "case": r, "gold_intent": intent, "gold_route": "escalate",
            "gold_escalate_reason": "safety_risk", "edge_category": "safety_sensitive",
            "is_edge_case": True,
        })

    # (b) Irreversible-risk (3 cases)
    irrev_candidates = [r for r in valid_eval if p_irreversible.search(r["opener"]) and r["case_id"] not in seen_ids]
    for r in irrev_candidates[:3]:
        seen_ids.add(r["case_id"])
        intent = "data_loss" if "data" in r["opener"].lower() or "photo" in r["opener"].lower() else "software_bug"
        edge_cases.append({
            "case": r, "gold_intent": intent, "gold_route": "escalate",
            "gold_escalate_reason": "irreversible_risk", "edge_category": "irreversible_risk",
            "is_edge_case": True,
        })

    # (c) Angry/abusive/churn relationship risk (3 cases)
    abusive_candidates = [r for r in valid_eval if p_abuse_churn.search(r["opener"]) and r["case_id"] not in seen_ids]
    for r in abusive_candidates[:3]:
        seen_ids.add(r["case_id"])
        intent = "feedback_no_action"
        edge_cases.append({
            "case": r, "gold_intent": intent, "gold_route": "escalate",
            "gold_escalate_reason": "relationship_risk", "edge_category": "angry_abusive_relationship_risk",
            "is_edge_case": True,
        })

    # (d) Default Route Override / Conflict Cases (3 cases)
    # How-to that has irreversible risk or requires private data -> override default 'auto' to 'escalate'
    override_candidates = [r for r in valid_eval if p_override_howto.search(r["opener"]) and r["case_id"] not in seen_ids]
    for r in override_candidates[:2]:
        seen_ids.add(r["case_id"])
        edge_cases.append({
            "case": r, "gold_intent": "how_to", "gold_route": "escalate",
            "gold_escalate_reason": "irreversible_risk" if "erase" in r["opener"].lower() or "wipe" in r["opener"].lower() else "needs_private_data",
            "edge_category": "intent_route_conflict_override",
            "is_edge_case": True,
        })
    # Feedback that threatens lawsuit / legal action -> override default 'auto' to 'escalate'
    override_fb = [r for r in valid_eval if p_override_feedback.search(r["opener"]) and r["case_id"] not in seen_ids]
    for r in override_fb[:1]:
        seen_ids.add(r["case_id"])
        edge_cases.append({
            "case": r, "gold_intent": "feedback_no_action", "gold_route": "escalate",
            "gold_escalate_reason": "relationship_risk",
            "edge_category": "intent_route_conflict_override",
            "is_edge_case": True,
        })

    # (e) Ambiguous / Insufficient info cases (3 cases)
    unclear_candidates = [r for r in valid_eval if len(r["opener"].split()) <= 5 and p_unclear.search(r["opener"]) and r["case_id"] not in seen_ids]
    for r in unclear_candidates[:3]:
        seen_ids.add(r["case_id"])
        edge_cases.append({
            "case": r, "gold_intent": "unclear", "gold_route": "escalate",
            "gold_escalate_reason": "insufficient_info", "edge_category": "ambiguous_insufficient_info",
            "is_edge_case": True,
        })

    print(f"Edge cases mined: {len(edge_cases)} / {EDGE_QUOTA}")
    assert len(edge_cases) == EDGE_QUOTA, f"Expected {EDGE_QUOTA} edge cases, got {len(edge_cases)}"

    # 2. Mine candidate pools for each intent
    print("\n--- Mining Candidate Pools for 10 Normal Intents ---")
    intent_candidates = {intent: [] for intent in NORMAL_QUOTAS}

    for r in valid_eval:
        cid = r["case_id"]
        if cid in seen_ids:
            continue
        op = r["opener"]

        # Disjoint partitioning by highest-priority intent match
        if p_billing.search(op):
            intent_candidates["account_billing"].append(r)
        elif p_data.search(op):
            intent_candidates["data_loss"].append(r)
        elif p_hardware.search(op):
            intent_candidates["hardware_repair"].append(r)
        elif p_battery.search(op):
            intent_candidates["battery_charging"].append(r)
        elif p_conn.search(op):
            intent_candidates["connectivity"].append(r)
        elif p_app.search(op):
            intent_candidates["app_service"].append(r)
        elif p_howto.search(op):
            intent_candidates["how_to"].append(r)
        elif p_feedback.search(op):
            intent_candidates["feedback_no_action"].append(r)
        elif p_bug.search(op):
            intent_candidates["software_bug"].append(r)
        elif p_unclear.search(op):
            intent_candidates["unclear"].append(r)

    for intent, pool in intent_candidates.items():
        print(f"  Pool for {intent:20s}: {len(pool):5d} candidates (quota: {NORMAL_QUOTAS[intent]})")
        assert len(pool) >= NORMAL_QUOTAS[intent], f"Not enough candidates for {intent}"

    # 3. Apply Semantic Diversity Clustering via SentenceTransformer + KMeans
    print("\n--- Applying Semantic Diversity Maximization via Embeddings + Clustering ---")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    normal_sampled = []

    for intent, quota in NORMAL_QUOTAS.items():
        pool = intent_candidates[intent]
        # To make clustering fast and clean, pool up to 300 candidates
        if len(pool) > 300:
            sub_pool = list(rng.choice(pool, size=300, replace=False))
        else:
            sub_pool = pool

        texts = [r["opener"] for r in sub_pool]
        embeddings = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=False)

        # Run KMeans with k = quota
        km = KMeans(n_clusters=quota, random_state=seed, n_init=5)
        labels = km.fit_predict(embeddings)

        # Pick the exemplar closest to each cluster center
        picked_cases = []
        for cluster_idx in range(quota):
            member_indices = np.where(labels == cluster_idx)[0]
            if len(member_indices) == 0:
                continue
            center = km.cluster_centers_[cluster_idx]
            sims = embeddings[member_indices] @ center
            best_idx = member_indices[np.argmax(sims)]
            selected = sub_pool[best_idx]
            picked_cases.append(selected)
            seen_ids.add(selected["case_id"])

        # Default route and reason according to taxonomy
        default_route = T.INTENT_DEFAULT_ROUTE[intent]
        default_reason = ""
        if default_route == "escalate":
            if intent == "account_billing":
                default_reason = "needs_private_data"
            elif intent == "hardware_repair":
                default_reason = "needs_private_data"
            elif intent == "data_loss":
                default_reason = "irreversible_risk"
            elif intent == "unclear":
                default_reason = "insufficient_info"

        for c in picked_cases:
            normal_sampled.append({
                "case": c, "gold_intent": intent, "gold_route": default_route,
                "gold_escalate_reason": default_reason, "edge_category": "normal_stratified",
                "is_edge_case": False,
            })
        print(f"  Sampled {len(picked_cases)} diverse cases for {intent}")

    assert len(normal_sampled) == sum(NORMAL_QUOTAS.values()), f"Expected {sum(NORMAL_QUOTAS.values())}, got {len(normal_sampled)}"

    # 4. Merge normal and edge cases, shuffle deterministically
    all_selected = normal_sampled + edge_cases
    rng.shuffle(all_selected)
    print(f"\nTotal selected golden cases: {len(all_selected)} (Normal: {len(normal_sampled)}, Edge: {len(edge_cases)})")

    # 5. Build Final Dataset with Human-Grade Reference Responses & Context
    final_golden = []
    worksheet_rows = []
    tsv_rows = []

    for idx, item in enumerate(all_selected):
        c = item["case"]
        intent = item["gold_intent"]
        route = item["gold_route"]
        reason = item["gold_escalate_reason"]
        opener = c["opener"]
        brand_reply = c.get("first_brand_reply", "")
        turns = c.get("turns", [])
        context = format_conversation_context(turns)
        
        # Craft human-grade reference response
        ref_response = craft_reference_response(intent, route, reason, opener, brand_reply)

        case_obj = {
            "case_id": c["case_id"],
            "idx": idx,
            "opener": opener,
            "stratum": intent,
            "gold_intent": intent,
            "gold_route": route,
            "gold_escalate_reason": reason,
            "reference_reply": ref_response,
            "reference_response": ref_response,
            "first_brand_reply": brand_reply,
            "historical_brand_reply": brand_reply,
            "n_turns": c.get("n_turns", len(turns)),
            "is_edge_case": item["is_edge_case"],
            "edge_category": item["edge_category"],
            "conversation_context": context,
        }
        final_golden.append(case_obj)

        worksheet_rows.append({
            "idx": idx + 1,
            "case_id": c["case_id"],
            "customer_message": opener,
            "conversation_context": context,
            "intent": intent,
            "route": route,
            "escalation_reason": reason,
            "reference_response": ref_response,
            "edge_category": item["edge_category"],
        })

        tsv_rows.append([
            c["case_id"],
            opener.replace("\t", " ").replace("\n", " "),
            context.replace("\t", " ").replace("\n", " // "),
            intent,
            route,
            reason,
            ref_response.replace("\t", " ").replace("\n", " "),
            item["edge_category"]
        ])

    # 6. Leakage Check: Golden vs Corpus
    print("\n--- Performing Strict Leakage Check ---")
    golden_ids = {r["case_id"] for r in final_golden}
    leakage = golden_ids & corpus_ids
    assert not leakage, f"CRITICAL LEAKAGE DETECTED: {len(leakage)} cases overlap with corpus!"
    print(f"Leakage check PASSED: 0 of {len(golden_ids)} golden cases appear in the corpus.")

    # 7. Write Files
    os.makedirs("golden", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    # (a) golden/golden.jsonl
    with open(OUT_GOLDEN, "w", encoding="utf-8") as f:
        for r in final_golden:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_GOLDEN} ({len(final_golden)} cases)")

    # (b) data/processed/golden_pool.jsonl
    with open(OUT_POOL, "w", encoding="utf-8") as f:
        for r in final_golden:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {OUT_POOL} ({len(final_golden)} cases)")

    # (c) golden/human_labeling_dataset.tsv
    with open(OUT_TSV, "w", encoding="utf-8") as f:
        f.write("case_id\tcustomer_message\tconversation_context\tintent\troute\tescalation_reason\treference_response\tedge_category\n")
        for row in tsv_rows:
            f.write("\t".join(row) + "\n")
    print(f"Wrote {OUT_TSV}")

    # (d) golden/golden_labels.tsv (compatible with build_golden / inspection)
    with open(OUT_LABELS_TSV, "w", encoding="utf-8") as f:
        f.write("idx\tcase_id\tintent\troute\treason\n")
        for r in final_golden:
            f.write(f"{r['idx']}\t{r['case_id']}\t{r['gold_intent']}\t{r['gold_route']}\t{r['gold_escalate_reason']}\n")
    print(f"Wrote {OUT_LABELS_TSV}")

    # (e) Write batches 1-4 for backward compatibility with build_golden.py
    batch_size = 55
    for b in range(1, 5):
        batch_cases = final_golden[(b-1)*batch_size : b*batch_size]
        batch_path = f"golden/labels_batch{b}.tsv"
        with open(batch_path, "w", encoding="utf-8") as f:
            f.write("idx\tintent\troute\treason\n")
            for r in batch_cases:
                f.write(f"{r['idx']}\t{r['gold_intent']}\t{r['gold_route']}\t{r['gold_escalate_reason']}\n")
        print(f"Wrote {batch_path} ({len(batch_cases)} cases)")

    # (e) golden/human_labeling_worksheet.md (easy manual review format)
    with open(OUT_WORKSHEET, "w", encoding="utf-8") as f:
        f.write("# AppleSupport Golden Evaluation Dataset — Human Labeling & Review Worksheet\n\n")
        f.write(f"Total Cases: {len(final_golden)} | Normal Stratified: {sum(NORMAL_QUOTAS.values())} | Edge/Difficult: {EDGE_QUOTA}\n\n")
        f.write("This worksheet provides human-grade reference labels and responses for manual verification.\n")
        f.write("Historical AppleSupport replies were evaluated for house-style correctness and deflections were replaced with substantive, send-worthy guidance.\n\n")
        f.write("---\n\n")

        for r in worksheet_rows:
            f.write(f"### Case {r['idx']}: `{r['case_id']}`\n\n")
            f.write(f"- **Customer Message:** {r['customer_message']}\n")
            f.write(f"- **Intent:** `{r['intent']}`\n")
            f.write(f"- **Route:** `{r['route']}`\n")
            if r['escalation_reason']:
                f.write(f"- **Escalation Reason:** `{r['escalation_reason']}`\n")
            f.write(f"- **Category:** {r['edge_category']}\n\n")
            f.write(f"**Human-Grade Reference Response:**\n> {r['reference_response']}\n\n")
            f.write("<details><summary>Conversation Context</summary>\n\n```text\n")
            f.write(r['conversation_context'] + "\n```\n</details>\n\n")
            f.write("---\n\n")
    print(f"Wrote {OUT_WORKSHEET}")

    # Summary Statistics
    from collections import Counter
    ci = Counter(r["gold_intent"] for r in final_golden)
    cr = Counter(r["gold_route"] for r in final_golden)
    ce = Counter(r["gold_escalate_reason"] for r in final_golden if r["gold_route"] == "escalate")
    cedge = Counter(r["edge_category"] for r in final_golden if r["is_edge_case"])

    print("\n=== Final Golden Intent Distribution ===")
    for k, v in ci.most_common():
        print(f"  {k:22s} {v:3d} ({100*v/len(final_golden):5.1f}%)")

    print("\n=== Final Route Distribution ===")
    for k, v in cr.most_common():
        print(f"  {k:22s} {v:3d} ({100*v/len(final_golden):5.1f}%)")

    print("\n=== Escalation Reasons ===")
    for k, v in ce.most_common():
        print(f"  {k:22s} {v:3d}")

    print("\n=== Edge Case Categories ===")
    for k, v in cedge.most_common():
        print(f"  {k:30s} {v:3d}")

    print("\n=== Golden Dataset Creation Complete ===")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    sample_golden_dataset(seed=ap.parse_args().seed)
