"""End-to-end wiring test with a stubbed LLM. Costs nothing, catches crashes."""
import json, sys, random, os
sys.path.insert(0, "src")
import llm

random.seed(0)
INTENTS = ["software_bug","battery_charging","app_service","hardware_repair",
           "account_billing","how_to","feedback_no_action","unclear","connectivity","data_loss"]

def fake_chat(messages, model=None, temperature=0.0, json_mode=False, **kw):
    sysmsg = messages[0]["content"]
    if "triage incoming customer tweets" in sysmsg:
        i = random.choice(INTENTS)
        esc = i in ("hardware_repair","account_billing","data_loss","unclear")
        return json.dumps({"intent": i, "route": "escalate" if esc else "auto",
                           "escalate_reason": "needs_private_data" if esc else "",
                           "confidence": 0.8, "rationale": "stub"})
    if "draft public reply tweets" in sysmsg:
        return json.dumps({"reply": "We're sorry about that. Try restarting your device, "
                                    "then let us know how it goes.", "grounded_in":[0]})
    return json.dumps({"grounded":3,"actionable":3,"tone":3,"policy":3,
                       "send_worthy": random.random()>0.35,
                       "failure_mode":"no_next_step","why":"stub"})

llm.chat = fake_chat
import agent, judge, retrieval

rows = [json.loads(l) for l in open("golden/golden.jsonl",encoding="utf-8")][:25]
ix = retrieval.build()
preds=[]
for r in rows:
    preds.append({"case_id":r["case_id"],"idx":r["idx"],"opener":r["opener"],
                  **agent.run_case(r["opener"], ix)})
os.makedirs("data/processed",exist_ok=True)
with open("data/processed/_smoke_preds.jsonl","w",encoding="utf-8") as f:
    for p in preds: f.write(json.dumps(p,ensure_ascii=False)+"\n")

jd=[{**p,"judge":judge.judge_one(p["opener"],p["reply"],p["route"])} for p in preds]
with open("data/processed/_smoke_judged.jsonl","w",encoding="utf-8") as f:
    for p in jd: f.write(json.dumps(p,ensure_ascii=False)+"\n")
print("smoke: agent + judge produced", len(jd), "rows")
print("sample:", json.dumps({k:jd[0][k] for k in ('intent','route','reply')}, ensure_ascii=False)[:200])
print("retrieved for case0:", len(preds[0]["retrieved"]), "hits")
