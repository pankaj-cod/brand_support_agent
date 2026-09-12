# Decision log

The non-obvious calls, and why. Numbers referenced here are reproduced by
`src/` scripts in the order given in the README.

**1. Brand = AppleSupport, chosen by measurement rather than volume.**
AmazonHelp has 169k outbound tweets to Apple's 107k, but I scored every candidate
brand for *resolution richness* (`src/brand_quality.py`): does a reply contain
actionable steps and avoid punting to another channel? Amazon scores 10.2%
substantive, Apple 18.5%. On absolute count of substantive replies — the thing
that actually feeds retrieval — Apple wins with ~20k against Amazon's ~17k, and
Apple's problem space is far more self-contained. AskPlayStation scored highest
by rate (27.8%) but has only 19k replies total.

**2. Apple's own "let's move to DM" habit is a natural escalation signal.**
52.5% of AppleSupport replies deflect to a private channel. That is not noise,
it is the brand telling us which cases it believes cannot be closed in public.
I use it as weak supervision to sanity-check my escalation policy, and I report
where my policy disagrees with Apple's behaviour rather than assuming Apple is right.

**3. The unit of work is a "case", not a tweet.**
TWCS is a flat table with parent/child links. I reconstruct threads
(`src/prepare_data.py`) and reduce each to the customer's opening message plus
the brand's reply chain: 2.81M tweets → 80,176 usable AppleSupport cases. The
agent is scored on what it can see at first contact, which is the opener alone.

**4. The taxonomy is induced, then hand-consolidated — not invented, and not left as clusters.**
I embedded 12k openers and ran KMeans (k=28). The clusters came out *topic*-shaped
(iOS 11, battery, the "I" glitch) and highly redundant — four separate clusters
were all the same autocorrect bug. Shipping clusters as intents would have given
a taxonomy that splits on subject matter while a support org needs to split on
*what the agent must do*. So I collapsed 28 clusters into 10 intents that differ
in required action. Clusters were an input to the design, not the output.

**5. Ten intents, including two that are not problems.**
`feedback_no_action` (venting, feature requests, churn threats) and `unclear`
exist because ~16% of real traffic is one of those two. A taxonomy with only
"real" issues forces the classifier to mislabel a sixth of its input, and the
agent then confidently troubleshoots a rant.

**6. `feedback_no_action` was renamed mid-labelling.**
It started as `complaint_no_request`. Around case 31 I hit "please consider a
virtual Home button" — a polite feature request with no complaint in it. Rather
than stretch the definition, I renamed the intent and re-read the earlier cases
against the new definition.

**7. `safety_risk` was added to the escalation reasons mid-labelling.**
Case 154 ("won't turn on, gets burning hot when I plug it in") and case 169 (an
Arabic tweet about a phone that caught fire) do not belong in any troubleshooting
bucket. A device that is overheating is not a ticket. Three golden cases carry it.

**8. Escalation is decided on risk and capability, never on difficulty.**
Four grounds only: needs private data, irreversible risk, relationship risk,
safety risk — plus `insufficient_info` and `no_known_remedy`. "This looks hard"
is explicitly not a reason, because that is how an escalation policy silently
becomes "escalate everything" and the automation stops paying for itself.

**9. Escalation recall is the headline metric, not accuracy.**
The two errors are not equal. Routing a data-loss case to a bot that says
"try restarting" is a different kind of wrong from escalating a rant. So
`src/metrics.py` reports recall on the escalate class with a bootstrap CI, plus
raw counts of missed and over-escalations, and refuses to summarise routing as
one accuracy number.

**10. Golden labels are hand-assigned by a person reading each message — never LLM-generated.**
This is the load-bearing decision in the whole evaluation. If an LLM produced the
labels, the headline "the agent agrees with the labels 70% of the time" would
measure agreement between two LLMs, not correctness. All 220 were read and
labelled from the customer message *alone* (`golden/labels_batch*.tsv`), without
looking at Apple's actual reply, because the agent does not get to see that either.

**11. The golden set is sampled stratified, not uniformly.**
Uniform sampling gives a golden set that is roughly a third iOS-11-autocorrect
complaints, because that is what October 2017 was. That would let a
majority-class predictor look respectable and would leave `data_loss` with one or
two examples. I allocate by sqrt of cluster size with a floor of 3
(`src/make_splits.py`), which damps the giant clusters without pretending the
true distribution is uniform. **Consequence: golden-set rates are not traffic rates**
and must not be read as "x% of Apple's volume is escalation-worthy".

**12. Retrieval corpus and golden set are disjoint, asserted in code.**
The agent grounds replies in historical replies. If a golden case were in the
corpus, the agent could retrieve the exact reply it is being graded against.
`src/make_splits.py` partitions first and then asserts the intersection is empty.

**13. Retrieval filters for substantive replies (threshold 0.8).**
Only 26.3% of corpus replies clear the bar. Retrieving unfiltered neighbours
grounds the agent in "DM us" boilerplate and teaches it to deflect — the exact
behaviour the product is supposed to replace.

**14. Classification and drafting are separate LLM calls.**
One blended call is cheaper. Two calls mean that when a reply is bad I can tell
whether the intent was wrong or the grounding was wrong. The eval is the
deliverable here, so I paid for the separation.

**15. The judge is reference-free and a different model family from the drafter.**
Reference-free because Apple's actual reply is a deflection about half the time —
scoring similarity to it would reward deflection. Different model
(generator `openai/gpt-oss-120b`, judge `qwen/qwen3.8-27b`) because a model
grading its own prose style is a known bias. See the judge-agreement section for
how much that is worth.

**16. The judge's headline output is binary ("would you send this unedited?"), not a mean score.**
Averaged 1-5 rubric scores drift upward and do not map to any decision. Send-worthy
maps directly to "can this be deployed".

**17. Failed LLM calls are recorded as failures, never silently labelled.**
The first version fell back to `unclear`/`escalate` on an API error. Under rate
limiting that quietly converted 11 of 20 cases into fake escalations and would
have *inflated* escalation recall. Failures now propagate and the runner resumes.

**18. Every LLM call is cached on disk by content hash.**
An eval harness whose numbers move between runs cannot answer "did my change
help?". The cache makes reruns free and byte-identical.
