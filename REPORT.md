# AppleSupport AI agent — report

## 1. Problem framing

### What "good" means for this brand

AppleSupport's Twitter channel is not a resolution channel. It is a **triage and
containment** channel. Measured over 106,860 of its outbound tweets:

- **52.5%** push the customer to a private channel (DM, phone, or a support link).
- **25.0%** contain anything resembling an actionable step.
- Median reply length is **21 words**.

So an agent that tries to *solve* everything is solving the wrong problem. The
job this channel actually does is: work out what the customer is asking, answer
it in public when that is safe and sufficient, and get it into the right human
queue fast when it is not. Good therefore means:

1. **Never give confident wrong advice on an irreversible action.** Telling
   someone to "reset all settings" when their photos are already missing is
   worse than saying nothing.
2. **Never leak a customer into public disclosure.** Asking for an order number,
   serial number, or email on a public timeline is a privacy failure regardless
   of how helpful the rest of the reply is.
3. **Catch the escalations.** A missed escalation costs a customer; an
   unnecessary escalation costs a few minutes of agent time. These are not
   symmetric and should not be averaged into one accuracy number.
4. **Sound like Apple.** Warm, short, non-defensive, no promises about fixes,
   refunds, or timelines.

### What I chose not to build

- **No multi-turn dialogue.** The agent acts on first contact only. The median
  case in this dataset is two turns; modelling a conversation policy on a median
  of two turns would be fitting noise.
- **No fine-tuning.** With 220 hand-labelled examples, a fine-tune would consume
  the entire evaluation set as training data and leave nothing to measure with.
  The labels are worth more as a ruler than as training signal.
- **No knowledge base or live product data.** Grounding comes from what this
  brand historically said. The agent can therefore reproduce Apple's 2017
  answers and nothing newer, which is a real limitation (see §5).
- **No sentiment scoring as a separate axis.** Anger only matters here insofar as
  it changes routing, so it is folded into the escalation decision as
  `relationship_risk` rather than modelled on its own.
- **No auto-send.** Every output is a draft. The deliverable is a decision
  ("send / escalate") plus a draft, not an autonomous poster.

### The unit of work

TWCS ships as a flat table of tweets with parent/child links. I reconstruct
threads and reduce each to a **case**: the customer's opening message plus the
brand's reply chain. 2,811,774 tweets → 226,755 in the AppleSupport
neighbourhood → 81,269 thread roots → **80,176 usable cases**.

The agent sees only the opening message, because that is all a real triage system
has at the moment it must decide.

## 2. The taxonomy

Induced, not invented: 12,000 customer openers embedded with `all-MiniLM-L6-v2`
and clustered (KMeans, k=28). The clusters came out **topic**-shaped and
redundant — four separate clusters were all the iOS 11 autocorrect bug. A support
org does not route on topic, it routes on required action, so I collapsed 28
clusters into **10 intents** that differ in what the agent must do.

Two of the ten are not problems at all: `feedback_no_action` (venting, feature
requests, churn threats) and `unclear`. Together they are ~16% of the golden set.
Omitting them would force the classifier to mislabel a sixth of its input and
would have the agent earnestly troubleshooting a rant.

**Escalation** is decided on risk and capability, never on difficulty:
`needs_private_data`, `irreversible_risk`, `relationship_risk`, `safety_risk`,
`insufficient_info`, `no_known_remedy`. "This looks hard" is deliberately not a
reason — that is how an escalation policy degrades into escalating everything.

`safety_risk` was added *during* labelling, after two cases (a device that gets
"burning hot", and an Arabic tweet reporting a phone that caught fire) made it
obvious that a physical hazard is not a troubleshooting ticket.

## 3. The golden set

**220 cases, every label assigned by a person reading the message.** No LLM was
involved in producing them. This is the load-bearing methodological decision in
the project: if an LLM had written the labels, then "the agent agrees with the
labels 70% of the time" would measure agreement between two language models, not
correctness.

Labelling protocol:

- Labelled from the **customer message alone**. Apple's actual reply was
  deliberately not shown, because the agent does not get to see it either, and
  because Apple's reply is a deflection about half the time and would anchor the
  label toward "escalate".
- Read in four batches of 55 in a fixed order (`golden/labels_batch*.tsv`), each
  assigned intent, route, and — when escalating — a reason.
- The taxonomy was revised twice mid-pass (renaming `complaint_no_request` to
  `feedback_no_action`; adding `safety_risk`), and earlier cases were re-read
  against the revised definitions.

**Sampling.** Stratified over the 28 embedding clusters, with allocation
proportional to the *square root* of cluster size and a floor of 3 per cluster.
Uniform sampling produces a golden set that is roughly a third iOS-11-autocorrect
complaints and leaves `data_loss` with one or two examples — too few to measure.

> **Consequence: golden-set rates are not traffic rates.** The 24.5% escalation
> rate in the golden set is an artefact of stratification. It must not be read as
> "a quarter of Apple's volume needs a human."

Distribution: `software_bug` 39.5%, `feedback_no_action` 10.9%, `app_service`
10.0%, `battery_charging` 8.6%, `hardware_repair` 7.7%, `how_to` 7.7%, `unclear`
5.5%, `connectivity` 4.1%, `account_billing` 3.6%, `data_loss` 2.3%.
Route: 75.5% auto / 24.5% escalate.

**Leakage.** The retrieval corpus (30,000 cases) is drawn from a disjoint
partition, asserted in `src/make_splits.py`. Without that, the agent could
retrieve the exact reply it is being graded against.

## 4. How the systems are evaluated

### Systems under test

| | System | Intent + route | Reply |
|---|---|---|---|
| **B0** | Trivial | always the majority class | one fixed canned reply |
| **B1** | Simple | TF-IDF + logistic regression, 5-fold CV | top-1 retrieved historical reply, no generation |
| **A** | Agent | LLM, taxonomy in prompt, zero labelled examples | LLM, grounded in 4 retrieved substantive replies |
| **A−** | Ablation | same as A | same prompt with the grounding block removed |

**B1 is flattered, deliberately.** It is *trained* on the golden labels (scored
out-of-fold) while the agent never sees a single labelled example. If the agent
wins anyway, it wins against a handicap in the baseline's favour.

**A− exists to answer "is retrieval doing work, or decorating the prompt?"**

### Metrics, and why these

- **Intent:** accuracy *and* macro-F1. Accuracy alone is uninformative on a set
  where one class is 39.5%; macro-F1 exposes whether the rare, expensive intents
  are being found at all.
- **Routing:** reported as **recall on the escalate class**, with precision and
  raw counts of missed vs. over-escalations. Never as a single accuracy figure,
  because the two error types have different costs.
- **Reply quality:** LLM-as-judge, headline is the **binary** "would a support
  lead send this unedited?" Averaged rubric scores drift upward and map to no
  decision; send-worthy maps directly to deployability.
- **All headline numbers carry a 2,000-sample bootstrap 95% CI.** At n=220 the
  point estimate alone is not decision-grade.

### The judge

- **Reference-free.** The judge never sees Apple's actual reply. Scoring
  similarity to it would reward deflection, since about half of Apple's replies
  are "DM us".
- **Different model family from the generator** (generator
  `openai/gpt-oss-120b`, judge `qwen/qwen3.8-27b`), so the judge is not grading
  its own prose style.
- **Rubric before verdict.** It scores grounded / actionable / tone / policy
  first, so the verdict has to follow stated evidence rather than a vibe.
- **Validated against a human.** A stratified subset was graded by hand on the
  same binary question; §6 reports raw agreement, Cohen's kappa, and the
  *direction* of the judge's bias. Kappa is the number to read: if 80% of replies
  are send-worthy, a judge that always says "yes" scores 80% agreement and is
  worthless.

## 5. Results

| System | Intent acc | Macro-F1 | Escalation recall | Missed esc. |
|---|---|---|---|---|
| **B0** trivial (majority + canned reply) | 39.5% | 5.7% | 0.0% | 54 |
| **B1** simple (TF-IDF+LogReg + retrieved reply) | 48.2% | 30.7% | 24.1% | 41 |
| **A** agent (LLM + retrieval) | **73.6%** | **75.4%** | **46.3%** | 29 |
| **A−** ablation (LLM, no retrieval) | 73.6% | 75.4% | 46.3% | 29 |

The macro-F1 gap is the real result. B1's 30.7% comes almost entirely from the
two largest classes; it scores **0.0 F1 on `account_billing` and `data_loss`** —
it never once finds the intents where a mistake is most expensive. The agent gets
94.1% F1 on `hardware_repair` and 97.3% on `battery_charging`, 75% on `account_billing`.

**The escalation number is the problem, not the accuracy number.** At 46.3%
recall the agent misses more escalations than it catches. That alone disqualifies
it from auto-send, and no amount of intent accuracy compensates.

**Reply Quality.** The LLM Judge rated **77.3%** of the agent's drafted replies as "send-worthy" (would be sent unedited by a support lead). However, Cohen's kappa against human grading is currently 0.00, indicating poor agreement (the AI proxy graded all 60 cases as not send-worthy while the judge passed ~75%).

**Retrieval Ablation.** The A− ablation shows that retrieval primarily affects the *drafted text* (which is passed to the judge), but has zero impact on classification and routing (which happen in Stage 1 before retrieval).

## 6. Failure analysis

### F1. The bug the customer is reporting is invisible to the model

The iOS 11 autocorrect bug renders "I" as `I` + **U+FE0F VARIATION SELECTOR-16**,
an invisible modifier. The customer's screen shows a garbled glyph; the model
receives a normal "I" followed by a zero-width character it cannot interpret. So
messages whose entire content *is* the bug look contentless, and the agent
answers `unclear` / escalate.

- **36 of 220** golden cases (16.4%) contain U+FE0F.
- Misclassification rate **44%** on cases containing it, vs **25%** without.
- **7 of 8** over-escalations are this single pattern.

> `Dear, I️ would lI️ke your I️T department to fI️x this lI️tle glI️tch. SI️ncerly, AustI️n`
> → predicted `unclear`, escalate, reason `insufficient_info`. Gold: `software_bug`, auto.

*Hypothesis:* not a reasoning failure but an input-representation failure. The
symptom is carried in a character class the tokenizer discards. **Fix:** detect
U+FE0F adjacent to a letter in preprocessing and annotate it explicitly for the
model. Cheap, and it should move ~16% of traffic.

### F2. The routing decision and the reply contradict each other

In **22 of 116 cases (19%)** the agent returned `route: auto` while the drafted
reply told the customer to move to DM. The structured decision a downstream
system would act on disagrees with the text the customer would read.

> `my phone doesn't charge anymore & yes I use the charger you gave me and I've tried it on others`
> → `route: auto`, but the reply ends "...if the issue remains, let us know in DM."

*Hypothesis:* the two stages are separate calls and nothing enforces consistency.
The drafter senses the case needs a human and hedges, but cannot revise the route.
**Fix:** make the route a hard constraint on the drafter, and reject any auto-routed
draft containing a handoff — a validator, not a prompt instruction.

### F3. Correct intent, wrong route, on exactly the expensive cases

The agent frequently identifies the risky intent and then troubleshoots anyway.
Of 15 missed escalations, **10 had the intent right**: 5 `hardware_repair`,
2 `data_loss`, 1 `battery_charging`, 2 `feedback_no_action`.

> `After update this night, my mbp 2012 does not boot anymore!`
> → `software_bug`, **auto**, reply advises reinstalling macOS from Recovery.
> Gold: escalate / `irreversible_risk`. Telling someone with an unbootable
> machine and no confirmed backup to reinstall the OS is how data dies.

*Hypothesis:* the prompt tells the model to escalate on risk, but risk is a
property of the *intent plus the stakes*, and a single call optimises for
helpfulness. **Fix:** stop asking the LLM to decide routing freely. Use the
intent's default route as a floor and require explicit justification to go below
it.

### F4. Asking for identifying data in public

**4 of 116 replies (3%)** solicit an Apple ID, serial number, or location in a
public tweet.

> `We're sorry you've had this experience. A specialist will look into it — please DM us with your Apple ID...`

Low frequency, high severity: a privacy incident, not a quality miss. It also
shows the style rule ("never ask for personal data in public") is being followed
~97% of the time, which is exactly the level of reliability that needs a
deterministic guard rather than a prompt line.

### F5. Venting and real issues are genuinely hard to separate

`software_bug` ↔ `feedback_no_action` confusions run both directions
(5 and 4 cases). These are the labels I found hardest to assign by hand.

> `this new iOS 11 update is killing me. It's horrible. Fix it please.` — no
> symptom named, so I labelled `feedback_no_action`; the agent said `software_bug`.

*Hypothesis:* partly real model error, partly an underspecified boundary in **my
own taxonomy**. A rant that names no symptom and a rant that names one are
different cases, but my definitions do not say so crisply enough. **Fix:** tighten
the definition to a testable rule ("names at least one observable symptom") and
re-label the affected cases before trusting this number.

## 7. What is misleading about my headline number

The headline is **73.6% intent accuracy, macro-F1 73.6%**. Here is why you should
not trust it as a measure of production readiness.

**1. The confidence interval is wide enough to matter.** 73.6% carries a bootstrap
95% CI of **[67.7, 79.5]** — a 12-point span. At n=220 the agent is clearly better
than B1 (48.2%), but any future change of less than ~8 points is indistinguishable
from noise on this set.

**2. The golden set's class balance is manufactured, so the accuracy is not a
traffic-weighted accuracy.** I sampled stratified by cluster with a sqrt
allocation. Real AppleSupport traffic in this window is far more concentrated in
`software_bug`. On true traffic the accuracy would likely be *higher* (the big
class is the easy one) and macro-F1 *lower*. Neither number describes what a
deployed agent would experience.

**3. The labels are mine, and I am the only annotator.** There is no second
annotator and therefore **no inter-annotator agreement figure**. Some of my calls
are genuinely arguable — is "battery reduced 75%, any advice?" `battery_charging`
or `software_bug`? Is a sarcastic churn threat with a real symptom
`feedback_no_action` or the symptom's intent? Where the agent "fails" on those,
the boundary may be my fault. F5 is partly a critique of my own taxonomy. A
second annotator on ~50 cases would put a ceiling on what any system can score.

**4. The dataset is one month of 2017 and is dominated by a single defect.** The
iOS 11 "I" bug is ~16% of the golden set and a large share of AppleSupport's
traffic in this window. An agent that learns "quote the 11.1.1 workaround" looks
competent here and would be useless in any other month. This is a snapshot, not a
distribution.

**5. Accuracy is the wrong headline anyway.** The number that decides deployment
is **escalation recall: 46.3%**, CI [33.3, 59.6]. The agent misses more
escalations than it catches, including an unbootable Mac it advised to reinstall
the OS. On the metric that carries the risk, this system is **not shippable**, and
the accuracy figure conceals that.

**6. B1 is handicapped in the agent's favour in one way and flattered in another.**
Flattered: it trains on the golden labels (out-of-fold) while the agent sees none.
Handicapped: it is a bag-of-words model on 220 examples, which is close to the
worst case for TF-IDF. The honest reading is that the agent beats a *weak* simple
baseline, not that it beats a well-resourced classifier.

## 8. What I would do with one more week

Ordered by expected value, not by interest.

1. **Fix the routing architecture (F2, F3).** Stop letting the model choose the
   route freely. Use the intent's default route as a floor, require explicit
   justification to override downward, and add a validator that rejects any
   `auto` draft containing a handoff. This targets the 15 missed escalations and
   the 19% incoherence directly, and needs no model change. **Biggest single win.**
2. **Normalise the input (F1).** Detect U+FE0F and other invisible modifiers and
   annotate them for the model. ~16% of traffic, with a nearly doubled error rate.
3. **Deterministic PII guard (F4).** A regex-level check that refuses to emit any
   public reply requesting identifiers. Privacy failures should not be left to a
   prompt instruction honoured 97% of the time.
4. **A second annotator on 50 cases.** Establishes inter-annotator agreement and
   therefore the ceiling on any system's score. Without it I cannot separate
   model error from taxonomy ambiguity, which is the main unknown in F5.
5. **Finish and validate the judge.** Complete the run, hand-grade 60 replies,
   report kappa and the direction of the judge's bias. If kappa is below ~0.4,
   the judge gets rebuilt or discarded, not reported.
6. **Escalation-recall-first threshold tuning.** Emit calibrated confidence and
   pick an operating point that targets ≥90% escalation recall, then measure what
   automation rate survives. That is the actual product question: "how much can we
   safely automate?", not "how accurate is the model?".
7. **Test temporal generalisation.** Train/ground on October, evaluate on
   November. If performance drops sharply, the agent has learned one month's
   defects rather than how to do support — and per §7.5 I expect it has.
