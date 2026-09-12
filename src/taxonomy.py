"""Intent taxonomy and escalation policy for the AppleSupport agent.

The taxonomy was INDUCED from the data (src/induce_taxonomy.py: embed 12k customer
openers -> KMeans k=28 -> inspect terms + exemplars), then hand-consolidated.
The clusters came out topic-shaped (iOS 11, battery, the "I" autocorrect glitch);
we collapsed them into intents that differ in WHAT THE AGENT MUST DO, because
routing, not topic tagging, is the job.
"""

INTENTS = {
    "software_bug": (
        "An Apple OS or built-in software is misbehaving: slowness, freezing, crashing, "
        "reboot loops, autocorrect/keyboard glitches, garbled characters, post-update "
        "regressions. The customer wants it fixed, not explained."),
    "battery_charging": (
        "Battery drains too fast, unexpected shutdowns at non-zero charge, device will not "
        "charge, or charger/cable problems."),
    "connectivity": (
        "Wi-Fi, Bluetooth, cellular, personal hotspot, AirPods/speaker audio routing, or "
        "call audio will not connect or keeps dropping."),
    "app_service": (
        "A specific Apple app or service is broken: Apple Music, App Store, iMessage, "
        "FaceTime, iCloud sync, Apple Watch, Apple TV, or a third-party app failing on the device."),
    "account_billing": (
        "Apple ID, password reset, two-factor codes, iCloud storage charges, subscriptions, "
        "refunds, orders, delivery, or the iPhone Upgrade Program."),
    "hardware_repair": (
        "Physical defect or damage: cracked or dead screen, broken speaker, swollen battery, "
        "water damage; or questions about repair, warranty, AppleCare, Genius Bar appointments."),
    "data_loss": (
        "Photos, messages, contacts, notes, or files have disappeared or been deleted, "
        "or a backup cannot be restored."),
    "how_to": (
        "An informational question about how something works or how to do something. "
        "Nothing is broken."),
    "feedback_no_action": (
        "Venting, sarcasm, praise, brand criticism, a churn threat, or a feature request -- "
        "anything with no specific problem the agent could act on."),
    "unclear": (
        "Too vague, truncated, or context-free to act on -- often an image or link with no "
        "description, or a bare 'help me'."),
}

# ---------------------------------------------------------------------------
# Escalation policy.
#
# We escalate on RISK and CAPABILITY, not on difficulty. Three grounds:
#   1. PRIVATE  - resolving it requires account-specific or personal data that must
#                 not be exchanged on a public timeline (Apple's own agents move
#                 these to DM, which is our weak-supervision signal).
#   2. IRREVERSIBLE - a wrong instruction destroys data or voids a repair. Bad advice
#                 here is far more expensive than a slow human reply.
#   3. RELATIONSHIP - the customer is abusive, threatening churn/legal action, or has
#                 already been failed once. A canned reply makes it worse.
#   4. SAFETY   - a physical hazard is described. Added during golden labelling: a
#                 device "burning hot" is not a troubleshooting ticket.
# Everything else is auto-handleable: public, reversible, first-line troubleshooting.
# ---------------------------------------------------------------------------
ESCALATE_REASONS = {
    "needs_private_data":   "Resolution requires account, order, or identity data that cannot be exchanged publicly.",
    "irreversible_risk":    "A wrong step risks permanent data loss or voids service/warranty.",
    "relationship_risk":    "Abuse, legal/churn threat, or a repeat failure; a templated reply would inflame it.",
    "insufficient_info":    "Not enough information to act, and the gap needs a human to unpick.",
    "no_known_remedy":      "No grounded historical resolution exists for this issue.",
    "safety_risk":          "Possible physical hazard (overheating, burns, swollen battery); needs a human immediately.",
}

# Default routing per intent. This is a PRIOR, not the decision -- the agent may
# override per case (e.g. an abusive how_to still escalates).
INTENT_DEFAULT_ROUTE = {
    "software_bug":         "auto",
    "battery_charging":     "auto",
    "connectivity":         "auto",
    "app_service":          "auto",
    "how_to":               "auto",
    "feedback_no_action":   "auto",
    "account_billing":      "escalate",
    "hardware_repair":      "escalate",
    "data_loss":            "escalate",
    "unclear":              "escalate",
}

INTENT_NAMES = list(INTENTS)
ROUTES = ["auto", "escalate"]

def taxonomy_prompt_block() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in INTENTS.items())

def escalation_prompt_block() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in ESCALATE_REASONS.items())
