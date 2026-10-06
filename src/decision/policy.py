"""Hard rules plus the three-lane decision, with a reason for every outcome.

Hard rules run first and can only make a decision more cautious: a ticket that matches one is never
auto-resolved, whatever the model says. They cover what a service desk would not hand to an AI on its own:
security incidents, privileged access, data deletion, ITIL changes and urgent tickets.
"""

import re
from dataclasses import dataclass, field

import numpy as np

# (name, pattern, lane). Patterns are case-insensitive and match whole words.
RULES = [
    ("security incident", r"phishing|malware|ransomware|virus|hacked|compromised|data breach|breach|"
                          r"suspicious (?:email|login|activity)|leaked|stolen (?:laptop|password|credentials)",
     "escalate"),
    ("privileged access", r"admin(?:istrator)? (?:rights|access|privileges)|root access|domain admin|"
                          r"elevated (?:rights|access|privileges)|grant (?:me )?access to", "assist"),
    ("data deletion", r"delete (?:all|the|my) (?:data|files|records|database)|drop (?:the )?database|wipe", "assist"),
    ("legal or personal data", r"gdpr|subject access request|legal hold|lawsuit|subpoena", "escalate"),
]
COMPILED = [(name, re.compile(rf"\b(?:{pattern})\b", re.IGNORECASE), lane) for name, pattern, lane in RULES]
LANE_ORDER = {"auto": 0, "assist": 1, "escalate": 2}


@dataclass
class Decision:
    lane: str                   # auto, assist or escalate
    probability: float          # estimated chance the retrieved articles solve the ticket
    reasons: list[str] = field(default_factory=list)


def rule_hits(text: str, priority: str | None = None, itil_type: str | None = None) -> list[tuple[str, str, str]]:
    """(rule name, matched text, lane) for every rule the ticket triggers."""
    hits = []
    for name, pattern, lane in COMPILED:
        m = pattern.search(text or "")
        if m:
            hits.append((name, m.group(0), lane))
    if itil_type and itil_type.strip().lower() == "change":
        hits.append(("ITIL change", itil_type, "assist"))  # changes need approval (change management)
    if priority and priority.strip().lower() in {"high", "critical", "urgent"}:
        hits.append(("urgent priority", priority, "assist"))
    return hits


def decide(probability: float, text: str, t_auto: float, t_escalate: float, priority: str | None = None,
           itil_type: str | None = None) -> Decision:
    p = float(probability)
    if p >= t_auto:
        lane, reason = "auto", f"confidence {p:.3f} is at or above the auto-resolve threshold {t_auto:.3f}"
    elif p >= t_escalate:
        upper = (f"below the auto-resolve threshold {t_auto:.3f}" if np.isfinite(t_auto)
                 else "no auto-resolve threshold met the error target on past tickets")
        lane, reason = "assist", f"confidence {p:.3f}; {upper}: draft an answer for an agent"
    else:
        lane, reason = "escalate", (f"confidence {p:.3f} is below {t_escalate:.3f}: the knowledge base "
                                    "probably has no answer")
    reasons = [reason]
    for name, matched, rule_lane in rule_hits(text, priority, itil_type):
        if LANE_ORDER[rule_lane] > LANE_ORDER[lane]:
            lane = rule_lane
        reasons.append(f"rule '{name}' matched '{matched}': at least {rule_lane}")
    return Decision(lane=lane, probability=round(p, 4), reasons=reasons)
