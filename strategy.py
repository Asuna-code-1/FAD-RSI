"""Train-only strategy memory. Retrieve from a snapshot, then evolve after the batch."""
import re
from collections import Counter
from difflib import SequenceMatcher

SEEDS = [
    ("algebra", "equation algebra equality expression", "Check equivalence before and after each transformation."),
    ("arithmetic", "integer sum product calculation", "Recompute intermediate quantities independently."),
    ("constraint", "positive distinct range condition", "List all constraints and check each candidate against them."),
]


def words(text):
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def entry(kind, conditions, instruction):
    return dict(kind=kind, conditions=conditions, instruction=instruction, tasks=[], helpful=0,
                harmful=0, failed=0, active=True, history=[["ADD", conditions, instruction]])


def seed_memory():
    return {k: entry(k, c, i) for k, c, i in SEEDS}


def retrieve(memory, text):
    query = Counter(words(text))
    scores = [(sum(query[w] for w in words(s["conditions"] + " " + s["kind"])), key)
              for key, s in memory.items() if s["active"]]
    score, key = max(scores, default=(0, None))
    return key if score else None


def evolve(memory, event):
    key = event.get("prior")
    if key:
        s = memory[key]
        s["helpful" if event["prior_ok"] else "failed"] += 1
        if event["prior_ok"] and event["task"] not in s["tasks"]:
            s["tasks"].append(event["task"])
        if s["failed"] >= 5 and s["helpful"] * 4 < s["failed"]:
            s["active"] = False
            s["history"].append(["DEPRECATE", event["task"]])
    d = event.get("diagnosis")
    if not d or not event["ok"]:
        return
    kind, cond, action = (d[k] for k in ("type", "conditions", "strategy"))
    # Merge equivalent instructions, even if the teacher used a different label.
    for existing, s in memory.items():
        if s["active"] and SequenceMatcher(None, action.lower(), s["instruction"].lower()).ratio() >= .86:
            key = existing
            if kind != s["kind"]:
                s["history"].append(["MERGE", kind, cond])
                s["conditions"] += " " + cond
            break
    else:
        key = kind
        if key in memory:
            s = memory[key]
            s["history"].append(["REFINE", cond, action])
            s.update(conditions=cond, instruction=action, active=True)
        else:
            memory[key] = entry(kind, cond, action)
    s = memory[key]
    if event["task"] not in s["tasks"]:
        s["tasks"].append(event["task"])
    s["helpful"] += 1
