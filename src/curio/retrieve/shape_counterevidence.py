"""Independent visual contradictions; no item-specific or catalogue rules."""

import math

from .joint_query import JointQuery

PROBES = {
    "parallel": "two separate vessel mouths are visible side by side, corresponding to two adjacent containers rather than a single vertical bottle",
    "accessory": "the main work is a human or animal figurine or sculpture holding or incorporating a small vessel as an accessory, rather than a stand-alone bottle or vase",
}


def counter_plan(name):
    # Reuse the exact diagnostic questions/cache. The version identifies the
    # question, not a claim that its empirical threshold is calibrated.
    return JointQuery(
        PROBES[name], False, ("v14-counter-diagnostic", name), "v14-counter-diagnostic"
    )


def apply_counterevidence(base_yes, scores):
    if type(base_yes) is not bool or set(scores) != set(PROBES):
        raise ValueError("Incomplete counterevidence")
    contradictions = []
    for name, value in scores.items():
        logit = value.get("yes_no_logit")
        if not isinstance(logit, (int, float)) or not math.isfinite(logit):
            raise ValueError("Invalid counterevidence inference")
        if logit >= 0:  # Fixed ordinary yes/no boundary, not tuned on test.
            contradictions.append(name)
    return {
        "yes": base_yes and not contradictions,
        "baseline_yes": base_yes,
        "contradictions": contradictions,
        "reason": "visual_contradiction"
        if contradictions
        else "supported"
        if base_yes
        else "baseline_insufficient_support",
    }
