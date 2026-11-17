"""Offline AI/source-assisted labels. Never called by image retrieval inference."""

import re

QUERIES = {
    "v8_red_bowl": [("form", "bowl"), ("attribute", "red")],
    "v8_green_plate": [("form", "plate"), ("attribute", "green")],
    "v8_glass_bottle": [("form", "vase"), ("material", "glass")],
    "v8_bronze_animal": [("form", "animal_sculpture"), ("material", "bronze")],
    "v8_floral_bowl": [("form", "bowl"), ("attribute", "floral")],
    "v8_lidded_porcelain_jar": [("form", "jar"), ("material", "porcelain"), ("attribute", "lid")],
    "v8_green_bowl": [("form", "bowl"), ("attribute", "green")],
    "v8_red_plate": [("form", "plate"), ("attribute", "red")],
    "v8_porcelain_cup": [("form", "cup"), ("material", "porcelain")],
    "v8_blue_white_vase": [("form", "vase"), ("attribute", "blue_white")],
    "v8_dragon_textile": [("form", "textile"), ("attribute", "dragon")],
    "v8_bronze_handles": [("form", "vessel"), ("material", "bronze"), ("attribute", "handles")],
}
VESSELS = {
    "vase",
    "jar",
    "bowl",
    "cup",
    "beaker",
    "ewer",
    "teapot",
    "censer",
    "brazier",
    "brush_washer",
    "box",
    "bird_feeder",
}
COLORS = {"red", "green", "blue_white"}


def form_state(forms, target):
    if not forms:
        return "unknown"
    accepted = VESSELS if target == "vessel" else {target}
    if not set(forms) & accepted:
        return "no"
    return "yes" if set(forms) <= accepted else "unknown"


def material_state(medium, target):
    text = medium.lower()
    if re.search(r"\b" + target + r"\b", text):
        return "yes"
    # Related terms (e.g. porcelaneous stoneware) are not a porcelain assertion.
    if target == "porcelain" and "porcelaneous" in text:
        return "unknown"
    if re.search(
        r"\b(porcelain|stoneware|earthenware|pottery|ceramic|bronze|iron|jade|stone|silk|ivory|marble|limestone|lacquer|glass|wood)\b",
        text,
    ):
        # Generic ceramic/metal does not resolve a narrower material subclass.
        if target == "porcelain" and re.search(r"\bceramic\b", text):
            return "unknown"
        return "no"
    return "unknown"


def judge_assisted(fact, item, qid, *, color_observable):
    evidence = []
    for kind, value in QUERIES[qid]:
        basis = "ai_visual_review"
        if kind == "form":
            state = form_state(fact.get("forms"), value)
        elif kind == "material":
            state = material_state(item.get("medium", ""), value)
            basis = "catalogue_medium"
        else:
            observed = fact.get(value)
            if (
                value in COLORS
                and not color_observable
                and (value != "blue_white" or observed is not False)
            ):
                state = "unknown"
            else:
                state = "unknown" if observed is None else "yes" if observed else "no"
            if value in COLORS and state == "unknown" and not color_observable:
                basis = "photo_color_unobservable"
        row = {"condition": kind + "/" + value, "state": state, "basis": basis}
        if kind == "material":
            row.update(catalogue_value=item.get("medium", ""), source_url=item.get("source_url"))
        evidence.append(row)
    states = [r["state"] for r in evidence]
    grade = 0 if "no" in states else -1 if "unknown" in states else 3
    photo_excluded = grade == -1 and any(r["basis"] == "photo_color_unobservable" for r in evidence)
    return {
        "grade": grade,
        "decision": grade == 3,
        "needs_review": grade == -1 and not photo_excluded,
        "reason": "explicit_mismatch"
        if grade == 0
        else "color_not_observable"
        if photo_excluded
        else "ambiguous"
        if grade == -1
        else "supported",
        "notes": fact["note"],
        "evidence": evidence,
        "source": "ai_visual_plus_catalogue",
        "reviewer": "root_ai",
        "human_confirmed": False,
        "independent_human_gold": False,
    }
