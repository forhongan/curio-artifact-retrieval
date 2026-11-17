"""Museum-aware query plans; retain gourd subtype counts and bilingual meaning."""

import re

from .joint_query import COLOR_KEYS, DEFINITIONS, JointQuery, compile_known

VERSION = "catalogue-visual-v15"
DESCRIPTIONS = {
    **DEFINITIONS,
    "form/gourd_generic": "a gourd-shaped bottle or vase, with two or more swollen body lobes joined by constricted waists",
    "form/gourd_double": "a double-gourd bottle or vase with exactly two swollen body lobes joined by a constricted waist",
    "form/gourd_triple": "a triple-gourd bottle or vase with exactly three swollen body lobes joined by two constricted waists",
}
GOURD_PATTERNS = [
    (
        "form/gourd_triple",
        r"(?:三节|三節|三腹|三层|三層)葫芦瓶|\b(?:triple|three\s+(?:lobed|section))\s+gourd(?:\s+shaped)?(?:\s+(?:vase|bottle))?\b",
    ),
    (
        "form/gourd_double",
        r"(?:双腹|雙腹|双节|雙節|二节|两节)葫芦瓶|\b(?:double|two\s+(?:lobed|section))\s+gourd(?:\s+shaped)?(?:\s+(?:vase|bottle))?\b",
    ),
    ("form/gourd_generic", r"葫芦瓶|\bgourd(?:\s+shaped)?(?:\s+(?:vase|bottle))?\b"),
]


def compile_catalogue(query):
    text = re.sub(r"[-–]", " ", query.strip().lower())
    if not text:
        raise ValueError("query must contain non-whitespace text")
    gourd = None
    remainder = text
    for key, pattern in GOURD_PATTERNS:
        if re.search(pattern, text):
            gourd = key
            remainder = re.sub(pattern, " ", text)
            break
    if gourd:
        remainder = re.sub(r"\b(?:a|an|the)\b", " ", remainder).strip()
        rest = compile_known(remainder) if remainder.strip() else None
        if remainder.strip() and rest is None:
            return None  # Never silently drop an unknown qualifier or negation.
        clauses = tuple(sorted({gourd, *(rest.clauses if rest else ())}))
        if any(c.startswith("form/gourd") and c != gourd for c in clauses):
            return None
        if gourd != "form/gourd_generic":
            # A subtype must pass its parent category as well as the extra
            # lobe-count requirement; adding a condition cannot enlarge a set.
            clauses = tuple(sorted({*clauses, "form/gourd_generic"}))
    else:
        old = compile_known(query)
        if old is None:
            return None
        clauses = old.clauses
    return JointQuery(
        "; ".join(DESCRIPTIONS[c] for c in clauses),
        any(c in COLOR_KEYS for c in clauses),
        clauses,
        VERSION,
    )


def catalogue_category(item):
    """Read the main object noun; decorative subjects and descriptions are excluded."""
    families = [
        (
            "other",
            r"\b(?:mirrors?|box(?:es)?|seals?|coins?|plaques?|pendants?|jewelry|textiles?|robes?|scrolls?|lids?|stoppers?|covers?|censers?|burners?)\b|镜|印章|盒|盖子|服饰",
        ),
        (
            "figure",
            r"\b(?:figurine|figure|statuette|sculpture|statue|painting|portrait)s?\b|塑像|人物像|绘画",
        ),
        ("water_vessel", r"\b(?:water[ -]coupes?|brush[ -]washers?|water[ -]pots?)\b|笔洗|水丞"),
        ("cup", r"\bcups?\b|酒杯|茶杯|耳杯|高足杯"),
        ("bowl", r"\bbowls?\b|碗"),
        ("plate_dish", r"\b(?:plates?|dish(?:es)?|saucers?)\b|盘|碟"),
        ("pouring_vessel", r"\b(?:teapot|ewer|pitcher|jug|kettle)s?\b|茶壶|执壶"),
        ("jar", r"\bjars?\b|罐"),
        ("bottle_vase", r"\b(?:bottles?|vases?|flasks?|sprinklers?)\b|瓶"),
    ]
    for field in ("object_name", "title"):
        value = str(item.get(field) or "")
        head = re.split(r"\b(?:with|depicting|decorated)\b", value.lower(), maxsplit=1)[0]
        for family, pattern in families:
            if re.search(pattern, head):
                return {"family": family, "field": field, "value": value}
    return None


def catalogue_fact(item, clause):
    """Return support/contradiction/unknown with exact field provenance."""
    category = catalogue_category(item)
    title = str(item.get("title") or "")
    medium = str(item.get("medium") or "")

    def fact(status, field, value, reason):
        return {
            "condition": clause,
            "status": status,
            "source": "catalogue",
            "field": field,
            "value": value,
            "reason": reason,
        }

    families = {
        "form_family/bowl": "bowl",
        "form_family/bottle_vase": "bottle_vase",
        "form_family/plate_dish": "plate_dish",
        "form_family/pouring_vessel": "pouring_vessel",
        "form/jar": "jar",
    }
    if clause in families and category:
        return fact(
            "supported" if category["family"] == families[clause] else "contradicted",
            category["field"],
            category["value"],
            "museum_object_category",
        )
    if clause == "form/meiping" or clause.startswith("form/gourd_"):
        if category and category["family"] != "bottle_vase":
            return fact(
                "contradicted", category["field"], category["value"], "different_object_category"
            )
        head = re.split(r"\b(?:with|depicting|decorated)\b", title.lower(), maxsplit=1)[0]
        if category and clause == "form/meiping" and re.search(r"\bmeiping\b|梅瓶", head):
            return fact("supported", "title", title, "museum_specialist_category")
        if category and clause.startswith("form/gourd_") and re.search(r"\bgourd\b|葫芦", head):
            count = (
                3
                if re.search(r"\btriple\b|三节|三節|三腹", head)
                else 2
                if re.search(r"\bdouble\b|双腹|双节|雙節", head)
                else None
            )
            requested = {"form/gourd_double": 2, "form/gourd_triple": 3}.get(clause)
            if requested is None or count == requested:
                return fact("supported", "title", title, "museum_specialist_category")
            if count:
                return fact("contradicted", "title", title, "different_gourd_lobe_count")
    if clause == "material/porcelain" and medium:
        if re.search(r"\bporcelain\b|瓷", medium, re.I):
            return fact("supported", "medium", medium, "museum_material")
        if re.search(
            r"\b(?:stoneware|earthenware|glass|jade|bronze|wood|silver|gold|lacquer|textile)\b",
            medium,
            re.I,
        ):
            return fact("contradicted", "medium", medium, "different_material")
    if clause == "broad_material/ceramic" and re.search(
        r"porcelain|stoneware|earthenware|pottery|ceramic|陶|瓷", medium, re.I
    ):
        return fact("supported", "medium", medium, "museum_material")
    for key, pattern in {
        "material/bronze": r"\bbronze\b|青铜",
        "material/glass": r"\bglass\b|玻璃",
    }.items():
        if clause == key and re.search(pattern, medium, re.I):
            return fact("supported", "medium", medium, "museum_material")
    return fact("unknown", None, None, "no_explicit_catalogue_fact")
