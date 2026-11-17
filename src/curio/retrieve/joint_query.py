"""Language-independent visual query plans; no candidate catalogue inputs."""

from dataclasses import asdict, dataclass
import hashlib
import json
import re

from .condition_boundary import CONDITIONS
from .condition_parser import parse_conditions

VERSION = "joint-query-v1"
DEFINITIONS = {k: values[0][0] for k, values in CONDITIONS.items()}
DEFINITIONS.update(
    {
        "form/meiping": "a meiping vase: small mouth, short neck, broad high shoulders, and a body tapering toward the foot; not merely any narrow-necked bottle",
        "form/gourd": "a gourd-shaped bottle or vase with distinct rounded upper and lower lobes joined by a constricted waist; not a single round body with a plain tubular neck",
        # Definitions grounded in the museum sources recorded in CONCEPT_SOURCES.
        "form/jade_cong": "a jade cong-form tube with a square or squarish exterior surrounding a circular central opening",
        "form/moonflask": "a moonflask (baoyueping): a flask with a flattened body, a broad circular front, and a neck rising from its upper edge",
        "shape/narrow_mouth": "a vessel whose actual mouth opening is distinctly narrower than its main body; do not confuse a long neck with a small mouth or a lid with a visible opening",
        "shape/open_mouth": "with an uncovered open mouth",
        "shape/round_body": "with a rounded body",
        "tag/blue_white": "a ceramic object with blue painted decoration on a white or pale ground, the visible appearance of blue-and-white ceramics; black or gray decoration in a monochrome photo is insufficient",
    }
)
for color in ["red", "green", "blue", "white", "black", "yellow", "brown", "pink"]:
    DEFINITIONS[f"tag/{color}"] = (
        f"with a predominantly {color} surface on the object itself, not just the background or a small colored motif"
    )
COLOR_KEYS = {
    "tag/" + c
    for c in ["red", "green", "blue", "white", "black", "yellow", "brown", "pink", "blue_white"]
}
SPECIAL = {
    "form/meiping": r"梅瓶|\bmeiping(?:\s+vase)?\b|\bplum(?:\s+blossom)?\s+vase\b",
    "form/gourd": r"葫芦瓶|\b(?:double\s+)?gourd(?:\s+shaped)?(?:\s+(?:vase|bottle))?\b",
    "form/jade_cong": r"玉琮|\bjade\s+cong\b",
    "form/moonflask": r"抱月瓶|包月瓶|\bmoon\s*flask\b|\bbaoyueping\b",
}
CONCEPT_SOURCES = {
    "form/jade_cong": "https://www.metmuseum.org/art/collection/search/72376",
    "form/moonflask": "https://www.britishmuseum.org/sites/default/files/2021-05/large_print_guide_room_95_chinese_ceramics.pdf",
    "form/meiping": "https://english.chnmus.net/en/collection/details.html?id=418104689879684282",
    "tag/blue_white": "https://www.vam.ac.uk/articles/chinese-blue-and-white-ceramics",
}


@dataclass(frozen=True)
class JointQuery:
    canonical: str
    requires_color: bool
    clauses: tuple[str, ...] = ()
    origin: str = "concept_catalog"

    @property
    def key(self):
        return hashlib.sha256(
            json.dumps({"version": VERSION, **asdict(self)}, sort_keys=True).encode()
        ).hexdigest()

    def payload(self):
        return {**asdict(self), "key": self.key, "version": VERSION}

    @property
    def question(self):
        return (
            "Judge only what is visibly supported by this photograph of a museum object. "
            "Ignore any text labels or catalogue identity. Does the depicted object satisfy ALL of the following description?\n"
            + self.canonical
            + "\nAnswer exactly Yes or No. Answer No if a required feature is absent or cannot be verified from this photograph."
        )


def compile_known(query: str) -> JointQuery | None:
    """Return None for free-form syntax so a language planner can preserve it.

    None is NOT an empty result and must never be converted to all-No.
    """
    text = re.sub(r"[-–]", " ", query.strip().lower())
    if not text:
        raise ValueError("query must contain non-whitespace text")
    special = []
    for key, pattern in SPECIAL.items():
        if re.search(pattern, text):
            special.append(key)
            text = re.sub(pattern, " ", text)
    parsed = (
        parse_conditions(text)
        if text.strip()
        else {"conditions": [], "fallback_reasons": [], "unverified_modifiers": []}
    )
    if parsed["fallback_reasons"]:
        return None
    keys = special + ["/".join(c) for c in parsed["conditions"]] + parsed["unverified_modifiers"]
    keys = ["tag/blue_white" if k == "blue_white_technique/yes" else k for k in keys]
    if special:
        keys = [k for k in keys if k != "form_family/bottle_vase"]
    keys = tuple(sorted(set(keys)))
    if not keys or any(k not in DEFINITIONS for k in keys):
        return None
    return JointQuery("; ".join(DEFINITIONS[k] for k in keys), bool(set(keys) & COLOR_KEYS), keys)


def plan_free_query(query, planner):
    """A generated translation preserves novel names, negation and relations."""
    known = compile_known(query)
    if known:
        return known
    result = planner(query)
    canonical = result.get("canonical", "")
    if not isinstance(canonical, str) or not 3 <= len(canonical.strip()) <= 1800:
        raise ValueError("查询语义转换失败，请换一种表达重试。")
    if type(result.get("requires_color")) is not bool:
        raise ValueError("查询语义转换缺少颜色可观察性信息。")
    # Negating a known visible feature does not make the query historical or
    # unverifiable by definition. Retain the negation in the actual model input.
    positive_form = re.sub(r"\b(?:without|not|no)\b", " ", canonical, flags=re.IGNORECASE)
    known_visual = compile_known(positive_form)
    if result.get("visual") is not True and not (positive_form != canonical and known_visual):
        raise ValueError("这条查询包含无法凭照片核实的要求，请改为描述可见外观。")
    normalized = compile_known(canonical)
    if normalized:
        return normalized
    parsed_colors = {"/".join(c) for c in parse_conditions(canonical)["conditions"]} & COLOR_KEYS
    canonical = re.sub(r"\s+", " ", canonical.strip()).rstrip(".").lower()
    return JointQuery(
        canonical, bool(parsed_colors or result["requires_color"]), origin="model_translation"
    )
