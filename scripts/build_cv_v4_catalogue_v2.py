"""Versioned catalogue-first CV v4 builder with explicit semantic facets.

Official museum fields remain the first authority. This builder keeps raw
class, object form, material, technique, component role, and photographed
subject separate. Free prose is attributed sentence by sentence; unresolved
mentions remain unknown review material. AI supplements use a separate schema
and validator and never overwrite an official catalogue fact.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATALOG = ROOT / "data/snapshots/collection_v2.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts/cv_v4_20261108/catalogue_v2_ai_supplement"
LEGACY_BUILDER = ROOT / "scripts/build_cv_v4_official.py"
VERSION = "catalogue-first-builder-v2"


OBJECT_FORMS = (
    ("bottle vase", r"\b(?:bottle vases?|meiping)\b"),
    ("vase", r"\b(?:vases?)\b"),
    ("bottle", r"\b(?:bottles?|snuff bottle(?:s)?)\b"),
    ("jar", r"\b(?:jars?|hu)\b"),
    ("bowl", r"\b(?:bowls?|basins?)\b"),
    ("dish", r"\b(?:dishes|dish)\b"),
    ("plate", r"\b(?:plates?|platters?)\b"),
    ("cup", r"\b(?:cups?|teabowls?|tea bowls?)\b"),
    ("teapot", r"\bteapots?\b"),
    ("ewer", r"\bewers?\b"),
    ("jug", r"\bjug(?:s)?\b"),
    ("flask", r"\bflasks?\b"),
    ("censer", r"\b(?:censer|incense burner|incense vessel)\b"),
    ("figure", r"\b(?:figure|figures|figurine|figurines|statue|statues|buddha|bodhisattva|guanyin)\b"),
    ("mirror", r"\bmirrors?\b"),
    ("box", r"\bbox(?:es)?\b"),
    ("screen", r"\bscreens?\b"),
    ("stand", r"\bstands?\b"),
    ("robe", r"\b(?:robe|robes|garment|garments|skirt|skirts|boot|boots|jacket|jackets)\b"),
    ("textile", r"\b(?:textiles?|quilts?|fabric|embroidery|tapestry)\b"),
    ("buckle", r"\bbuckles?\b"),
    ("pendant", r"\bpendants?\b"),
    ("ornament", r"\b(?:ornaments?|jewelry|jewellery)\b"),
    ("tile", r"\btiles?\b"),
    ("brick", r"\bbricks?\b"),
    ("fragment", r"\bfragments?\b"),
    ("panel", r"\bpanels?\b"),
    ("rubbing", r"\brubbings?\b"),
    ("painting", r"\bpaintings?\b|\bink on (?:paper|silk)\b|\bhanging scroll\b"),
    ("book", r"\bbooks?\b|\balbums?\b|\bwoodblock printed\b"),
)

COMPONENT_PATTERNS = (
    ("stopper", r"\bstoppers?\b"),
    ("lid", r"\blids?\b"),
    ("cover", r"\bcovers?\b"),
    ("stand", r"\bstands?\b"),
    ("base", r"\bbases?\b"),
    ("container", r"\bcontainers?\b"),
    ("case", r"\bcases?\b"),
    ("mount", r"\bmounts?\b"),
    ("drawer", r"\bdrawers?\b"),
    ("sleeve panel", r"\bsleeve panels?\b"),
    ("ink cake", r"\bink cakes?\b"),
    ("ink stick", r"\bink sticks?\b"),
)

MATERIAL_PATTERNS = (
    ("porcelain", r"\bporcelain\b"),
    ("earthenware", r"\b(?:earthenware|terracotta|pottery)\b"),
    ("stoneware", r"\bstoneware\b"),
    ("ceramic", r"\b(?:ceramics?|porcelaneous)\b"),
    ("metal", r"\b(?:metal|brass)\b"),
    ("bronze", r"\bbronze\b"),
    ("jade", r"\b(?:jade|nephrite)\b"),
    ("glass", r"\bglass\b"),
    ("rock crystal", r"\brock crystal\b"),
    ("crystal", r"\bcrystal\b"),
    ("quartz", r"\bquartz\b"),
    ("agate", r"\bagate\b"),
    ("coral", r"\bcoral\b"),
    ("ivory", r"\bivory\b"),
    ("bamboo", r"\bbamboo\b"),
    ("boxwood", r"\bboxwood\b"),
    ("sandalwood", r"\bsandalwood\b"),
    ("lacquer", r"\blacquer\b"),
    ("silk", r"\bsilk\b"),
    ("cotton", r"\bcotton\b"),
    ("wool", r"\bwool\b"),
    ("satin", r"\bsatin\b"),
    ("brocade", r"\bbrocade\b"),
    ("textile", r"\btextile\b"),
    ("enamel", r"\benamel(?:led)?\b"),
    ("gold", r"\bgold\b"),
    ("gilt", r"\bgilt(?:ing|ed)?\b"),
    ("silver", r"\bsilver\b"),
    ("copper", r"\bcopper\b"),
    ("iron", r"\biron\b"),
    ("wood", r"\bwood(?:en)?\b"),
    ("stone", r"\bstone\b"),
    ("marble", r"\bmarble\b"),
    ("ink", r"\bink\b"),
    ("paper", r"\bpaper\b"),
)

TECHNIQUE_PATTERNS = (
    ("painted", r"\bpaint(?:ed|ing)?\b"),
    ("carved", r"\bcarv(?:ed|ing)\b"),
    ("incised", r"\bincis(?:ed|ing)\b"),
    ("molded", r"\b(?:molded|moulded|molding|moulding)\b"),
    ("relief", r"\brelief\b"),
    ("applied", r"\bapplied\b"),
    ("glazed", r"\bglazed\b"),
    ("underglaze", r"\bunderglaze\b"),
    ("overglaze", r"\boverglaze\b"),
    ("enameled", r"\benamel(?:led)?\b"),
    ("woven", r"\b(?:woven|weaving)\b"),
    ("weave", r"\bweaves?\b"),
    ("embroidered", r"\b(?:embroider(?:ed|y)|needleloop(?:ed)?|needle-looped)\b"),
    ("tapestry", r"\btapestry\b"),
    ("cast", r"\bcast(?:ing)?\b"),
    ("inlaid", r"\binlaid\b"),
    ("chased", r"\bchased\b"),
    ("blown", r"\bblown\b"),
    ("burnished", r"\bburnished\b"),
    ("decorated", r"\bdecorat(?:ed|ion|ing)\b"),
)

COLOR_PATTERNS = (
    ("blue", r"\bblue\b"),
    ("white", r"\bwhite\b"),
    ("green", r"\bgreen\b"),
    ("red", r"\bred\b"),
    ("black", r"\bblack\b"),
    ("yellow", r"\byellow\b"),
    ("brown", r"\bbrown\b"),
    ("gray", r"\b(?:gray|grey)\b"),
    ("purple", r"\bpurple\b"),
    ("pink", r"\bpink\b"),
    ("turquoise", r"\bturquoise\b"),
    ("amber", r"\bamber\b"),
)

MOTIF_PATTERNS = (
    ("dragon", r"\bdragons?\b"),
    ("lotus", r"\blotus(?:es)?\b"),
    ("floral", r"\b(?:flowers?|floral|blossoms?|peony|peonies|plum blossoms?|prunus|chrysanthemum|orchid)\b"),
    ("bird", r"\bbirds?\b"),
    ("phoenix", r"\bphoenix(?:es)?\b"),
    ("fish", r"\bfish(?:es)?\b"),
    ("horse", r"\bhorses?\b"),
    ("tiger", r"\btigers?\b"),
    ("lion", r"\blions?\b"),
    ("butterfly", r"\bbutterfl(?:y|ies)\b"),
    ("cloud", r"\bclouds?\b"),
    ("landscape", r"\blandscapes?\b"),
    ("figures", r"\b(?:figures?|buddha|bodhisattva|guanyin)\b"),
    ("geometric", r"\b(?:geometric|scrollwork|diapers?|lozenges?)\b"),
    ("deer", r"\b(?:deer|stag)\b"),
    ("leopard", r"\bleopards?\b"),
    ("qilin", r"\bqilin\b"),
    ("medallion", r"\bmedallions?\b"),
    ("scene", r"\bscene|scenes|procession|processions|hunting\b"),
)

ERAS = (
    ("neolithic", r"\b(?:neolithic|yangshao|longshan|liangzhu)\b"),
    ("shang", r"\bshang\b"),
    ("zhou", r"\bzhou\b"),
    ("han", r"\bhan\b"),
    ("tang", r"\btang\b"),
    ("liao", r"\bliao\b"),
    ("song", r"\bsong\b"),
    ("yuan", r"\byuan\b"),
    ("ming", r"\bming\b"),
    ("qing", r"\bqing\b"),
)

CANDIDATE_PATTERNS = {
    "vessel_form": (
        ("bottle", r"\bbottles?\b"), ("covered", r"\bcovered\b"),
        ("lidded", r"\b(?:lid|lidded)\b"), ("flask", r"\bflasks?\b"),
        ("buckle", r"\bbuckles?\b"), ("stopper", r"\bstoppers?\b"),
        ("spout", r"\bspouts?\b"), ("handle", r"\bhandles?\b"),
        ("neck", r"\bneck\b"), ("mouth", r"\bmouth\b"),
        ("foot", r"\bfoot\b"), ("rim", r"\brim\b"),
        ("lid", r"\blids?\b"), ("cover", r"\bcovers?\b"),
        ("stand", r"\bstands?\b"), ("base", r"\bbases?\b"),
        ("container", r"\bcontainers?\b"), ("mount", r"\bmounts?\b"),
    ),
    "surface_treatment": (
        ("painted", r"\bpaint(?:ed|ing)?\b"), ("carved", r"\bcarv(?:ed|ing)\b"),
        ("relief", r"\brelief\b"), ("incised", r"\bincis(?:ed|ing)\b"),
        ("molded", r"\b(?:molded|moulded)\b"), ("applied", r"\bapplied\b"),
        ("glazed", r"\bglazed\b"), ("enameled", r"\benamel(?:led)?\b"),
    ),
    "multiplicity": (
        ("pair", r"\bpairs?\b"), ("two", r"\btwo\b"),
        ("three", r"\bthree\b"), ("four", r"\bfour\b"),
        ("several", r"\bseveral\b"), ("set", r"\bset\b"),
    ),
    "orientation": (
        ("standing", r"\bstanding\b"), ("seated", r"\bseated\b"),
        ("vertical", r"\bvertical\b"), ("horizontal", r"\bhorizontal\b"),
        ("upright", r"\bupright\b"),
    ),
    "appearance_style": (
        ("blue-and-white", r"\bblue[ -]and[ -]white\b"),
        ("glaze", r"\bglazes?\b"), ("patterned", r"\bpatterned\b"),
        ("celadon", r"\bceladon\b"), ("qingbai", r"\bqingbai\b"),
        ("monochrome", r"\bmonochrome\b"), ("polychrome", r"\bpolychrome\b"),
    ),
    "motif_extension": (
        ("scene", r"\bscenes?\b|\bprocessions?\b|\bhunting\b"),
        ("medallion", r"\bmedallions?\b"), ("peony", r"\bpeon(?:y|ies)\b"),
        ("chrysanthemum", r"\bchrysanthemums?\b"), ("waves", r"\bwaves?\b"),
        ("deer", r"\b(?:deer|stag)\b"), ("qilin", r"\bqilin\b"),
    ),
}

AI_SUPPLEMENT_FIELDS = (
    "visible_form_detail", "visible_part", "visible_palette", "visible_motifs",
    "visible_material", "visible_surface", "visible_multiplicity", "visible_orientation",
    "photographed_subject", "photographed_component",
)

AI_OBSERVATION_STATUSES = (
    "catalogue_confirmed_visible",
    "image_only_obvious",
)

# These are surface treatments or decorative materials.  They can still be a
# material facet when explicitly present in support_materials or when the
# museum's raw type is that material; they are not used to override a concrete
# type such as Wood merely because the technique string mentions decoration.
DECORATIVE_MATERIALS = {"enamel", "gilt", "ink"}
CONCRETE_TYPE_MATERIALS = {
    "glass": "glass", "jade": "jade", "lacquer": "lacquer", "wood": "wood",
    "stone": "stone", "ivory": "ivory", "enamel": "enamel",
    "metalwork": "metal", "textile": "textile",
}

COMPARISON_MARKERS = re.compile(
    r"\b(?:other (?:objects?|works?|examples?|bowls?|vases?|jars?|stands?)|unique among|"
    r"typically|generally|examples? of|type of|similar|comparable|unlike|in contrast|"
    r"predecessors?|earlier|later|historical|influenced|tradition)\b",
    re.I,
)
GENERIC_SUBJECTS = re.compile(r"^(?:this|these) (?:type|types|example|examples|wares?|badges?|objects?)\b", re.I)
OBJECT_NOUNS = re.compile(
    r"\b(?:body|glaze|surface|exterior|interior|decoration|design|rim|mouth|neck|foot|"
    r"handles?|base|cover|lid|stand|vase|jar|bowl|cup|bottle|dish|plate|object|figure|"
    r"painting|textile|robe|box|panel|mirror)\b", re.I)
DIRECT_PREDICATE = re.compile(
    r"\b(?:is|are|has|have|shows?|depicts?|features?|displays?|exhibits?|decorated|"
    r"painted|carved|incised|glazed|covered|surrounded|adorned|outlined|set against|comprises?|consists?)"
    r"\b",
    re.I,
)
NONSURFACE_PHRASES = re.compile(
    r"\b(?:red tea|black tea|White Tiger|Green Dragon|Red Bird|Dark Warrior|"
    r"dragon kilns?|red herring|black bear)\b",
    re.I,
)
COLOR_TERM_TYPES = {
    "red": "surface_color", "yellow": "surface_color", "white": "surface_color",
    "black": "surface_color", "blue": "surface_color", "green": "surface_color",
    "brown": "surface_color", "gray": "surface_color", "purple": "surface_color",
    "pink": "surface_color", "turquoise": "colorant_or_pigment", "amber": "colorant_or_pigment",
}
MOTIF_TERM_TYPES = {
    "dragon": "decorative_figure", "lotus": "plant_motif", "floral": "plant_motif",
    "bird": "animal_motif", "phoenix": "mythical_animal", "fish": "animal_motif",
    "horse": "animal_motif", "tiger": "animal_motif", "lion": "animal_motif",
    "butterfly": "animal_motif", "deer": "animal_motif", "leopard": "animal_motif",
    "qilin": "mythical_animal", "cloud": "weather_motif", "landscape": "scene_motif",
    "figures": "human_or_divine_figure", "geometric": "geometric_motif",
    "medallion": "ornamental_unit", "scene": "narrative_scene",
}
ATTRIBUTION_BLOCKERS = re.compile(
    r"\b(?:usually|typically|generally|often|examples? of|types? of|"
    r"bowls? of this type|other (?:objects?|works?|examples?|bowls?|vases?|jars?|stands?)|"
    r"unique among|instead of|unlike|in contrast)\b",
    re.I,
)


def _sentence_core(sentence: str) -> str:
    """Remove parentheticals and subordinate comparisons before attribution."""
    core = re.sub(r"\([^()]*\)", " ", sentence)
    parts = re.split(r"\b(?:although|though|while|whereas|but)\b", core, maxsplit=1, flags=re.I)
    return parts[0].strip() or core.strip()


def first_sentence(value: str) -> str:
    return next((sentence for sentence, _, _ in sentences(value)), "")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(text(v) for v in value if text(v))
    if isinstance(value, dict):
        return " ".join(text(v) for v in value.values() if text(v))
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", str(value))).strip()


def normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def evidence_text(value: str) -> str:
    """Normalize line wrapping and spacing for exact raw-record quote checking."""
    return re.sub(r"[ \t\r\n]+", " ", value).strip()


def sentences(value: str) -> list[tuple[str, int, int]]:
    """Return sentence text and half-open offsets from cleaned source text."""
    cleaned = text(value)
    output = []
    for match in re.finditer(r"[^.!?]+(?:[.!?]+|$)", cleaned):
        body = match.group(0).strip()
        if body:
            output.append((body, match.start(), match.end()))
    return output or ([(cleaned, 0, len(cleaned))] if cleaned else [])


def spans(value: str, patterns) -> list[dict]:
    found = []
    for name, pattern in patterns:
        for match in re.finditer(pattern, value, re.I):
            found.append({"value": name, "match": match.group(0), "start": match.start(),
                          "end": match.end(), "evidence": value})
    return sorted(found, key=lambda row: (row["start"], row["end"], row["value"]))


def unique_values(facets: Iterable[dict]) -> list[str]:
    return list(dict.fromkeys(facet["value"] for facet in facets))


def broad_material(value: str) -> str:
    if value in {"porcelain", "earthenware", "stoneware", "ceramic"}:
        return "ceramic"
    if value in {"silk", "cotton", "wool", "satin", "brocade", "textile"}:
        return "textile"
    if value in {"metal", "bronze", "gold", "gilt", "silver", "copper", "iron"}:
        return "metal"
    if value in {"jade", "stone", "marble", "agate", "crystal", "rock crystal", "quartz"}:
        return "stone"
    return value


def sentence_attribution(sentence: str, title: str) -> dict:
    """Classify whether a prose sentence directly attributes a property."""
    stripped = _sentence_core(sentence)
    if not stripped:
        return {"attributed": False, "reason": "empty_after_scope_cleanup"}
    if ATTRIBUTION_BLOCKERS.search(sentence):
        return {"attributed": False, "reason": "generic_or_comparand_subject"}
    if GENERIC_SUBJECTS.search(stripped):
        return {"attributed": False, "reason": "generic_cohort_subject"}
    if COMPARISON_MARKERS.search(stripped) and not re.match(r"^(?:this|the|its|here)\b", stripped, re.I):
        return {"attributed": False, "reason": "comparison_or_other_object_context"}
    if NONSURFACE_PHRASES.search(stripped):
        # A sentence can still be direct about a painting or figure, but named
        # symbols such as White Tiger are not palette evidence.
        pass
    title_norm = normalized(title)
    title_anchor = bool(title_norm and title_norm in normalized(stripped))
    starts_direct = bool(re.match(r"^(?:this|these|the|its|here|a |an )\b", stripped, re.I))
    predicate = bool(DIRECT_PREDICATE.search(stripped))
    object_noun = bool(OBJECT_NOUNS.search(stripped))
    attributed = (starts_direct and (predicate or object_noun)) or title_anchor
    return {"attributed": attributed, "reason": "direct_record_subject" if attributed else "subject_not_proven"}


def facet_sources(facets: list[dict]) -> list[str]:
    return list(dict.fromkeys(source for facet in facets for source in facet.get("source_fields", [])))


class CatalogueBuilderV2:
    def __init__(self, root: Path = ROOT):
        self.root = Path(root)

    def raw_record(self, item: dict) -> dict:
        path = self.root / "data/raw" / item["source"] / f"{item['source_id']}.json"
        return json.loads(path.read_text(encoding="utf-8"))

    def catalogue_fields(self, item: dict, raw: dict) -> dict:
        if item["source"] == "met":
            tags = [text(x.get("term") if isinstance(x, dict) else x) for x in (raw.get("tags") or [])]
            return {
                "title": text(raw.get("title")),
                "object_name": text(raw.get("objectName")),
                "classification": text(raw.get("classification")),
                "medium": text(raw.get("medium")),
                "technique": "",
                "support_materials": [],
                "era": " . ".join(text(raw.get(k)) for k in ("period", "dynasty", "reign", "objectDate") if text(raw.get(k))),
                "tags": [value for value in tags if value],
                "description": "",
            }
        return {
            "title": text(raw.get("title")),
            "object_name": text(raw.get("type")),
            "classification": text(raw.get("collection")),
            "medium": "",
            "technique": text(raw.get("technique") or raw.get("medium")),
            "support_materials": [text(value) for value in (raw.get("support_materials") or []) if text(value)],
            "era": " . ".join(text(raw.get(k)) for k in ("culture", "creation_date", "date_text") if text(raw.get(k))),
            "tags": [text(value) for value in (raw.get("artists_tags") or []) if text(value)],
            "description": " ".join(text(raw.get(k)) for k in ("description", "did_you_know", "early_education_description") if text(raw.get(k))),
        }

    def _component(self, title: str) -> tuple[dict, str | None]:
        leading_component = re.match(r"^(stand|cover|lid|stopper|base|drawer|mount|case|container|sleeve panel)\s+for\s+(.+)$", title, re.I)
        if leading_component:
            role = normalized(leading_component.group(1))
            matched = next((value for value, pattern in COMPONENT_PATTERNS if re.search(pattern, role, re.I)), role)
            return ({"value": matched, "status": "catalogued_explicit_component",
                     "source_field": "title", "evidence": title, "image_verified": False},
                    leading_component.group(2).strip())
        final = re.search(r"\(([^()]+)\)\s*$", title)
        if final:
            role = final.group(1).strip().casefold()
            matched = next((value for value, pattern in COMPONENT_PATTERNS if re.search(pattern, role, re.I)), None)
            if matched:
                source = title[:final.start()].strip()
                return ({"value": matched, "status": "catalogued_explicit_component",
                         "source_field": "title", "evidence": title, "image_verified": False},
                        source)
        # Component records in grouped catalogue sets use a colon after the
        # parent object: `Box with Ink Cakes: Green Ink Cake ...`.
        colon_component = re.search(
            r"^(.+?):\s*(?:an?\s+)?(?P<role>[^:]+?)(?:\s+in\s+shape\s+of\b.*)?$",
            title,
            re.I,
        )
        if colon_component:
            role_text = colon_component.group("role").strip()
            matched = next(
                (value for value, pattern in COMPONENT_PATTERNS if re.search(pattern, role_text, re.I)),
                None,
            )
            if matched:
                return ({"value": matched, "status": "catalogued_explicit_component",
                         "source_field": "title", "evidence": title, "image_verified": False},
                        colon_component.group(1).strip())
        with_component = re.search(r"\bwith (?:an? )?(" + "|".join(value for value, _ in COMPONENT_PATTERNS) + r")\b", title, re.I)
        if with_component:
            role = with_component.group(1).casefold()
            return ({"value": f"whole_with_{role}", "status": "catalogued_component_present",
                     "source_field": "title", "evidence": title, "image_verified": False}, title)
        return ({"value": "whole_or_unspecified", "status": "catalogued_or_unspecified",
                 "source_field": "title", "evidence": title, "image_verified": False}, title)

    def _forms(self, item: dict, fields: dict, component_source: str | None) -> tuple[list[dict], list[dict]]:
        title = component_source or fields["title"]
        forms, unattributed = [], []
        if item["source"] == "met":
            object_kind = fields["classification"].casefold().strip() or fields["object_name"].casefold().strip()
            if object_kind in {"painting", "paintings", "album", "albums"}:
                value = "painting" if "painting" in object_kind else "book"
                forms.append({"value": value, "source_field": "classification" if fields["classification"] else "objectName",
                              "evidence": fields["classification"] or fields["object_name"],
                              "status": "catalogued_broad_class"})
        # Prefer the main noun before parenthetical relations or colon suffixes;
        # headlines such as `Panel ...: Dragon` have the panel as photographed form.
        main_title = re.sub(r"\s*\([^()]+\)\s*$", "", title).strip()
        head = re.split(r"\s*(?::|\bwith\b|\bfrom\b)\s*", main_title, maxsplit=1, flags=re.I)[0]
        head_hits = spans(head, OBJECT_FORMS)
        for hit in head_hits:
            forms.append({"value": hit["value"], "source_field": "title", "evidence": head,
                          "status": "catalogued_direct"})
        if not forms and item["source"] == "met":
            for hit in spans(fields["object_name"], OBJECT_FORMS):
                forms.append({"value": hit["value"], "source_field": "objectName", "evidence": fields["object_name"],
                              "status": "catalogued_direct"})
        if not forms and item["source"] == "met" and fields["classification"]:
            classification_map = {
                "snuff bottles": "bottle", "vases": "vase", "bowls": "bowl",
                "jars": "jar", "cups": "cup", "dishes": "dish", "plates": "plate",
            }
            class_form = classification_map.get(fields["classification"].casefold().strip())
            if class_form:
                forms.append({"value": class_form, "source_field": "classification",
                              "evidence": fields["classification"], "status": "catalogued_broad_class"})
        if not forms and item["source"] == "cleveland":
            type_value = fields["object_name"].casefold()
            map_value = {"jewelry": "ornament", "textile": "textile",
                         "garment": "robe", "embroidery": "textile", "tapestry": "textile"}.get(type_value)
            if map_value:
                forms.append({"value": map_value, "source_field": "type", "evidence": fields["object_name"],
                              "status": "catalogued_broad_class"})
        if not forms:
            # A catalogued type can itself be a form noun ("Snuff bottle",
            # "Jar"). It is a final authority fallback after the title head.
            for hit in spans(fields["object_name"], OBJECT_FORMS):
                forms.append({"value": hit["value"],
                              "source_field": "type" if item["source"] == "cleveland" else "objectName",
                              "evidence": fields["object_name"], "status": "catalogued_broad_class"})
        # Other title nouns are candidate forms, not formal object-form labels.
        all_hits = spans(fields["title"], OBJECT_FORMS)
        formal = {facet["value"] for facet in forms}
        for hit in all_hits:
            if hit["value"] not in formal:
                unattributed.append({"kind": "object_form_candidate", "value": hit["value"],
                                     "source_field": "title", "evidence": hit["evidence"],
                                     "reason": "outside_main_title_noun"})
        return forms, unattributed

    def _material_technique(self, item: dict, fields: dict) -> tuple[list[dict], list[dict], list[dict]]:
        materials, techniques, unattributed = [], [], []
        # CMA support_materials has priority by explicit user requirement. The
        # field being empty is meaningful; technique is never promoted across.
        for value in fields["support_materials"]:
            for hit in spans(value, MATERIAL_PATTERNS):
                materials.append({"value": hit["value"], "broad": broad_material(hit["value"]),
                                  "source_field": "support_materials", "evidence": value,
                                  "status": "catalogued_explicit_material"})
        if item["source"] == "met":
            # Enamel, gilding, lacquer, and ink are coatings/decoration here;
            # they remain technique evidence rather than the support material.
            support_exclusions = {"enamel", "gold", "gilt", "lacquer", "ink"}
            for hit in spans(fields["medium"], MATERIAL_PATTERNS):
                if hit["value"] in support_exclusions:
                    continue
                materials.append({"value": hit["value"], "broad": broad_material(hit["value"]),
                                  "source_field": "medium", "evidence": fields["medium"],
                                  "status": "catalogued_medium"})
        else:
            class_value = fields["object_name"].casefold().strip()
            class_material = {
                "glass": "glass", "jade": "jade", "lacquer": "lacquer", "wood": "wood",
                "stone": "stone", "ivory": "ivory", "ceramic": "ceramic", "enamel": "enamel",
                "metalwork": "metal", "textile": "textile",
            }.get(class_value)
            # A populated explicit support list wins over broad type/class
            # fallbacks. Empty support data keeps the material-class fallback.
            if not fields["support_materials"] and class_value in {"glass", "jade", "lacquer", "wood", "stone", "ivory", "ceramic", "enamel", "textile"}:
                materials.append({"value": class_material or class_value, "broad": broad_material(class_material or class_value),
                                  "source_field": "type", "evidence": fields["object_name"],
                                  "status": "catalogued_material_class"})
            elif not fields["support_materials"] and class_value == "metalwork":
                materials.append({"value": "metal", "broad": "metal", "source_field": "type",
                                  "evidence": fields["object_name"], "status": "catalogued_material_class"})
            # CMA's curated ``technique`` field often names the material and
            # construction together (for example "agate", "carved wood", or
            # "Silk: tabby, brocaded").  It is valid secondary material evidence
            # when support_materials is absent and the raw type is broad (for
            # example Ceramic, Jewelry, or Textile).  A concrete type remains
            # authoritative so a wooden stand is not relabelled as its porcelain
            # subject merely because the title mentions porcelain.
            if not fields["support_materials"] and class_value not in CONCRETE_TYPE_MATERIALS:
                explicit_technique_materials = []
                for hit in spans(fields["technique"], MATERIAL_PATTERNS):
                    if hit["value"] in DECORATIVE_MATERIALS and class_value != "enamel":
                        continue
                    explicit_technique_materials.append({
                        "value": hit["value"], "broad": broad_material(hit["value"]),
                        "source_field": "technique", "evidence": fields["technique"],
                        "status": "catalogued_explicit_technique_material",
                    })
                materials.extend(explicit_technique_materials)
            # An explicit material in the title is still catalogue authority.
            if not materials:
                excluded = {"enamel", "gold", "gilt", "ink", "textile"}
                for hit in spans(fields["title"], MATERIAL_PATTERNS):
                    if hit["value"] in excluded:
                        continue
                    materials.append({"value": hit["value"], "broad": broad_material(hit["value"]),
                                      "source_field": "title", "evidence": fields["title"],
                                      "status": "catalogued_title_material"})
            for hit in spans(fields["technique"], TECHNIQUE_PATTERNS):
                techniques.append({"value": hit["value"], "source_field": "technique",
                                   "evidence": fields["technique"], "status": "catalogued_technique"})
        # The catalogue technique string also contains material words. Those
        # words are material evidence only if support_materials is explicitly
        # populated or the type is an explicit material class; title fallback
        # below handles other authority cases.
        if item["source"] == "met":
            for hit in spans(fields["medium"], TECHNIQUE_PATTERNS):
                techniques.append({"value": hit["value"], "source_field": "medium",
                                   "evidence": fields["medium"], "status": "catalogued_medium_technique"})
        # De-duplicate by value plus source field, retaining all evidence.
        def dedupe(facets):
            merged = {}
            for facet in facets:
                key = (facet["value"], facet["source_field"])
                if key not in merged:
                    merged[key] = {**facet, "source_fields": [facet["source_field"]], "evidence_values": [facet["evidence"]]}
                else:
                    merged[key]["evidence_values"].append(facet["evidence"])
            return list(merged.values())
        return dedupe(materials), dedupe(techniques), unattributed

    def _direct_color_facets(self, item: dict, fields: dict) -> tuple[list[dict], list[dict]]:
        facets, unknown = [], []
        # Title is a direct object naming source.
        for hit in spans(fields["title"], COLOR_PATTERNS):
            scope = ("ware_name_color"
                     if re.search(rf"\b{re.escape(hit['match'])}\s+ware\b", fields["title"], re.I)
                     else "title_color_name")
            facets.append({"value": hit["value"], "term_type": COLOR_TERM_TYPES[hit["value"]],
                           "source_field": "title", "evidence": fields["title"],
                           "scope": scope, "status": "catalogued_direct", "match": hit["match"]})
        # Medium/technique colors describe a coating/body, but remain catalogue
        # evidence separate from a complete image palette.
        medium_field = "medium" if item["source"] == "met" else "technique"
        medium_text = fields["medium"] if item["source"] == "met" else fields["technique"]
        for hit in spans(medium_text, COLOR_PATTERNS):
            facets.append({"value": hit["value"], "term_type": COLOR_TERM_TYPES[hit["value"]],
                           "source_field": medium_field, "evidence": medium_text,
                           "scope": "coating_or_material_color", "status": "catalogued_technique_color", "match": hit["match"]})
        # Tags are official direct annotations when present.
        for tag in fields["tags"]:
            for hit in spans(tag, COLOR_PATTERNS):
                facets.append({"value": hit["value"], "term_type": COLOR_TERM_TYPES[hit["value"]],
                               "source_field": "tags", "evidence": tag,
                               "scope": "tag_color", "status": "catalogued_direct", "match": hit["match"]})
        for sentence, start, _ in sentences(fields["description"]):
            attribution = sentence_attribution(sentence, fields["title"])
            sentence_hits = spans(sentence, COLOR_PATTERNS)
            if not sentence_hits:
                continue
            for hit in sentence_hits:
                phrase_match = NONSURFACE_PHRASES.search(sentence)
                nonsurface = bool(phrase_match and (hit["value"].casefold() in phrase_match.group(0).casefold()))
                clause = _sentence_core(sentence)
                hit_in_core = hit["start"] < len(clause) + 2
                object_surface = re.search(r"\b(?:body|glaze|enamel|surface|patina|pigment|lacquer|slip|object)\b", sentence, re.I)
                subject_clause = first_sentence(sentence)
                non_object_subject = bool(re.search(r"\b(?:tea|bear|bird|fish|horse|lion|tiger|figure)\b", subject_clause, re.I)) and not object_surface
                if attribution["attributed"] and hit_in_core and not nonsurface and not non_object_subject:
                    facets.append({"value": hit["value"], "term_type": COLOR_TERM_TYPES[hit["value"]],
                                   "source_field": "description", "evidence": sentence,
                                   "scope": "visible_or_decorative_prose", "status": "catalogued_direct_sentence",
                                   "match": hit["match"], "sentence_offset": start + hit["start"]})
                else:
                    reason = "named_non_surface_phrase" if nonsurface else (
                        "non_object_color_subject" if non_object_subject else
                        (attribution["reason"] if hit_in_core else "subordinate_comparison_clause")
                    )
                    unknown.append({"kind": "color", "value": hit["value"], "source_field": "description",
                                    "evidence": sentence, "match": hit["match"],
                                    "reason": reason})
        return self._merge_facets(facets), unknown

    def _direct_motif_facets(self, item: dict, fields: dict) -> tuple[list[dict], list[dict]]:
        facets, unknown = [], []
        for hit in spans(fields["title"], MOTIF_PATTERNS):
            if hit["value"] == "bird" and re.search(r"\bbird(?:s)?\s+feeders?\b", fields["title"], re.I):
                unknown.append({"kind": "motif", "value": hit["value"], "source_field": "title",
                                "evidence": fields["title"], "match": hit["match"],
                                "reason": "object_use_not_motif"})
                continue
            facets.append({"value": hit["value"], "term_type": MOTIF_TERM_TYPES[hit["value"]],
                           "source_field": "title", "evidence": fields["title"],
                           "status": "catalogued_direct", "match": hit["match"]})
        for tag in fields["tags"]:
            for hit in spans(tag, MOTIF_PATTERNS):
                facets.append({"value": hit["value"], "term_type": MOTIF_TERM_TYPES[hit["value"]],
                               "source_field": "tags", "evidence": tag,
                               "status": "catalogued_direct", "match": hit["match"]})
        for sentence, start, _ in sentences(fields["description"]):
            attribution = sentence_attribution(sentence, fields["title"])
            clause = _sentence_core(sentence)
            for hit in spans(sentence, MOTIF_PATTERNS):
                phrase_match = NONSURFACE_PHRASES.search(sentence)
                nonsurface = bool(phrase_match and (hit["value"].casefold() in phrase_match.group(0).casefold()))
                hit_in_core = hit["start"] < len(clause) + 2
                if attribution["attributed"] and hit_in_core and not nonsurface:
                    facets.append({"value": hit["value"], "term_type": MOTIF_TERM_TYPES[hit["value"]],
                                   "source_field": "description", "evidence": sentence,
                                   "status": "catalogued_direct_sentence", "match": hit["match"],
                                   "sentence_offset": start + hit["start"]})
                else:
                    unknown.append({"kind": "motif", "value": hit["value"], "source_field": "description",
                                    "evidence": sentence, "match": hit["match"],
                                    "reason": "named_non_surface_phrase" if nonsurface else
                                              (attribution["reason"] if hit_in_core else "subordinate_comparison_clause")})
        return self._merge_facets(facets), unknown

    @staticmethod
    def _merge_facets(facets: list[dict]) -> list[dict]:
        merged = {}
        for facet in facets:
            key = (facet["value"], facet["source_field"])
            if key not in merged:
                merged[key] = {**facet, "source_fields": [facet["source_field"]],
                               "evidence_values": [facet["evidence"]]}
            else:
                if facet["evidence"] not in merged[key]["evidence_values"]:
                    merged[key]["evidence_values"].append(facet["evidence"])
                if facet.get("match"):
                    variants = {item["match"] for item in merged[key].get("occurrences", [])}
                    if facet["match"] not in variants:
                        merged[key]["occurrences"] = [*merged[key].get("occurrences", []),
                                                      {"match": facet["match"], "evidence": facet["evidence"],
                                                       "sentence_offset": facet.get("sentence_offset")}]
        return list(merged.values())

    def _era(self, fields: dict) -> list[dict]:
        # Each era facet is tied to its own authority field, so a general
        # collection label cannot absorb a reign, date, or culture value.
        era_fields = []
        if fields["era"]:
            era_fields.append(("era", fields["era"]))
        return self._merge_facets([
            {**hit, "source_field": source, "status": "catalogued_date_or_culture"}
            for source, source_text in era_fields
            for hit in spans(source_text, ERAS)
        ])

    def build_one(self, item: dict) -> dict:
        raw = self.raw_record(item)
        fields = self.catalogue_fields(item, raw)
        component_role, component_source = self._component(fields["title"])
        forms, unknown_forms = self._forms(item, fields, component_source)
        if component_role["status"] == "catalogued_explicit_component":
            forms = [{"value": component_role["value"], "source_field": "title", "evidence": fields["title"],
                      "status": "catalogued_explicit_component"}]
        materials, techniques, unknown_material = self._material_technique(item, fields)
        colors, unknown_colors = self._direct_color_facets(item, fields)
        motifs, unknown_motifs = self._direct_motif_facets(item, fields)
        era = self._era(fields)
        # Component context: preserve the parent object form as context rather
        # than turning a stand, cover, or container into the whole object.
        if component_role["status"] == "catalogued_explicit_component" and component_source:
            # The parent record's title head identifies the whole object. Text
            # after `with`/`from` can describe decoration or other figures.
            parent_main = re.sub(r"\s*\([^()]+\)\s*$", "", component_source).strip()
            parent_head = re.split(
                r"\s*(?::|\bwith\b|\bfrom\b)\s*", parent_main, maxsplit=1, flags=re.I
            )[0]
            parent_forms = self._merge_facets([
                {**hit, "source_field": "title", "status": "catalogued_parent_context"}
                for hit in spans(parent_head, OBJECT_FORMS)
            ])
            component_role["parent_form_facets"] = parent_forms
            component_role["parent_title"] = parent_head
        photographed = {
            "value": component_role["value"] if component_role["status"] == "catalogued_explicit_component" else "unknown",
            "status": "catalogued_title_component_unverified_image" if component_role["status"] == "catalogued_explicit_component" else "unknown",
            "source_field": "title", "evidence": fields["title"], "image_verified": False,
        }
        missing = [name for name, value in (("object_form", forms), ("material_facets", materials),
                                             ("colors", colors), ("motifs", motifs), ("era", era)) if not value]
        raw_class = {"value": fields["classification"] if item["source"] == "met" else fields["object_name"],
                     "source_field": "classification" if item["source"] == "met" else "type",
                     "evidence": fields["object_name"] if item["source"] == "cleveland" else fields["classification"],
                     "status": "catalogued_raw_class" if (fields["classification"] or fields["object_name"]) else "unknown"}
        return {
            "item_id": item["item_id"], "source": item["source"], "split": item["split"],
            "object_group": item["object_group"], "image_path": item["image_path"],
            "image_sha256": item["image_sha256"], "label_source": "official_catalogue_v2",
            "schema_version": VERSION,
            "official_catalog": {
                "title": fields["title"], "object_name": fields["object_name"],
                "classification": fields["classification"], "medium": fields["medium"],
                "technique": fields["technique"], "support_materials": fields["support_materials"],
                "era": fields["era"], "tags": fields["tags"], "description": fields["description"],
                "source_url": item.get("source_url", ""),
                "record_path": f"data/raw/{item['source']}/{item['source_id']}.json",
            },
            "derived_ontology": {
                "raw_class": raw_class,
                "object_form": forms,
                "material_facets": materials,
                "technique_facets": techniques,
                "component_role": component_role,
                "photographed_subject": photographed,
                "color_facets": colors,
                "motif_facets": motifs,
                "era_facets": era,
                # Compatibility views contain only attributed catalogue values.
                "object_type": unique_values(forms),
                "material": unique_values(materials),
                "broad_material": sorted({facet["broad"] for facet in materials}),
                "colors": unique_values([facet for facet in colors
                                         if facet["scope"] in {"title_color_name", "tag_color",
                                                               "visible_or_decorative_prose"}]),
                "coating_colors": unique_values([facet for facet in colors
                                                 if facet["scope"] == "coating_or_material_color"]),
                "motifs": unique_values(motifs),
                "era": unique_values(era),
                "unattributed_mentions": unknown_forms + unknown_material + unknown_colors + unknown_motifs,
                "unknown_fields": missing,
                "ai_supplement_eligible_fields": list(AI_SUPPLEMENT_FIELDS),
            },
            "ai_visible_supplement": None,
        }

    def build(self, rows: list[dict]) -> list[dict]:
        labels = [self.build_one(item) for item in rows]
        by_item = {item["item_id"]: item for item in rows}
        groups = defaultdict(list)
        for item in rows:
            groups[item["object_group"]].append(item)
        for label in labels:
            item = by_item[label["item_id"]]
            siblings = groups[item["object_group"]]
            label["group_context"] = {
                "group_id": item["object_group"], "member_count": len(siblings),
                "sibling_ids": [row["item_id"] for row in siblings if row["item_id"] != item["item_id"]],
                "policy": "group controls split isolation; each item retains its own raw fields and facets",
            }
        return labels


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def ontology_candidates(labels: list[dict], patterns=None) -> list[dict]:
    patterns = patterns or CANDIDATE_PATTERNS
    # Include raw official fields plus full-catalogue candidate state.
    output = []
    for category, category_patterns in patterns.items():
        for value, pattern in category_patterns:
            matched, fields, examples = 0, Counter(), []
            for label in labels:
                official = label["official_catalog"]
                tags_text = text(official["tags"])
                values = [("title", official["title"]), ("object_name", official["object_name"]),
                          ("classification", official["classification"]),
                          ("medium", official["medium"]), ("technique", official["technique"]),
                          ("support_materials", text(official["support_materials"])),
                          ("tags", tags_text),
                          ("description", official["description"])]
                hit_fields = [field for field, source in values if source and re.search(pattern, source, re.I)]
                if hit_fields:
                    matched += 1
                    fields.update(hit_fields)
                    if len(examples) < 8:
                        examples.append({"item_id": label["item_id"], "title": official["title"], "fields": hit_fields})
            output.append({
                "category": category, "candidate": value, "item_coverage": matched,
                "coverage_rate": round(matched / len(labels), 6) if labels else 0,
                "source_field_counts": dict(fields.most_common()), "examples": examples,
                "status": "candidate_not_registered_ontology",
                "warning": "Catalogue-text support; not proof that every mention is visible or refers to this object.",
            })
    return sorted(output, key=lambda row: (-row["item_coverage"], row["category"], row["candidate"]))


def official_text(label: dict) -> str:
    return text(label["official_catalog"])


def validate_supplement_payload(payload: dict, labels_by_id: dict[str, dict], root: Path = ROOT,
                                *, phase: str | None = None) -> dict:
    errors = []
    item_id = payload.get("item_id") or payload.get("review_id")
    label = labels_by_id.get(item_id)
    claims = payload.get("claims")
    no_supplement_reason = payload.get("no_supplement_reason")
    required_payload = ("item_id", "image_path", "image_sha256", "catalog_record_path",
                        "catalog_record_sha256", "reviewer", "created_at", "claims")
    allowed_payload = set(required_payload) | {"no_supplement_reason"}
    for field in sorted(set(payload) - allowed_payload):
        errors.append({"code": "unsupported_payload_field", "field": field})
    for field in required_payload:
        if field == "claims":
            if not isinstance(payload.get(field), list):
                errors.append({"code": "missing_payload_field", "field": field})
        elif not payload.get(field):
            errors.append({"code": "missing_payload_field", "field": field})
    if payload.get("created_at"):
        try:
            datetime.fromisoformat(str(payload["created_at"]).replace("Z", "+00:00"))
        except ValueError:
            errors.append({"code": "created_at_invalid"})
    if claims:
        if no_supplement_reason not in (None, ""):
            errors.append({"code": "no_supplement_reason_with_nonempty_claims"})
    elif not errors and (not isinstance(no_supplement_reason, str) or not no_supplement_reason.strip()):
        errors.append({"code": "no_supplement_reason_required"})
    if label is None:
        errors.append({"code": "unknown_item_id", "item_id": item_id})
    else:
        image_path = root / str(payload.get("image_path", ""))
        record_path = root / str(payload.get("catalog_record_path", ""))
        if payload.get("image_path") != label["image_path"]:
            errors.append({"code": "queued_image_path_mismatch"})
        if payload.get("image_sha256") != label["image_sha256"]:
            errors.append({"code": "queued_image_sha_mismatch"})
        if payload.get("catalog_record_path") != label["official_catalog"]["record_path"]:
            errors.append({"code": "queued_catalog_record_path_mismatch"})
        if not image_path.is_file():
            errors.append({"code": "image_missing", "path": payload.get("image_path")})
        elif payload.get("image_sha256") != digest(image_path):
            errors.append({"code": "image_hash_mismatch", "path": payload.get("image_path")})
        if not record_path.is_file():
            errors.append({"code": "catalog_record_missing", "path": payload.get("catalog_record_path")})
        elif payload.get("catalog_record_sha256") != digest(record_path):
            errors.append({"code": "catalog_record_hash_mismatch", "path": payload.get("catalog_record_path")})
    if not isinstance(claims, list):
        errors.append({"code": "claims_missing"})
        return {"accepted": False, "item_id": item_id, "errors": errors, "claims": []}
    no_supplement_reason = payload.get("no_supplement_reason")
    if not claims:
        if not isinstance(no_supplement_reason, str) or not no_supplement_reason.strip():
            errors.append({"code": "no_supplement_reason_required"})
        return {"accepted": not errors, "item_id": item_id, "errors": errors, "claims": [],
                "no_supplement_reason": no_supplement_reason}
    allowed_visibility = {"clear"}
    accepted = []
    for index, claim in enumerate(claims):
        allowed_claim = {"field", "value", "catalogue_evidence", "image_evidence",
                         "observation_status", "training_role", "visibility", "confidence"}
        for field in sorted(set(claim) - allowed_claim):
            errors.append({"code": "unsupported_claim_json_field", "claim_index": index, "field": field})
        missing = [field for field in ("field", "value", "catalogue_evidence", "image_evidence",
                                       "observation_status", "training_role", "visibility",
                                       "confidence") if field not in claim]
        if missing:
            errors.append({"code": "claim_missing_fields", "claim_index": index, "fields": missing})
            continue
        field = claim["field"]
        if field not in AI_SUPPLEMENT_FIELDS:
            errors.append({"code": "unsupported_claim_field", "claim_index": index, "field": field})
            continue
        catalogue_evidence = claim["catalogue_evidence"]
        if not isinstance(catalogue_evidence, dict):
            errors.append({"code": "catalogue_evidence_missing", "claim_index": index})
            continue
        for evidence_field in sorted(set(catalogue_evidence) - {"field", "quote", "relation"}):
            errors.append({"code": "unsupported_catalogue_evidence_field", "claim_index": index,
                           "field": evidence_field})
        if any(key not in catalogue_evidence for key in ("field", "quote", "relation")):
            errors.append({"code": "catalogue_evidence_incomplete", "claim_index": index})
            continue
        allowed_relations = {
            "supports_claim", "resolves_ambiguity", "provides_context",
            "does_not_state_property",
        }
        if catalogue_evidence["relation"] not in allowed_relations:
            errors.append({"code": "invalid_catalogue_relation", "claim_index": index})
            continue
        confidence = claim["confidence"]
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool) or not 0 <= float(confidence) <= 1:
            errors.append({"code": "confidence_below_review_threshold", "claim_index": index})
            continue
        if claim["observation_status"] not in AI_OBSERVATION_STATUSES:
            errors.append({"code": "invalid_observation_status", "claim_index": index})
            continue
        if claim["training_role"] != "candidate_append_only":
            errors.append({"code": "invalid_training_role", "claim_index": index})
            continue
        if claim["visibility"] not in allowed_visibility:
            errors.append({"code": "invalid_visibility", "claim_index": index})
            continue
        image_evidence = claim["image_evidence"]
        if not isinstance(image_evidence, dict):
            errors.append({"code": "image_evidence_invalid", "claim_index": index})
            continue
        for evidence_field in sorted(set(image_evidence) - {"region", "observation"}):
            errors.append({"code": "unsupported_image_evidence_field", "claim_index": index,
                           "field": evidence_field})
        if any(key not in image_evidence for key in ("region", "observation")):
            errors.append({"code": "image_evidence_incomplete", "claim_index": index})
            continue
        for field in ("region", "observation"):
            if not text(image_evidence.get(field)):
                errors.append({"code": "image_evidence_incomplete", "claim_index": index, "field": field})
                break
        else:
            pass
        if errors and errors[-1].get("claim_index") == index:
            continue
        if label:
            name = normalized(text(catalogue_evidence.get("field")))
            quote = evidence_text(text(catalogue_evidence.get("quote")))
            if not name or not quote:
                errors.append({"code": "catalogue_evidence_incomplete", "claim_index": index})
                continue
            record_path = root / label["official_catalog"]["record_path"]
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                record = {}
            if name not in record:
                errors.append({"code": "unsupported_catalogue_evidence_field", "claim_index": index})
                continue
            if quote not in evidence_text(text(record[name])):
                errors.append({"code": "catalogue_evidence_not_in_record", "claim_index": index})
                continue
        expected_status = ("catalogue_confirmed_visible"
                           if catalogue_evidence["relation"] in {"supports_claim", "resolves_ambiguity"}
                           else "image_only_obvious")
        if claim["observation_status"] != expected_status:
            errors.append({"code": "observation_status_conflicts_with_catalogue_evidence",
                           "claim_index": index})
            continue
        accepted.append({**claim, "claim_index": index})
    return {"accepted": bool(accepted) and len(accepted) == len(claims) and not errors,
            "item_id": item_id, "errors": errors, "claims": accepted}


def validate_supplement_file(path: Path, labels_by_id: dict[str, dict], root: Path = ROOT,
                             *, phase: str | None = None) -> dict:
    payloads = read_jsonl(path)
    results = [validate_supplement_payload(payload, labels_by_id, root, phase=phase) for payload in payloads]
    seen = Counter(payload.get("item_id") for payload in payloads)
    duplicates = sorted(item_id for item_id, count in seen.items() if item_id and count > 1)
    if duplicates:
        for result in results:
            if result.get("item_id") in duplicates:
                result["accepted"] = False
                result["errors"].append({"code": "duplicate_item_id"})
    return {"path": str(path), "records": len(results),
            "unique_items": len(seen), "duplicate_item_ids": duplicates,
            "accepted": all(result["accepted"] for result in results),
            "accepted_claims": sum(len(result["claims"]) for result in results if result["accepted"]),
            "results": results}


def build_ai_review_queue(labels: list[dict], limit: int = 240) -> tuple[list[dict], dict]:
    ranked = []
    for label in labels:
        derived = label["derived_ontology"]
        score = 0
        reasons = []
        if not derived["colors"]:
            score += 2
            reasons.append("color_unknown")
        if not derived["motifs"]:
            score += 2
            reasons.append("motif_unknown")
        if derived["unattributed_mentions"]:
            score += 3
            reasons.append("catalogue_mention_unattributed")
        if label["official_catalog"]["description"] and not derived["motifs"]:
            score += 1
        raw_class = derived["raw_class"]["value"].casefold()
        if raw_class in {"glass", "jade", "metalwork", "enamel", "lacquer", "wood", "stone", "ivory", "textile", "ceramic"}:
            score += 1
            reasons.append("broad_raw_class")
        if score:
            ranked.append((score, hashlib.sha256(label["item_id"].encode()).hexdigest(), label, reasons))
    ranked.sort(key=lambda value: (-value[0], value[1]))
    queue = []
    for score, tie_break, label, reasons in ranked[:limit]:
        official = label["official_catalog"]
        queue.append({
            "task": "record_visible_observations_after_full_catalogue_review",
            "priority_score": score, "priority_reasons": reasons,
            "item_id": label["item_id"], "split": label["split"],
            "image_path": label["image_path"], "image_sha256": label["image_sha256"],
            "catalog_record_path": official["record_path"],
            "catalog_record_sha256": digest(ROOT / official["record_path"]),
            "require_complete_catalog_read": True, "require_image_review": True,
            "official_catalog": official,
            "derived_unknown_fields": label["derived_ontology"]["unknown_fields"],
            "unattributed_mentions": label["derived_ontology"]["unattributed_mentions"],
            "allowed_supplement_fields": list(AI_SUPPLEMENT_FIELDS),
            "output_schema": "ai_supplement.schema.json",
            "catalogue_authority": "immutable; image observations never replace or rewrite official fields",
            "tools_allowed": True,
            "may_return_no_claims": True,
        })
    return queue, {"candidate_items": len(ranked), "queued_items": len(queue), "limit": limit,
                   "not_ai_run": True, "selection": "priority score then SHA-256(item_id)",
                   "tools_allowed": True, "empty_claims_allowed_with_reason": True}


def legacy_fix_counts(labels: list[dict], catalog: Path) -> dict:
    legacy_module = load_module(LEGACY_BUILDER, "legacy_cv_v4_builder_for_comparison")
    rows = read_jsonl(catalog)
    legacy = {row["item_id"]: legacy_module.build_one(row) for row in rows}
    counts = Counter()
    examples = defaultdict(list)
    for label in labels:
        old = legacy[label["item_id"]]["derived_ontology"]
        new = label["derived_ontology"]
        old_colors, new_colors = set(old["colors"]), set(new["colors"])
        removed_colors = sorted(old_colors - new_colors)
        if removed_colors:
            counts["legacy_color_values_removed"] += len(removed_colors)
            if len(examples["color_removed"]) < 20:
                examples["color_removed"].append({"item_id": label["item_id"], "removed": removed_colors})
        old_material, new_material = set(old["material"]), set(new["material"])
        removed_material = sorted(old_material - new_material)
        if removed_material:
            counts["legacy_material_values_removed"] += len(removed_material)
            if len(examples["material_removed"]) < 20:
                examples["material_removed"].append({"item_id": label["item_id"], "removed": removed_material})
        old_motifs, new_motifs = set(old["motifs"]), set(new["motifs"])
        removed_motifs = sorted(old_motifs - new_motifs)
        if removed_motifs:
            counts["legacy_motif_values_removed"] += len(removed_motifs)
            if len(examples["motif_removed"]) < 20:
                examples["motif_removed"].append({"item_id": label["item_id"], "removed": removed_motifs})
        if "bottle" in old["object_type"] and "bottle" not in new["object_type"] and str(new["component_role"]["value"]).endswith("stopper"):
            counts["legacy_component_bottle_suppressed"] += 1
    return {"counts": dict(counts), "examples": dict(examples),
            "comparison": "legacy builder output versus catalogue v2 compatibility values"}


def builder_defect_counts(labels: list[dict], legacy: dict[str, dict]) -> dict:
    """Count deterministic defects addressed by v2, with item-level evidence."""
    counts = Counter()
    evidence = defaultdict(list)
    for label in labels:
        old = legacy[label["item_id"]]["derived_ontology"]
        new = label["derived_ontology"]
        if len(old["colors"]) != len(set(old["colors"])):
            counts["legacy_duplicate_color_values"] += 1
        # Every v2 colour facet has an explicit source and a literal strict token.
        for facet in new["color_facets"]:
            source = facet["evidence"]
            value = facet["value"]
            if value not in {"gray", "grey", "turquoise", "amber"} and not re.search(rf"\b{re.escape(value)}\b", source, re.I):
                counts["v2_color_without_literal_value"] += 1
                evidence["v2_color_without_literal_value"].append({"item_id": label["item_id"], "value": value})
        component = new["component_role"]
        if component["status"] == "catalogued_explicit_component":
            counts["v2_explicit_component_items"] += 1
            if len(new["object_form"]) == 1 and new["object_form"][0] == component["value"]:
                counts["v2_component_form_separated"] += 1
        if old["material"] and not new["material"]:
            counts["old_material_now_unknown"] += 1
            if len(evidence["old_material_now_unknown"]) < 20:
                evidence["old_material_now_unknown"].append({"item_id": label["item_id"], "old": old["material"],
                                                               "technique": new["technique_facets"]})
        if label["source"] == "cleveland" and label["official_catalog"]["technique"]:
            technique_names = {facet["value"] for facet in new["technique_facets"]}
            material_names = {facet["value"] for facet in new["material_facets"]}
            # Material terms may legitimately appear in a technique string; the
            # invariant is that no technique-only token is material evidence.
            if technique_names and not (technique_names & material_names):
                counts["cma_technique_kept_out_of_material"] += 1
        if new["unattributed_mentions"]:
            counts["items_with_unattributed_catalogue_mentions"] += 1
        if any(item["reason"] in {"subject_not_proven", "subordinate_comparison_clause",
                                   "comparison_or_other_object_context", "generic_or_comparand_subject"}
               for item in new["unattributed_mentions"]):
            counts["items_with_unknown_attribution"] += 1
    return {"counts": dict(counts), "evidence_examples": dict(evidence),
            "interpretation": "These are deterministic transformations, not an estimate of total catalogue truth."}


def write_issue_ledger(path: Path, labels: list[dict], legacy: dict[str, dict]) -> None:
    rows = []
    for label in labels:
        old = legacy[label["item_id"]]["derived_ontology"]
        new = label["derived_ontology"]
        for mention in new["unattributed_mentions"]:
            rows.append({"item_id": label["item_id"], "title": label["official_catalog"]["title"],
                         "kind": mention["kind"], "value": mention["value"],
                         "source_field": mention["source_field"], "evidence": mention["evidence"],
                         "reason": mention["reason"], "status": "unknown",
                         "legacy_derived": old["colors"] if mention["kind"] == "color" else old["motifs"]})
        if new["raw_class"]["source_field"] == "type" and new["raw_class"]["value"].casefold() in {
                "ceramic", "glass", "jade", "metalwork", "lacquer", "wood", "stone", "ivory", "enamel", "textile"}:
            rows.append({"item_id": label["item_id"], "title": label["official_catalog"]["title"],
                         "kind": "raw_class", "value": new["raw_class"]["value"],
                         "source_field": "type", "evidence": new["raw_class"]["evidence"],
                         "reason": "raw_type_is_class_not_form", "status": "separated",
                         "legacy_derived": old["object_type"]})
        if label["source"] == "cleveland" and label["official_catalog"]["technique"]:
            rows.append({"item_id": label["item_id"], "title": label["official_catalog"]["title"],
                         "kind": "technique", "value": label["official_catalog"]["technique"],
                         "source_field": "technique", "evidence": label["official_catalog"]["technique"],
                         "reason": "cma_technique_not_promoted_to_material", "status": "separated",
                         "legacy_derived": old["material"]})
        component = new["component_role"]
        if component["status"] != "catalogued_or_unspecified":
            rows.append({"item_id": label["item_id"], "title": label["official_catalog"]["title"],
                         "kind": "component_role", "value": component["value"], "source_field": "title",
                         "evidence": component["evidence"], "reason": component["status"], "status": "separated",
                         "legacy_derived": old["object_type"]})
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def group_report(labels: list[dict]) -> dict:
    groups = defaultdict(list)
    for label in labels:
        groups[label["object_group"]].append(label)
    cross = [group for group, members in groups.items() if len({member["split"] for member in members}) > 1]
    mixed = []
    for group, members in groups.items():
        classes = sorted({member["derived_ontology"]["raw_class"]["value"] for member in members if member["derived_ontology"]["raw_class"]["value"]})
        if len(members) > 1 and len(classes) > 1:
            mixed.append({"group_id": group, "split": members[0]["split"], "raw_classes": classes,
                          "members": [{"item_id": member["item_id"], "title": member["official_catalog"]["title"],
                                       "raw_class": member["derived_ontology"]["raw_class"]["value"],
                                       "object_form": member["derived_ontology"]["object_type"],
                                       "component_role": member["derived_ontology"]["component_role"]["value"]}
                                      for member in members]})
    return {"group_count": len(groups), "multi_item_groups": sum(len(members) > 1 for members in groups.values()),
            "multi_item_group_rows": sum(len(members) for members in groups.values() if len(members) > 1),
            "cross_split_group_count": len(cross), "cross_split_groups": cross,
            "groups_with_distinct_raw_classes": len(mixed), "mixed_class_examples": mixed[:25]}


def write_schema(path: Path, labels: list[dict]) -> None:
    example = next((label for label in labels if label["item_id"] == "cma_150155"),
                   next(label for label in labels if label["image_path"]))
    official = example["official_catalog"]
    payload = {
        "item_id": example["item_id"], "image_path": example["image_path"],
        "image_sha256": example["image_sha256"], "catalog_record_path": official["record_path"],
        "catalog_record_sha256": digest(ROOT / official["record_path"]),
        "reviewer": "AI model and run identifier", "created_at": "2026-11-08T00:00:00Z",
        "no_supplement_reason": None,
        "claims": [{"field": "visible_palette",
                    "value": ["red", "blue", "pale pink", "green"],
                    "catalogue_evidence": {
                        "field": "technique",
                        "quote": "porcelain with overglaze enamel decoration",
                        "relation": "provides_context",
                    },
                    "image_evidence": {
                        "region": "visible bowl exterior",
                        "observation": "Red, blue, pale pink, and green decoration is visible on the exterior.",
                    },
                    "observation_status": "image_only_obvious",
                    "training_role": "candidate_append_only",
                    "visibility": "clear", "confidence": 0.9}],
    }
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "CV v4 catalogue-first AI visible supplement",
        "type": "object", "additionalProperties": False,
        "required": ["item_id", "image_path", "image_sha256", "catalog_record_path",
                     "catalog_record_sha256", "reviewer", "created_at", "claims",
                     "no_supplement_reason"],
        "properties": {
            "item_id": {"type": "string"}, "image_path": {"type": "string"},
            "image_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "catalog_record_path": {"type": "string"},
            "catalog_record_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "reviewer": {"type": "string", "minLength": 1},
            "created_at": {"type": "string", "format": "date-time"},
            "no_supplement_reason": {
                "type": ["string", "null"], "description": "Required when claims is empty."
            },
            "claims": {"type": "array", "items": {
                "type": "object", "additionalProperties": False,
                "required": ["field", "value", "catalogue_evidence", "image_evidence",
                             "observation_status", "training_role", "visibility", "confidence"],
                "properties": {
                    "field": {"enum": list(AI_SUPPLEMENT_FIELDS)},
                    "value": {"type": ["string", "number", "boolean", "array"]},
                    "catalogue_evidence": {
                        "type": "object", "additionalProperties": False,
                        "required": ["field", "quote", "relation"],
                        "properties": {
                            "field": {"type": "string", "minLength": 1},
                            "quote": {"type": "string", "minLength": 1},
                            "relation": {"enum": ["supports_claim", "resolves_ambiguity",
                                                  "provides_context", "does_not_state_property"]},
                        },
                    },
                    "image_evidence": {
                        "type": "object", "additionalProperties": False,
                        "required": ["region", "observation"],
                        "properties": {
                            "region": {"type": "string", "minLength": 1},
                            "observation": {"type": "string", "minLength": 1},
                        },
                    },
                    "observation_status": {"enum": list(AI_OBSERVATION_STATUSES)},
                    "training_role": {"const": "candidate_append_only"},
                    "visibility": {"enum": ["clear"]},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                },
            }},
        },
        "example": payload,
    }
    path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_prompt(path: Path) -> None:
    path.write_text(
        "# Catalogue-first AI visible supplement\n\n"
        "Read the complete JSON museum record, inspect the original image, and use any tools that help you "
        "verify pixels or crop regions. Official catalogue fields are immutable primary supervision and remain "
        "authoritative. Read and use the complete record to establish exact identity, distinguish the photographed "
        "component, interpret ambiguous visual forms, and avoid guessing. The registry is open access; follow its "
        "documented API or image conventions when useful, inspect crops, and cite the exact source you checked. "
        "The record may omit obvious visible colors, shapes, parts, and surface features; those omissions are not "
        "prohibitions against recording what the pixels plainly show.\n\n"
        "Your task is to record useful visible observations for a later, reviewed supplemental training candidate. "
        "Restating a visible "
        "property that the catalogue also states is valid: keep the catalogue fact in `catalogue_evidence` and "
        "the independently observed pixels in `image_evidence`. The observation may resolve catalogue ambiguity, "
        "provide visual detail, or supply context not stated in the record. Never rewrite, replace, or treat a "
        "catalogue metadata value as false. If the complete record and image leave no useful visible observation, "
        "return `claims: []` and a concrete `no_supplement_reason`.\n\n"
        "Return one JSON object matching `ai_supplement.schema.json`. Every claim must provide `field`, `value`, "
        "`catalogue_evidence`, `image_evidence`, `observation_status`, `training_role`, `visibility`, and "
        "`confidence`. `catalogue_evidence` must name a raw-record field and quote it exactly. Use "
        "`supports_claim` or `resolves_ambiguity` only when that quote states the same property; use "
        "`does_not_state_property` when the record establishes the object identity but omits the visible property; "
        "use `provides_context` for other relevant context. Set `training_role` to `candidate_append_only`; a "
        "supplement is never finished gold until reviewed. `image_evidence` must identify the image region and describe the "
        "visible form, part, color, motif, surface, count, orientation, photographed subject, or photographed "
        "component. Only claim `clear` observations another reviewer can verify; omit partial or ambiguous "
        "observations. Unknown remains unknown, and no absence is a negative label.\n\n"
        "Allowed fields: `visible_form_detail`, `visible_part`, `visible_palette`, `visible_motifs`, "
        "`visible_material`, `visible_surface`, `visible_multiplicity`, `visible_orientation`, `photographed_subject`, "
        "`photographed_component`.\n\n"
        "The image path, image SHA-256, catalog record path, and catalog SHA-256 in the review queue are mandatory. "
        "Return one object per queue item. AI output is an append-only review observation and never modifies the "
        "official catalogue label or frozen annotations.\n",
        encoding="utf-8",
    )


def write_supplement_example(output: Path) -> dict:
    """Write a hash-valid empty-claim example; tests use a separate override rejection."""
    label = {
        "item_id": "cma_101719", "image_path": "data/processed/images/cma_101719.jpg",
        "image_sha256": "265f60e22057f06598e6e65290e329eab396ebc6aaefb9b73327c0c3e32925db",
        "catalog_record_path": "data/raw/cleveland/101719.json",
        "catalog_record_sha256": digest(ROOT / "data/raw/cleveland/101719.json"),
        "reviewer": "fixture_no_supplement", "created_at": "2026-11-08T00:00:00Z",
        "claims": [],
        "no_supplement_reason": "The complete record and original image already provide sufficient supervision.",
    }
    (output / "ai_supplement_no_claims_example.jsonl").write_text(
        json.dumps(label, ensure_ascii=False) + "\n", encoding="utf-8")
    return label


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--ai-queue-limit", type=int, default=240)
    parser.add_argument("--validate-supplements", type=Path)
    parser.add_argument("--supplement-phase", choices=("model", "review"), default="review")
    args = parser.parse_args(argv)
    catalog = args.catalog.resolve()
    output = args.output.resolve()
    rows = read_jsonl(catalog)
    builder = CatalogueBuilderV2(ROOT)
    labels = builder.build(rows)
    by_id = {label["item_id"]: label for label in labels}
    if args.validate_supplements:
        validation = validate_supplement_file(args.validate_supplements.resolve(), by_id, ROOT,
                                              phase=args.supplement_phase)
        print(json.dumps(validation, ensure_ascii=False, indent=2))
        return 0 if validation["accepted"] else 1
    output.mkdir(parents=True, exist_ok=True)
    labels_path = output / "labels_catalogue_v2.jsonl"
    labels_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels), encoding="utf-8")
    candidates = ontology_candidates(labels)
    with (output / "ontology_candidates.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("category", "candidate", "item_coverage", "coverage_rate",
                                                    "source_field_counts", "status"))
        writer.writeheader()
        for row in candidates:
            csv_row = {key: row[key] for key in writer.fieldnames}
            csv_row["source_field_counts"] = json.dumps(row["source_field_counts"], ensure_ascii=False)
            writer.writerow(csv_row)
    (output / "ontology_candidates.json").write_text(json.dumps(candidates, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    queue, queue_summary = build_ai_review_queue(labels, max(0, args.ai_queue_limit))
    (output / "ai_review_queue.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in queue), encoding="utf-8")
    write_schema(output / "ai_supplement.schema.json", labels)
    write_prompt(output / "ai_supplement_prompt.md")
    write_supplement_example(output)
    fixes = legacy_fix_counts(labels, catalog)
    legacy_module = load_module(LEGACY_BUILDER, "legacy_cv_v4_builder_for_counts")
    legacy = {row["item_id"]: legacy_module.build_one(row) for row in rows}
    deterministic = builder_defect_counts(labels, legacy)
    write_issue_ledger(output / "catalogue_issue_ledger.jsonl", labels, legacy)
    groups = group_report(labels)
    frequencies = {slot: Counter(value for label in labels for value in label["derived_ontology"][slot]).most_common()
                   for slot in ("object_type", "material", "colors", "motifs", "era")}
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(), "schema_version": VERSION,
        "catalog": str(catalog), "catalog_sha256": digest(catalog),
        "labels": str(labels_path), "labels_sha256": digest(labels_path),
        "items": len(labels), "splits": dict(Counter(label["split"] for label in labels)),
        "coverage": {slot: sum(bool(label["derived_ontology"][slot]) for label in labels)
                     for slot in ("object_form", "material_facets", "technique_facets", "colors", "motifs", "era")},
        "unknown_counts": {slot: sum(not label["derived_ontology"][slot] for label in labels)
                           for slot in ("object_form", "material_facets", "colors", "motifs", "era")},
        "unattributed_mention_count": sum(len(label["derived_ontology"]["unattributed_mentions"]) for label in labels),
        "component_counts": dict(Counter(label["derived_ontology"]["component_role"]["value"] for label in labels).most_common()),
        "photographed_subject_counts": dict(Counter(label["derived_ontology"]["photographed_subject"]["value"] for label in labels).most_common()),
        "frequencies": frequencies, "defect_fixes_vs_legacy": fixes,
        "deterministic_defect_fixes": deterministic,
        "groups": groups, "ontology_candidate_count": len(candidates),
        "ontology_candidates_file": str(output / "ontology_candidates.csv"),
        "ai_review": queue_summary, "ai_review_queue_file": str(output / "ai_review_queue.jsonl"),
        "authority": "Official catalogue fields first; AI supplements are review-only and cannot overwrite official facts.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "group_report.json").write_text(json.dumps(groups, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "defect_fixes.json").write_text(json.dumps(fixes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        f"# CV v4 catalogue builder v2\n\nGenerated by `scripts/build_cv_v4_catalogue_v2.py`. "
        f"Schema `{VERSION}`. Official catalogue values are immutable primary supervision. "
        "AI observations are append-only review material.\n\n"
        "- `labels_catalogue_v2.jsonl`: deterministic catalogue facets and compatibility views.\n"
        "- `ontology_candidates.json/csv`: unregistered concept candidates and coverage.\n"
        "- `ai_review_queue.jsonl`: deterministic, not-yet-run review queue.\n"
        "- `ai_supplement.schema.json` and `ai_supplement_prompt.md`: clear-only visible-observation contract.\n"
        "- `ai_supplement_no_claims_example.jsonl`: validated empty-claim example.\n"
        "- `defect_fixes.json`, `group_report.json`, `summary.json`: audit and group evidence.\n",
        encoding="utf-8")
    print(json.dumps({"output": str(output), "items": len(labels), "coverage": summary["coverage"],
                      "unknown_counts": summary["unknown_counts"],
                      "unattributed_mention_count": summary["unattributed_mention_count"],
                      "defect_fixes": fixes["counts"], "ai_review": queue_summary,
                      "groups": {key: groups[key] for key in ("multi_item_groups", "cross_split_group_count",
                                                              "groups_with_distinct_raw_classes")}},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
