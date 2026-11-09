"""Dynamic museum-reference retrieval with strict gallery isolation.

This module accompanies ``scripts/run_dynamic_reference_v4.py``.  Museum
metadata is allowed to plan/select a reference photograph; it is never passed
to gallery scoring.  Gallery metadata is only used by the evaluation layer to
create an independently diagnosed target pool.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import time
from typing import Iterable
from urllib.parse import urlencode

from PIL import Image, ImageOps
import requests

from curio.retrieve.references import (
    ERA_RULES,
    GalleryExclusions,
    image_fingerprints,
    normalize_record,
)


SOURCE_BASES = {
    "cleveland": "https://openaccess-api.clevelandart.org/api/artworks/",
    "met": "https://collectionapi.metmuseum.org/public/collection/v1.1/search",
}
MET_OBJECT = "https://collectionapi.metmuseum.org/public/collection/v1/objects/{}"


@dataclass(frozen=True)
class ConceptualRule:
    key: str
    aliases: tuple[str, ...]
    query_phrases: tuple[str, ...]
    evidence_pattern: str
    family: str


CONCEPTS = (
    ConceptualRule(
        "blue_white", ("青花", "blue and white", "blue-and-white", "underglaze blue",
                       "blue white", "青花瓷", "钴蓝彩"),
        ("blue and white porcelain", "underglaze blue porcelain",
         "Chinese blue and white ceramic"),
        r"blue[ -]and[ -]white|underglaze blue|cobalt.{0,40}blue|青花", "decoration"),
    ConceptualRule(
        "porcelain", ("瓷", "陶瓷", "陶器", "瓷瓶", "瓷罐", "ceramic", "porcelain",
                      "stoneware"),
        ("Chinese porcelain", "Chinese ceramic"),
        r"porcelain|ceramic|stoneware|earthenware|瓷|陶", "material"),
    ConceptualRule(
        "famille_rose", ("粉彩", "famille rose", "fencai"),
        ("famille rose porcelain", "Chinese fencai porcelain"),
        r"famille rose|fencai|粉彩", "decoration"),
    ConceptualRule(
        "polychrome", ("粉彩", "famille rose", "fencai", "五彩", "wucai", "斗彩",
                        "doucai", "珐琅彩", "enamel"),
        ("famille rose porcelain", "Chinese famille rose enamel porcelain",
         "wucai porcelain"),
        r"famille rose|fencai|polychrome|wucai|doucai|enamel.{0,35}(porcelain|ware)|五彩|斗彩|珐琅",
        "decoration"),
    ConceptualRule(
        "celadon", ("青瓷", "龙泉", "龍泉", "celadon", "longquan"),
        ("Chinese celadon porcelain", "Longquan celadon vase"),
        r"celadon|longquan|青瓷|龙泉|龍泉", "decoration"),
    ConceptualRule(
        "meiping", ("梅瓶", "meiping"), ("Chinese meiping vase", "plum vase China"),
        r"meiping|plum vase|梅瓶", "form"),
    ConceptualRule(
        "leather_flask", ("皮囊式壶", "皮囊壶", "皮囊", "马镫壶", "马镫", "鸡冠壶",
                           "leather flask", "leather pouch", "leather bag", "saddle flask"),
        ("Liao dynasty ewer leather", "Liao ceramic flask", "Liao dynasty pilgrim flask",
         "Liao dynasty saddle flask"),
        r"leather.{0,30}(flask|pouch|bag|vessel)|saddle.{0,20}flask|皮囊|马镫|鸡冠壶", "form"),
    ConceptualRule(
        "gourd", ("葫芦", "葫蘆", "gourd", "double gourd"),
        ("Chinese double gourd vase", "gourd-shaped Chinese vase"),
        r"gourd|葫芦|葫蘆", "form"),
    ConceptualRule(
        "jar", ("罐", "罐子", "jar"), ("Chinese porcelain jar",),
        r"\bjar\b|\bkuan\b|罐", "form"),
    ConceptualRule(
        "bowl", ("碗", "盌", "bowl"), ("Chinese porcelain bowl",),
        r"\bbowl\b|\bwan\b|碗|盌", "form"),
    ConceptualRule(
        "dish", ("盘", "盤", "碟", "dish", "plate"), ("Chinese porcelain dish",),
        r"\bdish\b|\bplate\b|\bsaucer\b|盘|盤|碟", "form"),
    ConceptualRule(
        "vase", ("瓶", "vase", "bottle"), ("Chinese porcelain vase",),
        r"vase|bottle|瓶", "form"),
    ConceptualRule(
        "dragon", ("龙纹", "龍紋", "dragon"), ("Chinese dragon porcelain", "dragon ceramic China"),
        r"\bdragons?\b|\blong\b|龙|龍", "motif"),
    ConceptualRule(
        "floral", ("花卉", "莲花", "蓮花", "牡丹", "lotus", "peony", "floral", "flower"),
        ("Chinese floral porcelain", "lotus design Chinese ceramic"),
        r"floral|flower|lotus|peony|花卉|莲花|蓮花|牡丹", "motif"),
)

CONCEPT_BY_KEY = {row.key: row for row in CONCEPTS}
DYNAMIC_ERAS = {
    **ERA_RULES,
    "Liao": ("辽代", "遼代", "辽朝", "遼朝", "liao dynasty"),
    "Han": ("汉代", "漢代", "han dynasty"),
    "Tang": ("唐代", "唐朝", "tang dynasty"),
    "Jin": ("金代", "金朝", "jin dynasty"),
    "Southern Song": ("南宋", "southern song"),
    "Northern Song": ("北宋", "northern song"),
}

ERA_YEAR_RANGES = {
    "Wanli": (1572, 1620),
    "Kangxi": (1662, 1722),
    "Yongzheng": (1723, 1735),
    "Qianlong": (1736, 1795),
    "Xuande": (1426, 1435),
    "Yongle": (1403, 1424),
    "Jiajing": (1522, 1567),
    "Zhengde": (1506, 1521),
    "Chongzhen": (1628, 1644),
    "Ming": (1368, 1644),
    "Qing": (1644, 1911),
    "Song": (960, 1279),
    "Yuan": (1271, 1368),
    "Liao": (907, 1125),
    "Han": (-206, 220),
    "Tang": (618, 907),
    "Jin": (1115, 1234),
    "Southern Song": (1127, 1279),
    "Northern Song": (960, 1127),
}

STRUCTURED_ERA_FIELDS = (
    "culture", "date", "objectDate", "creation_date", "period", "reign",
)
STRUCTURED_DATE_FIELDS = ("date", "objectDate", "creation_date")
STRUCTURED_ERA_DATE_FIELDS = (
    "date", "objectDate", "creation_date", "period", "reign", "culture",
)
OBJECT_IDENTITY_FIELDS = (
    "title", "objectName", "object_name", "type", "object_type", "classification",
)
CERAMIC_CONCEPTS = {"blue_white", "polychrome", "famille_rose", "celadon"}
REIGN_LABELS = {
    "Wanli", "Kangxi", "Yongzheng", "Qianlong", "Xuande", "Yongle",
    "Jiajing", "Zhengde", "Chongzhen",
}
MATERIAL_EVIDENCE = re.compile(
    r"\b(?:porcelain|porcelaneous|ceramic|stoneware|earthenware|pottery)\b|瓷|陶", re.I
)


def _match_alias(text: str, alias: str) -> bool:
    if re.search(r"[\u3400-\u9fff]", alias):
        return alias in text
    return bool(re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", text, re.I))


def _match_planner_alias(text: str, alias: str) -> bool:
    """Match query concepts without splitting the Longquan place/ware name."""
    if alias in {"龙", "龍"}:
        return bool(re.search(alias + r"(?!泉)", text))
    return _match_alias(text, alias)


def era_information(query: str) -> list[dict]:
    text = query.casefold()
    rows = []
    for label, aliases in DYNAMIC_ERAS.items():
        if any(_match_alias(text, alias) for alias in (*aliases, label)):
            rows.append({"label": label, "aliases": list(aliases),
                         "query_term": f"{label}"})
    # Broad dynasty terms should follow the narrower reign label, while both
    # remain explicit constraints rather than silently expanding the intent.
    by_label = {r["label"]: r for r in rows}
    if "Wanli" in by_label and "Ming" in by_label:
        by_label["Ming"]["implied_by"] = "Wanli"
    return rows


def plan_dynamic_query(query: str, *, verified_only: bool = False) -> dict:
    """Translate a user description into editable museum search phrases.

    Planning reads only the input query.  No target or prior-answer file is
    consulted.  ``verified_only`` returns the strict reference constraints;
    the default metadata-query plan is deliberately more permissive so an
    underspecified search can still yield candidates for the visual fallback.
    """
    text = query.strip()
    if not text:
        raise ValueError("Query must be nonempty")
    concepts = [row.key for row in CONCEPTS
                if any(_match_planner_alias(text, a) for a in row.aliases)]
    # A bare 瓷/ceramic token means porcelain but should not silently select
    # celadon or every decorative technique in the vocabulary.
    if not any(_match_alias(text, a) for a in ("青瓷", "龙泉", "龍泉", "celadon", "longquan")):
        concepts = [key for key in concepts if key != "celadon" or any(
            _match_alias(text, alias) for alias in CONCEPT_BY_KEY[key].aliases
            if alias not in {"青瓷", "celadon"})]
    # Specific glaze/technique overrides the generic porcelain/material word.
    if "blue_white" in concepts:
        concepts = [key for key in concepts if key not in {"polychrome", "celadon"}
                    or any(_match_alias(text, a) for a in CONCEPT_BY_KEY[key].aliases
                           if a not in {"瓷", "porcelain", "ceramic"})]
    # Preserve an explicitly named ceramic token (especially Chinese 瓷) as a
    # material constraint while removing only the generic form/material term.
    if any(key in concepts for key in ("meiping", "gourd", "leather_flask", "jar", "bowl", "dish")) and "vase" in concepts:
        concepts.remove("vase")
    if "porcelain" in concepts and not re.search(
            r"\b(?:porcelain|ceramic|stoneware)\b|瓷|陶", text, re.I):
        concepts.remove("porcelain")
    # Deduplicate by family while keeping explicit decorative and form concepts.
    concepts = list(dict.fromkeys(concepts))
    eras = era_information(text)
    explicit_ceramic = ("porcelain" in concepts
                        or any(key in CERAMIC_CONCEPTS for key in concepts))
    required_material = ["ceramic"] if explicit_ceramic else []
    form_keys = [key for key in concepts if CONCEPT_BY_KEY[key].family == "form"]
    deco_keys = [key for key in concepts if CONCEPT_BY_KEY[key].family != "form"]
    if not concepts and not eras:
        # English queries can be queried literally.  Chinese without a known
        # visual concept remains explicit and needs a manual query override.
        if re.search(r"[\u3400-\u9fff]", text):
            terms = []
        else:
            terms = [text]
    else:
        era_prefix = [r["label"] for r in eras if not r.get("implied_by")]
        base = "Chinese"
        if era_prefix:
            base = " ".join(era_prefix) + " " + base
        if deco_keys:
            # Use only the first material/decoration term: concatenating every
            # synonym produces unnecessarily strict API searches.
            phrase = CONCEPT_BY_KEY[deco_keys[0]].query_phrases[0]
            phrase = phrase.replace("Chinese ", "").replace("porcelain ", "porcelain ")
            base += " " + phrase
        elif not form_keys:
            base += " art"
        terms = []
        for key in form_keys:
            terms.append(base + " " + CONCEPT_BY_KEY[key].query_phrases[0].replace("Chinese ", ""))
        if deco_keys or not form_keys:
            terms.append(base)
        # These short official-vocabulary phrases are the fallback if the
        # structured query is too restrictive.
        for rule in CONCEPTS:
            if rule.key in concepts:
                terms.extend(rule.query_phrases)
    terms = list(dict.fromkeys(term.strip() for term in terms if term.strip()))

    # Compact intent variants recover museum identity vocabulary without
    # weakening validation.  They are era-bounded at the API and every result
    # still has to pass structured culture/date and concept checks.  In
    # particular, Cleveland catalogues Liao flasks as "Leather Bag-Shaped
    # Flasks", while an unnaturally long compound phrase retrieves neither.
    era_words = [row["label"] for row in eras if not row.get("implied_by")]
    compact = []
    for key in concepts:
        rule = CONCEPT_BY_KEY[key]
        # A short form phrase is more useful to the museum free-text index than
        # a compound "Chinese ... vase" phrase. All validation remains strict.
        if rule.family == "form":
            compact.extend(rule.query_phrases[1:])
        for alias in rule.aliases:
            if re.search(r"[\u3400-\u9fff]", alias):
                continue
            for prefix in era_words or [""]:
                compact.append(f"{prefix} {alias}".strip())
        for phrase in rule.query_phrases:
            for prefix in era_words or [""]:
                compact.append(f"{prefix} {phrase}".strip())
    if form_keys and deco_keys:
        form_alias = CONCEPT_BY_KEY[form_keys[0]].aliases[-1]
        deco_alias = CONCEPT_BY_KEY[deco_keys[0]].aliases[0]
        for prefix in era_words or [""]:
            compact.extend((f"{prefix} {deco_alias} {form_alias}".strip(),
                            f"{prefix} {form_alias} {deco_alias}".strip()))
    discovery_terms = list(dict.fromkeys(term for term in compact if term))
    if form_keys:
        # Always retain the bare museum form identity so structured API date
        # filters can discover objects whose free text omits the reign.
        discovery_terms.extend(CONCEPT_BY_KEY[form_keys[0]].query_phrases)
        discovery_terms = list(dict.fromkeys(discovery_terms))
    # Use explicit museum identity for these sparse forms.  Generic era-only
    # searches flood the bounded budget with unrelated objects.
    terms = list(dict.fromkeys(discovery_terms + terms))
    if verified_only:
        # Keep only museum phrases whose required concept and era are explicit
        # in the planned phrase itself.
        terms = [term for term in terms if any(rule.query_phrases[0].casefold() in term.casefold()
                                                for rule in CONCEPTS if rule.key in concepts)]
    return {
        "planner": "description_concepts_v4", "planner_version": 4,
        "input_query": text, "concept_keys": concepts,
        "era_constraints": eras,
        "required_concepts": concepts,
        "required_material": required_material,
        "query_terms": terms,
        "discovery_terms": discovery_terms,
        "search_date_range": museum_date_range(eras),
        "sources": ["cleveland", "met"],
        "fallback": {
            "trigger": "never for required era/reference constraints",
            "policy": "search broader museum vocabulary within the same era; do not select unverified references",
        },
        "metadata_role": "reference selection and diagnostic truth only",
        "target_ranking_inputs": ["reference_image_pixels", "gallery_image_pixels"],
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
                         encoding="utf-8")
    temporary.replace(path)


def _jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                 for row in rows), encoding="utf-8")
    temporary.replace(path)


def _phash_distance(first: str | None, second: str | None) -> int | None:
    if not first or not second:
        return None
    return (int(first, 16) ^ int(second, 16)).bit_count()


def _ahash_distance(first: str | None, second: str | None) -> int | None:
    if not first or not second or len(first) != len(second):
        return None
    return sum(a != b for a, b in zip(first, second))


def metadata_concept_keys(record: dict, source: str) -> list[str]:
    """Concept evidence; prose supports concepts but never object era."""
    text = " ".join(str(record.get(key, "")) for key in (
        "title", "medium", "description", "objectName", "classification",
        "type", "technique", "culture", "period", "reign",
    )).casefold()
    return [rule.key for rule in CONCEPTS if re.search(rule.evidence_pattern, text, re.I)]


def metadata_era_text(record: dict) -> str:
    """Return only source-controlled attribution fields, never curatorial prose."""
    value = " ".join(str(record.get(key, "")) for key in STRUCTURED_ERA_FIELDS).casefold()
    # Some Met records attach a non-contemporaneous mark to an otherwise
    # different attribution.  A 'Kangxi mark' is not evidence that the object
    # was made during Kangxi, whereas 'Kangxi period' or a structured date is.
    for label, aliases in DYNAMIC_ERAS.items():
        for alias in aliases:
            if re.search(r"[\u3400-\u9fff]", alias):
                continue
            value = re.sub(
                r"(?<![a-z])" + re.escape(alias) + r"\s+mark(?![a-z])", "", value, flags=re.I
            )
    return value


def museum_date_range(eras: list[dict]) -> dict | None:
    """Return the narrowest structured range implied by the requested era."""
    candidates = [(ERA_YEAR_RANGES[row["label"]], row["label"]) for row in eras
                  if row.get("label") in ERA_YEAR_RANGES]
    if not candidates:
        return None
    (start, end), label = min(candidates, key=lambda value: value[0][1] - value[0][0])
    return {"label": label, "start": start, "end": end}


def _culture_class(record: dict) -> str | None:
    culture = str(record.get("culture", "")).casefold()
    markers = (
        ("china", (r"\bchina\b", r"\bchinese\b", "中国", "中國")),
        ("japan", (r"\bjapan\b", r"\bjapanese\b", "日本")),
        ("korea", (r"\bkorea\b", r"\bkorean\b", "韩国", "韓國", "朝鲜", "朝鮮")),
        ("vietnam", (r"\bvietnam\b", r"\bvietnamese\b", "越南")),
        ("europe", (r"\bfrance\b", r"\bfrench\b", r"\beurope\b", r"\beuropean\b",
                    r"\bengland\b", r"\benglish\b", r"\bnetherlands\b", r"\bdutch\b")),
        ("ancient_near_east", (r"\bparthian\b", r"\bseleucid\b", r"\biran\b",
                               r"\bmesopotamia\b")),
    )
    for label, patterns in markers:
        if any(re.search(pattern, culture, re.I) for pattern in patterns):
            return label
    if re.search(r"\b(?:liao|yuan|ming|qing|song)\b|辽|遼|元|明|清|宋", culture, re.I):
        return "china"
    return None


def _date_intervals(value: object) -> list[tuple[int, int]]:
    """Parse approximate exhibition dates from structured museum date strings."""
    text = " ".join(str(part) for part in value if part) if isinstance(value, list) else str(value or "")
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    intervals: list[tuple[int, int]] = []
    for match in re.finditer(
            r"(?P<a>\d{1,4})(?P<abce>\s*BCE)?\s*-\s*(?P<b>\d{1,4})(?P<bbce>\s*BCE)?",
            text, re.I):
        first = int(match.group("a"))
        second = int(match.group("b"))
        if len(match.group("b")) <= 2 and first >= 1000 and second < 100:
            second += (first // 100) * 100
            if second < first:
                second += 100
        if match.group("abce"):
            first = -first
        if match.group("bbce"):
            second = -second
        intervals.append((min(first, second), max(first, second)))
    if intervals:
        return intervals
    for ordinal in range(1, 22):
        prefix = r"\b(?:(early|mid|late)\s+)?" + str(ordinal) + r"(?:st|nd|rd|th)\s+century"
        match = re.search(prefix, text, re.I)
        if match:
            start = (ordinal - 1) * 100 + 1
            position = (match.group(1) or "").casefold()
            if position == "early":
                intervals.append((start, start + 49))
            elif position == "mid":
                intervals.append((start + 40, start + 69))
            elif position == "late":
                intervals.append((start + 60, start + 99))
            else:
                intervals.append((start, start + 99))
            return intervals
    for match in re.finditer(r"\b([1-9]\d{3})s\b", text):
        start = int(match.group(1))
        intervals.append((start, start + 99))
    for match in re.finditer(r"(?<!\d)([1-9]\d{3})(?!\d)", text):
        # Ignore four-digit tokens already represented by a range or decade.
        if any(start <= int(match.group(1)) <= end for start, end in intervals):
            continue
        intervals.append((int(match.group(1)), int(match.group(1))))
    return intervals


def _has_exact_year_range(value: object) -> bool:
    text = " ".join(str(part) for part in value if part) if isinstance(value, list) else str(value or "")
    text = re.sub(r"[\u2010-\u2015]", "-", text)
    return bool(re.search(r"\d{3,4}\s*-\s*\d{1,4}", text))


def _era_match(record: dict, era: dict) -> bool:
    structured = metadata_era_text(record)
    target = ERA_YEAR_RANGES.get(era["label"])
    date_text = " ".join(str(record.get(key, "")) for key in STRUCTURED_DATE_FIELDS)
    intervals = _date_intervals(date_text)
    named = any(_match_alias(structured, alias) for alias in era["aliases"])
    # A "mark" attribution is not necessarily an object made during the
    # mark's period.  Claims of "mark and period" or "mark and reign" provide
    # that attribution; a bare mark does not.
    if era["label"] in REIGN_LABELS:
        reign_source = " ".join(str(record.get(key, "")) for key in (
            "culture", "period", "reign"
        )).casefold()
        named = named and bool(re.search(
            rf"{re.escape(era['label'].casefold())}.{{0,45}}"
            r"(?:mark[ -](?:and|&)[ -](?:period|reign)|reign)",
            reign_source,
        ) or re.search(
            rf"{re.escape(era['label'].casefold())}[^,.;]{{0,45}}"
            r"\b(?:mark[ -](?:and|&)[ -](?:period|reign)|period|reign)\b",
            reign_source,
        ))
    # Japanese 1600s dates numerically overlap the Ming range.  The
    # structured culture is therefore part of the era conjunction, not a
    # separate advisory check.
    culture_ok, _ = _culture_compatible(record, {"era_constraints": [era]})
    if not culture_ok:
        return False
    if target is None or not intervals:
        # A prose claim or a bare dynasty label without a structured interval
        # does not satisfy.  This deliberately favours false negatives.
        return False
    interval_match = any(max(start, target[0]) <= min(end, target[1])
                         for start, end in intervals)
    if era["label"] in REIGN_LABELS:
        # Exact reign queries require the reign label itself, not merely a
        # numeric date that happens to intersect its range.
        return named and interval_match
    # The broad label must be present in a source-controlled field, alongside
    # the structured date interval.  Numeric overlap alone is not attribution.
    return bool(named and interval_match)


def _reign_is_intersected(record: dict, label: str) -> bool:
    structured = metadata_era_text(record)
    aliases = DYNAMIC_ERAS[label]
    return any(_match_alias(structured, alias) for alias in aliases)


def _era_constraints_match(record: dict, plan: dict) -> bool:
    """Require each reign and its broader dynasty to agree when both are named."""
    eras = plan.get("era_constraints", [])
    if not eras:
        return True
    checks = {era["label"]: _era_match(record, era) for era in eras}
    if not all(checks.values()):
        return False
    broad = [era for era in eras if not era.get("implied_by")]
    narrow = [era for era in eras if era.get("implied_by")]
    for era in narrow:
        # An explicit broader dynasty is authoritative over a reign named in
        # the same structured attribution.
        parents = [row for row in broad if era["label"] in row.get("aliases", ())]
        if parents and not any(_reign_is_intersected(record, row["label"])
                               for row in parents):
            return False
    return True


def _culture_compatible(record: dict, plan: dict) -> tuple[bool, str | None]:
    if not plan.get("era_constraints"):
        return True, _culture_class(record)
    culture_class = _culture_class(record)
    return culture_class == "china", culture_class


def _concept_text(record: dict, source: str, family: str | None = None) -> str:
    if family == "form":
        keys = OBJECT_IDENTITY_FIELDS
    else:
        keys = (*OBJECT_IDENTITY_FIELDS, "medium", "description", "technique")
    return " ".join(str(record.get(key, "")) for key in keys).casefold()


def reference_metadata_score(record: dict, source: str, plan: dict) -> tuple[float, dict]:
    evidence = metadata_concept_keys(record, source)
    if "porcelain" in plan["concept_keys"]:
        # The material is a hard constraint, but do not let generic ceramic
        # vocabulary stand in for a request that explicitly says porcelain.
        evidence = [key for key in evidence if key != "porcelain"]
    wanted = set(plan["concept_keys"])
    explicit_forms = {key for key in wanted if CONCEPT_BY_KEY[key].family == "form"}
    text = metadata_concept_text(record, source)
    identity = " ".join(str(record.get(key, "")) for key in
                        OBJECT_IDENTITY_FIELDS).casefold()
    concepts = {key: key in evidence for key in wanted}
    if "porcelain" in wanted:
        # Explicit Chinese ``瓷`` maps to the ceramic material family.  The
        # museum may correctly catalogue a Longquan object as stoneware; it is
        # still within the user's ceramic intent.
        concepts["porcelain"] = bool(MATERIAL_EVIDENCE.search(text))
    total = max(1, len(wanted))
    concept_fraction = sum(concepts.values()) / total
    era_checks = {era["label"]: _era_match(record, era)
                  for era in plan["era_constraints"]}
    era_fraction = sum(era_checks.values()) / max(1, len(era_checks))
    fragment = bool(re.search(r"\b(shard|fragment|fragmentary|sherd)\b|碎片|残片|殘片",
                              identity, re.I))
    # A small score penalty was frozen before rankings were inspected.  It
    # removes catalogue details/shards when whole-object photographs are
    # available, but does not assert that every rejected candidate is wrong.
    form_evidence = {key: bool(re.search(CONCEPT_BY_KEY[key].evidence_pattern,
                                         _concept_text(record, source, "form"), re.I))
                     for key in explicit_forms}
    form_ok = all(form_evidence.values())
    material_text = " ".join(str(record.get(key, "")) for key in
                             ("medium", "technique", "description", "type", "classification"))
    material_ok = (not plan.get("required_material")
                   or bool(MATERIAL_EVIDENCE.search(material_text)))
    culture_ok, culture_class = _culture_compatible(record, plan)
    era_ok = _era_constraints_match(record, plan)
    strict_match = all(concepts.values()) and form_ok and material_ok and culture_ok and era_ok
    failure_reasons = []
    failure_reasons.extend(f"missing_concept:{key}" for key, matched in concepts.items() if not matched)
    failure_reasons.extend(f"missing_form:{key}" for key, matched in form_evidence.items() if not matched)
    if not material_ok:
        failure_reasons.append("material_mismatch")
    if not culture_ok:
        failure_reasons.append("culture_not_china")
    failure_reasons.extend(f"missing_era:{key}" for key, matched in era_checks.items() if not matched)
    if not era_ok and all(era_checks.values()):
        failure_reasons.append("era_constraints_conflict")
    # Prefer complete object recordings.  A cover/lid/stand is a component
    # photograph and lacks the full silhouette expected from a reference bank.
    component = bool(re.search(r"\b(?:cover|lid|stand|base|fragment)\b|盖|蓋|底座",
                               identity, re.I))
    score = (0.60 * concept_fraction + 0.20 * era_fraction + 0.10 * float(bool(identity))
             + 0.10 * float(form_ok) - 0.35 * float(fragment or component))
    return max(0.0, score), {
        "concept_match": concepts, "concept_fraction": concept_fraction,
        "era_match": era_checks, "era_fraction": era_fraction,
        "form_match": form_ok, "form_evidence": form_evidence,
        "material_match": material_ok, "required_material": plan.get("required_material", []),
        "culture_class": culture_class, "culture_compatible": culture_ok,
        "strict_match": strict_match, "failure_reasons": failure_reasons,
        "fragment": fragment, "component_record": component,
        "metadata_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def metadata_concept_text(record: dict, source: str) -> str:
    return " ".join(str(record.get(key, "")) for key in (
        "title", "medium", "description", "objectName", "classification",
        "type", "technique", "culture", "period", "reign", "objectDate",
        "creation_date", "date",
    )).casefold()


def raw_search_metadata(record: dict) -> dict:
    """Expose original museum fields without rewriting their strings."""
    return {key: record.get(key) for key in (
        "objectID", "id", "accessionNumber", "accession_number", "title", "culture",
        "objectDate", "creation_date", "period", "reign", "medium", "technique",
        "objectName", "type", "classification", "department", "isPublicDomain",
        "share_license_status", "objectURL", "url", "primaryImageSmall", "primaryImage",
        "images",
    ) if record.get(key) is not None}


def _source_record(source: str, data: dict) -> dict:
    return data.get("data", data) if source == "cleveland" else data


class DynamicMuseumAcquirer:
    """Live, cached Cleveland/Met acquisition for one reproducible query plan."""

    def __init__(self, root: Path, output: Path, exclusions: GalleryExclusions,
                 *, offline: bool = False, timeout: tuple[float, float] = (8, 40),
                 cache_dir: Path | None = None):
        self.root = root
        self.output = output
        self.cache_dir = cache_dir or output
        self.exclusions = exclusions
        self.offline = offline
        self.http = requests.Session()
        self.http.headers["User-Agent"] = "CurioDynamicReferenceResearch/4.0 (reproducible evaluation)"
        self.http.headers["Accept"] = "application/json,image/*;q=0.9,*/*;q=0.5"
        self.timeout = timeout
        for name in ("http/search", "http/object", "http/image", "candidates", "metadata", "images"):
            (output / name).mkdir(parents=True, exist_ok=True)
        # HTTP cache may be shared by runs whose audit directories differ.
        (self.cache_dir / 'http/image').mkdir(parents=True, exist_ok=True)
        self.request_log: list[dict] = []
        self.failures: list[dict] = []
        self.discovery_reports: list[dict] = []

    def _log_request(self, request_id: str, url: str, status: int | None, *,
                     content_type: str, byte_count: int, started: float, cached: bool,
                     error: str | None = None) -> None:
        row = {"request_id": request_id, "request_url": url, "status_code": status,
               "content_type": content_type, "byte_count": byte_count,
               "elapsed_ms": round((time.perf_counter() - started) * 1000, 3),
               "from_cache": cached, "retrieved_at": datetime.now(timezone.utc).isoformat()}
        if error:
            row["error"] = error
        self.request_log.append(row)

    def get_json(self, url: str, kind: str, request_id: str, *, retries: int = 2) -> dict:
        cache = self.cache_dir / "http" / kind / (hashlib.sha256(url.encode()).hexdigest() + ".json")
        if cache.exists():
            payload = json.loads(cache.read_text(encoding="utf-8"))
            self.request_log.append({
                **payload["request"], "from_cache": True,
                "original_retrieved_at": payload["request"].get("retrieved_at"),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_ms": 0,
            })
            return payload["response"]
        if self.offline:
            raise FileNotFoundError(f"offline cache miss: {url}")
        started = time.perf_counter()
        response = None
        last_error = None
        for attempt in range(retries + 1):
            try:
                response = self.http.get(url, timeout=self.timeout)
                self._log_request(request_id, url, response.status_code,
                                  content_type=response.headers.get("content-type", ""),
                                  byte_count=len(response.content), started=started,
                                  cached=False)
                if response.status_code == 429 or response.status_code >= 500:
                    response.raise_for_status()
                response.raise_for_status()
                data = response.json()
                _atomic_json(cache, {"request": self.request_log[-1], "response": data})
                return data
            except (requests.RequestException, ValueError) as exc:
                last_error = str(exc)
                if response is None:
                    self.failures.append({
                        "query_id": request_id, "stage": "http_attempt",
                        "request_url": url, "attempt": attempt + 1, "error": last_error,
                    })
                if attempt < retries:
                    time.sleep(1.5 ** attempt)
        raise RuntimeError(f"request failed after retries: {url}: {last_error}")

    def get_image(self, row: dict, request_id: str) -> tuple[bytes, dict]:
        url = row["image_url"]
        digest = hashlib.sha256(url.encode()).hexdigest()
        cache = self.cache_dir / "http/image" / (digest + ".bin")
        if cache.exists():
            binary = cache.read_bytes()
            status_path = self.cache_dir / "http/image" / (digest + ".status")
            status = int(status_path.read_text(encoding="utf-8")) if status_path.exists() else 200
            request = {"request_id": request_id, "request_url": url, "status_code": status,
                       "content_type": "image/*", "byte_count": len(binary), "elapsed_ms": 0,
                       "from_cache": True, "retrieved_at": datetime.now(timezone.utc).isoformat()}
            self.request_log.append(request)
            return binary, request
        if self.offline:
            raise FileNotFoundError(f"offline image cache miss: {url}")
        started = time.perf_counter()
        response = self.http.get(url, timeout=(8, 60))
        self._log_request(request_id, url, response.status_code,
                          content_type=response.headers.get("content-type", ""),
                          byte_count=len(response.content), started=started, cached=False)
        response.raise_for_status()
        cache.write_bytes(response.content)
        (cache.parent / (digest + ".status")).write_text(str(response.status_code))
        return response.content, self.request_log[-1]

    def search(self, source: str, term: str, *, limit: int = 30,
               offset: int = 0, date_range: dict | None = None) -> list[dict]:
        if source == "cleveland":
            params = {"q": term, "cc0": 1, "has_image": 1, "culture": "China", "limit": limit,
                      "skip": offset}
            if date_range:
                params.update({"created_after": date_range["start"],
                               "created_before": date_range["end"]})
            url = SOURCE_BASES[source] + "?" + urlencode(params)
        elif source == "met":
            params = {"q": term, "hasImages": "true", "limit": limit, "offset": offset}
            if date_range:
                params.update({"dateBegin": date_range["start"],
                               "dateEnd": date_range["end"]})
            url = SOURCE_BASES[source] + "?" + urlencode(params)
        else:
            raise ValueError(f"Unknown source: {source}")
        rid = hashlib.sha256(url.encode()).hexdigest()[:16]
        data = self.get_json(url, "search", rid)
        return [row["id"] for row in data.get("data", []) if row.get("id") is not None] if source == "cleveland" else (data.get("objectIDs") or [])

    def object(self, source: str, source_id: str) -> dict:
        url = (SOURCE_BASES["cleveland"] + str(source_id) if source == "cleveland"
               else MET_OBJECT.format(source_id))
        data = self.get_json(url, "object", f"obj-{source}-{source_id}")
        return _source_record(source, data)

    def fetch(self, query_id: str, plan: dict, *, per_query: int = 20,
              max_references_considered: int = 60, page_depth: int = 3,
              max_metadata_candidates: int = 300) -> dict:
        candidate_rows: list[dict] = []
        seen: set[str] = set()
        accepted: list[dict] = []
        rejected: list[dict] = []
        terms = plan.get("discovery_terms") or plan["query_terms"]
        # Broader identity searches first: exact compound searches can return
        # no IDs even when the museum contains valid objects under adjacent
        # vocabulary (for example Liao "Leather Bag-Shaped Flasks").
        terms = sorted(terms, key=lambda term: (len(term.split()), len(term)))
        # The dynasty is a hard structured requirement.  Searches without it
        # waste the bounded request budget and make sparse queries even
        # sparser.  This is reference discovery only, never ranking.
        era_aliases = [
            value.casefold()
            for era in plan.get("era_constraints", [])
            for value in (*era["aliases"], era["label"], era["label"].replace(" ", ""))
        ]
        def era_bound(term: str) -> bool:
            lowered = term.casefold()
            return any(re.search(r"(?<![a-z])" + re.escape(alias) + r"(?![a-z])", lowered)
                       for alias in era_aliases)
        era_filter_removed = len(terms) - sum(
            not era_aliases or era_bound(term)
            for term in terms
        )
        # API date fields already enforce the range. Keep a short form-only
        # query such as "Meiping vase" as a fallback: Met returns structured
        # Kangxi dates for many objects whose free-text record omits the reign.
        terms = [term for term in terms if not era_aliases or era_bound(term)
                 or (len(term.split()) <= 3 and any(
                     alias.casefold() in term.casefold()
                     for key in plan.get("concept_keys", [])
                     if CONCEPT_BY_KEY[key].family == "form"
                     for alias in CONCEPT_BY_KEY[key].aliases
                     if not re.search(r"[\u3400-\u9fff]", alias)))]
        terms.sort(key=lambda term: (len(term.split()), len(term)))
        per_term_limit = max(8, per_query)
        metadata_inspected = 0
        stop_discovery = False
        terms_searched = []
        for term_index, term in enumerate(terms):
            term_candidates = 0
            if len(strict_reference_rows(accepted, plan)) >= 5:
                break
            terms_searched.append(term)
            for source in plan["sources"]:
                if term_candidates >= per_term_limit:
                    break
                try:
                    # Deep paging handles a known API limitation: free-text
                    # ranking can put paintings/prints above the requested
                    # ceramic form.  Each page remains logged and bounded.
                    ids = []
                    page_size = min(100, per_query)
                    for page in range(page_depth):
                        ids.extend(self.search(
                            source, term, limit=page_size, offset=page * page_size,
                            date_range=plan.get("search_date_range")
                        ))
                except Exception as exc:
                    self.failures.append({"query_id": query_id, "source": source,
                                          "term": term, "stage": "search", "error": str(exc)})
                    continue
                term_quota = max(4, per_term_limit // max(1, len(plan["sources"])))
                considered_for_source = 0
                for source_id in ids:
                    rid = f"{'cma' if source == 'cleveland' else 'met'}_{source_id}"
                    if rid in seen:
                        continue
                    seen.add(rid)
                    if term_candidates >= per_term_limit:
                        break
                    if considered_for_source >= term_quota:
                        break
                    if metadata_inspected >= max_metadata_candidates:
                        stop_discovery = True
                        break
                    considered_for_source += 1
                    term_candidates += 1
                    metadata_inspected += 1
                    try:
                        record = self.object(source, str(source_id))
                        normalized = normalize_record(source, record)
                        score, diagnostics = reference_metadata_score(record, source, plan)
                        candidate = {**normalized, "metadata_score": score,
                                     "original_metadata": raw_search_metadata(record),
                                     "metadata_api_url": (SOURCE_BASES["cleveland"] + str(source_id) if source == "cleveland" else MET_OBJECT.format(source_id)),
                                     "metadata_diagnostics": diagnostics,
                                     "discovery_term": term, "discovery_url":
                                     (SOURCE_BASES["cleveland"] + "?" + urlencode({"q": term, "cc0": 1, "has_image": 1, "culture": "China", "skip": 0, "limit": min(10, per_query)})
                                      if source == "cleveland" else SOURCE_BASES["met"] + "?" +
                                      urlencode({"q": term, "hasImages": "true", "limit": min(10, per_query), "offset": 0}))}
                        candidate_rows.append(candidate)
                        if diagnostics["fragment"] or (plan["concept_keys"] and not any(diagnostics["concept_match"].values())):
                            rejected.append({"reference_id": rid, "stage": "metadata_screen", "reason": "fragment_or_no_requested_concept_evidence"})
                            continue
                        image_required = diagnostics["strict_match"] or (
                            diagnostics["form_match"] and diagnostics["material_match"]
                            and all(diagnostics["concept_match"].values()) and (
                                not plan.get("era_constraints")
                                or any(diagnostics["era_match"].values())))
                        exclusion = self.exclusions.reason(normalized)
                        if exclusion:
                            rejected.append({"reference_id": rid, "stage": "identity_exclusion",
                                             **exclusion})
                            continue
                        if not image_required:
                            rejected.append({"reference_id": rid, "stage": "strict_metadata_rejection",
                                             "reason": "hard_constraint_failed",
                                             "failure_reasons": diagnostics["failure_reasons"]})
                            continue
                        binary, request = self.get_image(candidate, f"img-{rid}")
                        with Image.open(io.BytesIO(binary)) as opened:
                            image = ImageOps.exif_transpose(opened).convert("RGB")
                            if min(image.size) < 256 or max(image.size) < 512:
                                rejected.append({"reference_id": rid,
                                                 "reason": "below_quality_floor", "size": list(image.size)})
                                continue
                            candidate["image_sha256"] = hashlib.sha256(binary).hexdigest()
                            candidate.update(image_fingerprints(image))
                            candidate.update(width=image.width, height=image.height)
                            candidate["accepted_at"] = datetime.now(timezone.utc).isoformat()
                            candidate["source_response_sha256"] = hashlib.sha256(
                                json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                            exclusion = self.exclusions.reason(candidate)
                            if exclusion:
                                rejected.append({"reference_id": rid, "stage": "pixel_exclusion",
                                                 **exclusion})
                                continue
                        image_path = self.output / "images" / (rid + ".jpg")
                        image_path.write_bytes(binary)
                        metadata_path = self.output / "metadata" / (rid + ".json")
                        _atomic_json(metadata_path, {"request_url": candidate["metadata_api_url"],
                                                    "record": record})
                        candidate["image_path"] = image_path.relative_to(self.root).as_posix()
                        candidate["metadata_path"] = metadata_path.relative_to(self.root).as_posix()
                        candidate["image_request"] = request
                        duplicate_of = None
                        for prior in accepted:
                            ph = _phash_distance(candidate.get("phash64"), prior.get("phash64"))
                            ah = _ahash_distance(candidate.get("ahash256"), prior.get("ahash256"))
                            if candidate["image_sha256"] == prior["image_sha256"] or (ph is not None and ph <= 6) or (ah is not None and ah <= 4):
                                duplicate_of = {"reference_id": prior["reference_id"],
                                                "phash64_distance": ph, "ahash256_distance": ah}
                                break
                        if duplicate_of:
                            rejected.append({"reference_id": rid, "stage": "near_duplicate",
                                             "duplicate_of": duplicate_of})
                            continue
                        accepted.append(candidate)
                    except Exception as exc:
                        self.failures.append({"query_id": query_id, "source": source,
                                              "source_id": source_id, "term": term,
                                              "stage": "metadata_or_image", "error": str(exc)})
                if stop_discovery:
                    break
            if stop_discovery:
                break
        self.discovery_reports.append({
            "query_id": query_id, "terms": terms,
            "terms_searched": terms_searched,
            "era_filter_removed_term_count": era_filter_removed,
            "strict_candidates_at_stop": len(strict_reference_rows(accepted, plan)),
            "metadata_candidates_inspected": metadata_inspected,
            "search_ids_returned": len(seen),
            "search_failures": sum(
                term in str(row.get("request_url", ""))
                and row.get("stage") in {"search", "http_attempt"}
                for row in self.failures for term in terms
            ),
            "page_depth": page_depth,
            "per_query": per_query, "max_metadata_candidates": max_metadata_candidates,
        })
        return {"query_id": query_id, "plan": plan, "candidates": candidate_rows,
                "accepted": accepted, "rejected": rejected,
                "discovery": {"terms": terms, "metadata_candidates_inspected": metadata_inspected,
                              "page_depth": page_depth, "per_query": per_query,
                              "max_metadata_candidates": max_metadata_candidates}}

    def save_discovery(self, path: Path) -> None:
        _jsonl(path, self.discovery_reports)

    def save_manifest(self, path: Path) -> None:
        _jsonl(path, self.request_log)


def select_reference_set(accepted: list[dict], plan: dict, *, limit: int = 5,
                         allow_fallback: bool = False) -> list[dict]:
    """Select strict references, or explicitly labelled concept fallbacks."""
    strict_rows = strict_reference_rows(accepted, plan)
    if not allow_fallback:
        candidate_rows = strict_rows
    elif len(strict_rows) >= 2:
        candidate_rows = strict_rows
    else:
        candidate_rows = [row for row in accepted
                          if row.get("metadata_diagnostics", {}).get("concept_match")]
    ranked = sorted(candidate_rows, key=lambda row: (
        -bool(row.get("metadata_diagnostics", {}).get("strict_match")),
        -float(row.get("metadata_score", 0)), row["reference_id"]))
    selected, used = [], set()
    for row in ranked:
        family = f"{row['source']}:{re.sub(r'[^0-9a-z]+', '', row.get('accession', '').casefold())[:14]}:{re.sub(r'[^a-z0-9]+', '', row.get('title', '').casefold())[:32]}"
        if family in used:
            continue
        verified = bool(row.get("metadata_diagnostics", {}).get("strict_match"))
        selected.append({**row, "selection_family": family,
                         "reference_constraint_status": ("verified_requested_constraints" if verified else
                                                         "broad_visual_fallback_constraints_unverified")})
        used.add(family)
        if len(selected) >= limit:
            break
    return selected


def strict_references_available(accepted: list[dict], plan: dict) -> bool:
    return len(strict_reference_rows(accepted, plan)) >= 2


def strict_reference_rows(accepted: list[dict], plan: dict) -> list[dict]:
    """Deduplicate hard-constraint matches while preserving deterministic order."""
    rows = {}
    for row in accepted:
        diagnostics = row.get("metadata_diagnostics", {})
        if diagnostics.get("strict_match"):
            rows.setdefault(row["reference_id"], row)
    unique = []
    seen_families = set()
    for row in rows.values():
        if row.get("metadata_diagnostics", {}).get("component_record"):
            continue
        # Accession prefixes can differ on sibling object recordings; using the
        # companion title as a conservative fallback avoids treating three
        # captioned copies as three independent references.
        accession = re.sub(r"[^0-9a-z.]+", "", str(row.get("accession", "")).casefold())
        title = re.sub(r"[^a-z0-9]+", "", str(row.get("title", "")).casefold())
        # Keep the leading accession series even when titles differ across
        # captioned components of the same physical set.
        parts = accession.split(".")
        # Accession families are meaningful for Cleveland (e.g. .a/.b sibling
        # captions). Met's accession numbers are independent object identities;
        # collapsing those by title loses genuine, visually distinct references.
        if row.get("source") == "cleveland" and len(parts) >= 2:
            parent = ".".join(parts[:2])
            family = f"cleveland:{parent}"
        else:
            family = f"{row.get('source', '')}:{accession or title}"
        if family in seen_families:
            continue
        seen_families.add(family)
        unique.append(row)
    return unique


def selection_diagnostics(accepted: list[dict], selected: list[dict], plan: dict,
                          *, fallback_used: bool) -> dict:
    strict_count = len(strict_reference_rows(accepted, plan))
    fallback_count = sum(row.get("reference_constraint_status") ==
                         "broad_visual_fallback_constraints_unverified" for row in selected)
    verified_count = len(selected) - fallback_count
    return {
        "accepted_count": len({row["reference_id"] for row in accepted}),
        "strict_accepted_count": strict_count,
        "selected_count": len(selected),
        "broad_fallback_count": fallback_count,
        "verified_selected_count": verified_count,
        "validation_status": ("verified" if fallback_count == 0 else
                              "partially_unverified" if fallback_count < len(selected) else
                              "unverified"),
        "status": ("strict_references_selected" if strict_count >= 2 else
                   "insufficient_strict_references"),
        "fallback_used_for_discovery": fallback_used,
        "policy": "all requested concepts plus form, material, Chinese culture, and structured era checks",
    }


def load_gallery(root: Path) -> tuple[list[dict], GalleryExclusions]:
    rows = [json.loads(line) for line in
            (root / "data/snapshots/collection_v2.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()]
    raw = {}
    for row in rows:
        path = root / "data/raw" / row["source"] / f"{row['source_id']}.json"
        if path.exists():
            raw[row["item_id"]] = json.loads(path.read_text(encoding="utf-8"))
    exclusions = GalleryExclusions(rows, raw)
    fingerprint_path = root / "data/references/gallery_fingerprints.json"
    if fingerprint_path.exists():
        saved = json.loads(fingerprint_path.read_text(encoding="utf-8"))
        for item_id, value in saved.get("phash64", {}).items():
            exclusions.add_gallery_phash(item_id, value)
    else:
        for row in rows:
            path = root / row["image_path"]
            if str(row["item_id"]).startswith("cma_") and path.exists():
                with Image.open(path) as image:
                    exclusions.add_gallery_phash(row["item_id"], image_fingerprints(image)["phash64"])
    return rows, exclusions
