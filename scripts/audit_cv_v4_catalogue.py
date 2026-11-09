"""Audit CV v4 catalogue-derived ontology without changing labels or corpus.

The script rebuilds labels in memory through the builder's public helpers, then
emits item evidence and aggregates for semantic review. It never writes
`data/`, `eval/`, or the official v4 label directory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/cv_v4_20261108/catalogue_audit"
BUILDER_PATH = ROOT / "scripts/build_cv_v4_official.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("cv_v4_official_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def raw_for(item: dict) -> dict:
    path = ROOT / "data/raw" / item["source"] / f"{item['source_id']}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def plain(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(plain(v) for v in value if plain(v))
    return re.sub(r"<[^>]+>", " ", str(value)).strip()


def excerpt(value: str, match: re.Match, radius: int = 120) -> str:
    left, right = max(0, match.start() - radius), min(len(value), match.end() + radius)
    return re.sub(r"\s+", " ", value[left:right]).strip()


def add(issue_rows: list[dict], item: dict, code: str, severity: str, evidence: str,
        derived, expected: str, fix: str, *, field: str = "", raw: dict | None = None) -> None:
    issue_rows.append({
        "item_id": item["item_id"], "source": item["source"], "split": item["split"],
        "object_group": item["object_group"], "code": code, "severity": severity,
        "field": field, "title": item.get("title", ""), "evidence": evidence,
        "derived": derived, "recommended_outcome": expected, "deterministic_fix": fix,
        "raw_accession": (raw or {}).get("accessionNumber" if item["source"] == "met" else "accession_number", ""),
    })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    protected_roots = [(ROOT / "data").resolve(), (ROOT / "eval").resolve(),
                       (ROOT / "artifacts/cv_v4_20261108/official").resolve()]
    if any(output == protected or protected in output.parents for protected in protected_roots):
        raise ValueError("Audit output cannot be inside protected data/eval/official locations")
    output.mkdir(parents=True, exist_ok=True)

    builder = load_builder()
    catalog_path = ROOT / "data/snapshots/collection_v2.jsonl"
    items = [json.loads(line) for line in catalog_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = [builder.build_one(item) for item in items]
    by_label = {row["item_id"]: row for row in labels}
    issues: list[dict] = []
    rows: list[dict] = []

    # Use the builder's own expanded patterns to record exact false-positive
    # boundaries. A raw term is only reported when it changes a derived label.
    method_tokens = {
        "blue": r"\bblue\b", "green": r"\bgreen\b", "black": r"\bblack\b",
        "dark": r"\bdark\b", "red": r"\bred\b", "gray": r"\bgray\b",
        "grey": r"\bgrey\b", "glass": r"\bglass\b", "lacquer": r"\blacquer(?:ed)?\b",
        "ceramic": r"\b(?:ceramic|glaze[ds]?|glazed)\b",
    }
    technique_markers = re.compile(
        r"\b(?:painted|painting|glaze(?:d)?|enamel(?:led)?|lacquer(?:ed)?|carved|incised|inlaid|"
        r"blown|cast|woven|weave|dyed|print(?:ed)?|decorated|decorative|technique|high-fired|low-fired)\b",
        re.I,
    )
    ambiguous_comparison = re.compile(
        r"\b(?:another|other (?:objects?|works?|examples?|bowls?|vases?|jars?|plates?|paintings?)|unique among|"
        r"predecessors?|predecessor of|prototype|comparable|unlike|in contrast|instead of|earlier|later|"
        r"historical|illustrat(?:e|es|ed|ing)|depict(?:s|ed|ing)|representation)",
        re.I,
    )
    direct_object = re.compile(
        r"\b(?:this|the) (?:[A-Za-z-]+ ){0,4}(?:is|has|shows?|exhibits?|displays?|covered|glazed|colou?red)|"
        r"\b(?:this|the) [A-Za-z-]+ (?:vase|jar|bowl|cup|bottle|dish|plate|object)[^.]{0,30}(?:is|has|shows?|glazed)|"
        r"\b(?:this|the) (?:[A-Za-z-]+ )?(?:glaze|body|surface) (?:is|has|shows?)|"
        r"\b(?:this|the) .{0,80}\b(?:glaze|enamel|colour|color|patina)\b",
        re.I,
    )

    for item, label in zip(items, labels):
        raw = raw_for(item)
        source = item["source"]
        derived = label["derived_ontology"]
        fields = {
            "title": plain(raw.get("title")),
            "identity_type": plain(raw.get("type") if source == "cleveland" else raw.get("objectName")),
            "classification": plain(raw.get("classification")) if source == "met" else plain(raw.get("collection")),
            "medium_or_technique": plain(raw.get("technique") or raw.get("medium")),
            "culture_or_date": " ".join(plain(raw.get(k)) for k in
                                       (("culture", "creation_date", "date_text") if source == "cleveland"
                                        else ("period", "dynasty", "reign", "objectDate"))),
            "tags": plain([x.get("term") if isinstance(x, dict) else x for x in (raw.get("tags") if source == "met" else raw.get("artists_tags")) or []]),
            "description": " ".join(plain(raw.get(k)) for k in ("description", "did_you_know", "early_education_description")) if source == "cleveland" else "",
        }
        rows.append({
            "item_id": item["item_id"], "source": source, "split": item["split"],
            "object_group": item["object_group"], "raw_fields": fields,
            "derived": derived,
        })

        # CMA's type is frequently a material/class. Preserve this as a type
        # source only when it agrees with or adds an object noun not available elsewhere.
        if source == "cleveland":
            type_term = fields["identity_type"].strip().casefold()
            type_materials = {"glass", "jade", "metalwork", "enamel", "lacquer", "wood", "stone", "ivory", "textile"}
            if type_term in type_materials:
                add(issues, item, "cma_type_is_material_or_medium", "warning",
                    f"raw `type` is `{fields['identity_type']}`, a material/class rather than the pictured object form; title is `{fields['title']}`.",
                    fields["identity_type"], "Keep it as a material/class facet; do not use it as an object-type label.",
                    "For CMA, parse object type from title/object identity terms, and use `type` only under an explicit material/class mapping. Never append `type` text to object_type merely because it matches a material noun.",
                    field="identity_type", raw=raw)

        # The two local patterns omit or mis-handle word boundaries. Detect
        # actual matches that the broader pattern produces but a strict token does not.
        color_text = fields["title"] + " " + fields["medium_or_technique"] + " " + fields["description"]
        for color in derived["colors"]:
            token = method_tokens.get(color)
            if token is None:
                continue
            matched = re.search(token, color_text, re.I)
            if matched is None:
                builder_pattern = next((regex for name, regex in builder.COLOR_PATTERNS if name == color), None)
                builder_match = re.search(builder_pattern, color_text, re.I) if builder_pattern else None
                location = excerpt(color_text, builder_match) if builder_match else color_text[:240]
                matched_fragment = re.sub(r"\s+", " ", builder_match.group(0)) if builder_match else ""
                add(issues, item, "color_without_token_boundary", "error",
                    f"builder pattern for {color} matched literal `{matched_fragment}`, which is not standalone `{color}`: " + location,
                    derived["colors"], f"Do not emit `{color}` for this text.",
                    "Use compiled token patterns with `(?<![A-Za-z])` / `(?![A-Za-z])` boundaries and labels such as `gray|grey`, `red`, and `dark`; test each pattern against the raw field that produced it.",
                    field="colors", raw=raw)

        # Color words in technique labels are often real glaze/enamel colours.
        # Keep them as technique evidence; only flag general medium-with-
        # decorative-color cases when no direct glaze/enamel/body syntax exists.
        direct_colour_technique = re.compile(r"\b(?:glaze[ds]?|glazed|enamel(?:led)?|body|slip|patina|pigment)\b", re.I)
        for color in derived["colors"]:
            pattern = next((regex for name, regex in builder.COLOR_PATTERNS if name == color), None)
            match = re.search(pattern, fields["medium_or_technique"], re.I) if pattern else None
            if match and not direct_colour_technique.search(fields["medium_or_technique"]):
                add(issues, item, "color_mentioned_only_in_medium_label", "warning",
                    f"medium/technique `{fields['medium_or_technique']}` produced color `{color}`: " + excerpt(fields["medium_or_technique"], match),
                    derived["colors"], "Store as catalogue coating/technique evidence, not a separate observed palette.",
                    "Parse medium and technique into separate fields first. Preserve `glaze`, `enamel`, `body`, `slip`, `patina`, and `pigment` color statements as technique/material facets; require a separate palette field before treating the extracted colors as the image's complete visible palette.",
                    field="colors", raw=raw)

        # Text mentions in curatorial prose can describe a different work or a
        # historical type. The builder has no attribution parser, so flag a
        # conservative high-precision set rather than claiming all mentions bad.
        desc = fields["description"]
        if desc:
            for kind, values in (("motifs", derived["motifs"]), ("colors", derived["colors"])):
                patterns = builder.MOTIF_PATTERNS if kind == "motifs" else builder.COLOR_PATTERNS
                for value in values:
                    pattern = next((regex for name, regex in patterns if name == value), None)
                    match = re.search(pattern, desc, re.I) if pattern else None
                    if match:
                        nearby = excerpt(desc, match)
                        sentence_matches = list(re.finditer(r"[^.!?]+[.!?]?", desc))
                        current_sentence = next((sentence.group(0) for sentence in sentence_matches
                                                 if sentence.start() <= match.start() < sentence.end()), desc)
                        strong_ambiguity = bool(ambiguous_comparison.search(current_sentence))
                        if strong_ambiguity:
                            add(issues, item, "description_comparison_risk", "warning",
                                f"`{value}` comes from description and is in a possible comparison/historical context: {nearby}",
                                values, "Leave unknown unless the sentence is a direct positive statement about this object.",
                                "Do not parse free prose unless each match has a sentence-level subject rule. Use a curated `decoration`, `palette`, or `visible_supplement` field; preserve prose separately and require direct lexical patterns (`this ... is [colour]`, `decorated with ...`) for automatic support.",
                                field=kind, raw=raw)
                        needs_subject_review = (
                            kind == "colors" and
                            not direct_object.search(current_sentence) and
                            not re.search(r"\b(?:vase|jar|bowl|cup|bottle|dish|plate|object|body|glaze|enamel|surface|patina)\b", current_sentence, re.I)
                        )
                        if needs_subject_review:
                            add(issues, item, "description_needs_subject_review", "info",
                                f"Description-only `{value}` has no direct object/surface subject in its sentence: {nearby}",
                                values, "Review the sentence subject before using the value as this item's property.",
                                "Split prose into sentences, retain match offsets, and require direct object/glaze/body syntax or a curated decoration field before promotion. If no attribution can be proved, leave the property unknown.",
                                field=kind, raw=raw)

                        # `red tea`, `black bear`, and a named comparand are not
                        # visible surface properties of the record even though
                        # they are colour tokens. Apply the audit correction only
                        # to the high precision phrases seen in raw prose.
                        nonsurface = (
                            (value == "red" and re.search(r"\bred tea\b", current_sentence, re.I)) or
                            (value == "black" and re.search(r"\bblack (?:bear|tea)\b", current_sentence, re.I)) or
                            (value == "white" and re.search(r"\bWhite Tiger\b", current_sentence)) or
                            (value == "green" and re.search(r"\bGreen Dragon\b", current_sentence)) or
                            (value == "red" and re.search(r"\bRed Bird\b", current_sentence))
                        )
                        if kind == "colors" and nonsurface and not re.search(
                                r"\b(?:object|body|glaze|enamel|surface|patina|painting|figure|literal|scene)\b", current_sentence, re.I):
                            add(issues, item, "color_term_has_nonsurface_referent", "error",
                                f"`{value}` is description-only in a named non-surface phrase: {nearby}",
                                values, f"Do not emit `{value}` as this object's color.",
                                "Suppress named non-surface collocations (`red tea`, `black tea`, `White Tiger`, `Green Dragon`, `Red Bird`) unless the same sentence identifies a painted figure or palette for this object.",
                                field=kind, raw=raw)

        # Titles can have colour words in a named glaze/style rather than a
        # verified visible property. Report these for explicit review.
        for color in derived["colors"]:
            pattern = next((regex for name, regex in builder.COLOR_PATTERNS if name == color), None)
            match = re.search(pattern, fields["title"], re.I) if pattern else None
            if match and not re.search(pattern, fields["description"] + " " + fields["medium_or_technique"], re.I):
                add(issues, item, "title_color_is_named_style", "warning",
                    "Title supplies " + excerpt(fields["title"], match),
                    derived["colors"], "Treat title color/style as catalogue text, not a complete observed palette.",
                    "Require an explicit palette field or a color-in-title mapping that distinguishes named decorative styles (`blue-and-white`) from one visible color and records the evidence layer.",
                    field="colors", raw=raw)

        # Component detection: raw titles are the only structured component
        # signal, but parent/part semantics depend on grouping and image view.
        title = fields["title"]
        if re.search(r"\bstoppers?\b|\blid\b|\bcover\b|\bbase\b|\bmount\b", title, re.I):
            component_hint = bool(re.search(r"\((?:­?stopper|lid|cover|base|mount)\)|\bwith (?:a )?(?:stopper|lid|cover|base|mount)\b", title, re.I))
            expected = "component likely" if component_hint else "object likely; do not infer sibling from the word alone"
            add(issues, item, "component_semantics_need_view_context", "warning" if component_hint else "info",
                f"Title `{title}`; raw type `{fields['identity_type']}`; group `{item['object_group']}`.",
                derived["object_type"], expected,
                "Resolve component vs whole object from an explicit raw component/detail flag or a per-item visual-review record. Keep parent-fragment relation, component class, and photographed subject separate; do not infer from title alone.",
                field="object_type", raw=raw)

        # Era source is separated from description by the builder, which avoids
        # prose false positives. Record that as a passed invariant with evidence.
        era_text = fields["culture_or_date"]
        if any(token in era_text for token in ("Ming", "Tang", "Song", "Qing", "Yuan", "Han", "Zhou", "Liao", "Shang")):
            era_evidence = "era comes from culture/date/reign fields"
        else:
            era_evidence = "no era token in culture/date/reign fields"
        if derived["era"] and not era_text:
            add(issues, item, "era_without_era_field", "error", era_evidence, derived["era"], "No era.",
                "Derive era only from the dedicated authority/date fields; never from title, description, tags, or object classification.", field="era", raw=raw)

        # Material vs technique: the builder calls CMA technique `medium`. Flag
        # values with method verbs that are classification mismatches.
        medium = fields["medium_or_technique"]
        method_match = technique_markers.search(medium)
        if source == "cleveland" and method_match:
            add(issues, item, "technique_used_as_medium", "warning",
                "CMA technique field: " + excerpt(medium, method_match), derived["material"],
                "Keep technique separate; material should use an explicit material facet.",
                "Map CMA `technique` to technique/process, and resolve material from `support_materials`, title/object type only where explicit, or leave material null. Do not label a process word as a broad material.",
                field="material", raw=raw)

    issue_counts = dict(sorted(Counter(issue["code"] for issue in issues).items()))
    severity_counts = dict(sorted(Counter(issue["severity"] for issue in issues).items()))
    unique_item_count = len({issue["item_id"] for issue in issues})
    # Exact patterns for known false-positive tests, including invalid suffixes.
    pattern_probes = {}
    for name, pattern in builder.COLOR_PATTERNS:
        tests = {"blue": ["blue", "bluebird"], "red": ["red", "shattered"],
                 "black": ["black", "blackware"], "dark": ["dark", "darkroom"],
                 "gray": ["gray", "grayscale"], "grey": ["grey", "greyscale"]}.get(name, [name, name + "ware"])
        pattern_probes[name] = [{"text": s, "matches": bool(re.search(pattern, s, re.I))} for s in tests]

    # Concrete pipeline invariants and raw source agreements.
    groups = defaultdict(list)
    for item in items:
        groups[item["object_group"]].append(item)
    split_crossings = []
    for group, members in groups.items():
        if len({member["split"] for member in members}) > 1:
            split_crossings.append({"group_id": group, "splits": sorted({member["split"] for member in members}),
                                    "item_ids": [member["item_id"] for member in members]})
    mixed_groups = []
    for group, members in groups.items():
        if len(members) < 2:
            continue
        raw_types = sorted({plain(raw_for(member).get("type") if member["source"] == "cleveland" else raw_for(member).get("objectName")) for member in members})
        if len(raw_types) > 1:
            mixed_groups.append({
                "group_id": group, "split": members[0]["split"],
                "raw_types": raw_types,
                "members": [{"item_id": member["item_id"], "title": member["title"],
                             "raw_type": plain(raw_for(member).get("type") if member["source"] == "cleveland" else raw_for(member).get("objectName")),
                             "raw_medium_or_technique": member["medium"],
                             "derived": by_label[member["item_id"]]["derived_ontology"]} for member in members],
            })

    # Deterministic expected outcomes for named raw records used in the report.
    known_examples = {
        "cma_101719": {
            "observed": by_label["cma_101719"]["derived_ontology"]["object_type"],
            "evidence": "Title `Snuff Bottle (stopper)`; parent/body records share `cma_101717`.",
            "expected": ["stopper"],
            "status": "builder output is correct for this specific title",
        },
        "cma_123657": {
            "observed": by_label["cma_123657"]["derived_ontology"]["object_type"],
            "evidence": "Title `Snuff Bottle with Stopper (stopper)`; raw `type` is `Ceramic`.",
            "expected": ["stopper"],
            "status": "builder output is correct for this specific title; raw type remains a material/class mismatch",
        },
        "cma_130118": {
            "observed": by_label["cma_130118"]["derived_ontology"]["object_type"],
            "evidence": "Title `Goose and Goslings (Stopper)`; raw `type` is `Jade`.",
            "expected": ["stopper"],
            "status": "builder output is correct for this specific title",
        },
        "cma_135011": {
            "observed": by_label["cma_135011"]["derived_ontology"],
            "evidence": "Group `cma_135008`: raw titles `Covered Jar`, `Covered Jar (lid)`, `Covered Jar (stand)`; stand raw `type` is `Wood` but shares porcelain technique text.",
            "expected": "Separate photographed-record component and material from parent object; preserve group relation without copying parent semantics.",
            "status": "concrete unproven component/material semantics",
        },
        "met_41314": {
            "observed": by_label["met_41314"]["derived_ontology"],
            "evidence": "Medium is `Porcelain with overglazed enamel colors (Jingdezhen ware), coral stopper`; builder emits only porcelain material and a horse motif.",
            "expected": "material facets should include body porcelain, decoration enamel, and stopper coral; motif depends on the scene being regarded as decoration.",
            "status": "missing component material and incomplete motif semantics",
        },
        "cma_76525": {
            "observed": by_label["cma_76525"]["derived_ontology"]["colors"],
            "evidence": "Description discusses `Red tea`, `pale green or white-glazed bowls`, and `dark-glazed teabowls` as general types; medium is `White stoneware with ivory-white glaze`.",
            "expected": "white from this object's explicit ivory-white glaze; red is not this object's color; green/black refer to comparand types.",
            "status": "color attribution contamination in free prose",
        },
        "cma_126531": {
            "observed": by_label["cma_126531"]["derived_ontology"],
            "evidence": "Rank-badge description is generic; `landscape`, `cloud`, `bird`, and `dark backgrounds` describe badges in general.",
            "expected": "Do not promote generic description to per-item motif/color without item-level support.",
            "status": "description is cohort-level, not necessarily this item's observed decoration",
        },
        "cma_148523": {
            "observed": by_label["cma_148523"]["derived_ontology"],
            "evidence": "Title `Uncut Robe`; raw `type` is `Garment`; builder object type includes textile because it scans title and type, while description names an item that would be sewn into a gown.",
            "expected": ["textile"],
            "status": "textile title/type correctly maps broad textile; garment status comes from raw class",
        },
        "cma_111783": {
            "observed": by_label["cma_111783"]["derived_ontology"]["colors"],
            "evidence": "Medium says `Northern Celadon ware`; color label emits green through a celadon alias.",
            "expected": "Record celadon as a named glaze/ware; only map to green as a coarse catalogue color if that synonym policy is explicit.",
            "status": "fine color lost through synonym mapping",
        },
        "cma_119670": {
            "observed": by_label["cma_119670"]["derived_ontology"]["colors"],
            "evidence": "Medium says `Northern Black ware`; color label emits black from a ware-type name.",
            "expected": "Keep ware name and visible palette/coating separate.",
            "status": "ware-name color vs object color ambiguity",
        },
        "cma_121889": {
            "observed": by_label["cma_121889"]["derived_ontology"],
            "evidence": "Title says `Box with Ink Cakes`; medium describes box plus multicolour molded ink; component titles include separate green/red/etc. ink cakes.",
            "expected": "Separate box, ink cakes, and their colors; parent color union must not imply all colors are visible on the box image.",
            "status": "split-group component/color ambiguity",
        },
        "cma_380090": {
            "observed": by_label["cma_380090"]["derived_ontology"]["motifs"],
            "evidence": "Description mentions `dragon kilns` while discussing firing technologies, not a visual dragon motif.",
            "expected": "No dragon motif from this phrase.",
            "status": "motif false positive from technique/place wording",
        },
        "cma_117940": {
            "observed": by_label["cma_117940"]["derived_ontology"]["motifs"],
            "evidence": "`unique among other excavated drum stands ... typically show two birds standing on tigers`; this sentence discusses other objects.",
            "expected": "Do not infer bird/tiger for this item from this comparison.",
            "status": "historical-object attribution risk",
        },
        "cma_154392": {
            "observed": by_label["cma_154392"]["derived_ontology"],
            "evidence": "Raw type is `Embroidery`; description says `The painting depicts ...`; material is opaque watercolor and gold/silver on silk.",
            "expected": "At least painting and textile/embroidery support; no broad textile-only collapse.",
            "status": "mixed media collapses to textile",
        },
    }
    raw_material_mismatches = []
    for item in items:
        if item["source"] != "cleveland":
            continue
        raw = raw_for(item)
        tech = plain(raw.get("technique"))
        label = by_label[item["item_id"]]["derived_ontology"]
        if tech and not label["material"]:
            raw_material_mismatches.append({"item_id": item["item_id"], "technique": tech, "labels": []})
    report = {
        "protocol": "cv-v4-catalogue-audit-v1",
        "mode": "read-only",
        "builder": str(BUILDER_PATH.relative_to(ROOT)).replace("\\", "/"),
        "builder_sha256": digest(BUILDER_PATH),
        "catalog": str(catalog_path.relative_to(ROOT)).replace("\\", "/"),
        "catalog_sha256": digest(catalog_path),
        "rows": len(items), "unique_items_with_findings": unique_item_count,
        "finding_instances": len(issues), "issue_counts": issue_counts,
        "severity_counts": severity_counts,
        "raw_schema": {
            "met": {"structured_type": "objectName", "classification": "classification", "medium": "medium", "era": "period/dynasty/reign/objectDate", "description": "absent"},
            "cleveland": {"structured_type": "type", "classification": "collection", "medium_or_technique": "technique", "era": "culture/creation_date/date_text", "description": "description/did_you_know"},
        },
        "focused_conclusions": [
            {"topic": "builder source mapping", "conclusion": "The builder succeeds mechanically but CMA type/collection/technique fields are assigned across incompatible semantic roles."},
            {"topic": "technique vs material", "conclusion": "For CMA the builder uses only `technique` as medium and never `support_materials`; technique verbs therefore enter the material domain."},
            {"topic": "era parsing", "conclusion": "The dedicated era field is separated from prose, so no description-only era false positives were found; issue risk is unresolved reign/date overlap and overly broad token matching in dedicated fields."},
            {"topic": "colors", "conclusion": "Colors are catalogue-string mentions, not structured or image-observed colors; patterns have boundary and technique/title-style risks."},
            {"topic": "motifs", "conclusion": "Tags and title are the main motif sources, but optional curatorial prose can name historical comparisons or other works; prose requires subject-level review."},
            {"topic": "split grouping", "conclusion": "The artifact input itself has no cross-split group after v2 freezing; this validates isolation for the input but does not validate that the group construction semantically separates distinct objects and images."},
        ],
        "pattern_probes": pattern_probes,
        "split_groups": {"group_count": len(groups), "non_singleton_groups": sum(len(v) > 1 for v in groups.values()),
                         "multi_item_group_count": sum(len(v) > 1 for v in groups.values()),
                         "multi_item_group_rows": sum(len(v) for v in groups.values() if len(v) > 1),
                         "groups_with_distinct_raw_types": len(mixed_groups),
                         "cross_split_group_count": len(split_crossings), "cross_split_groups": split_crossings,
                         "groups_with_distinct_raw_types_examples": mixed_groups[:12]},
        "known_examples": known_examples,
        "cleveland_technique_without_extracted_material_count": len(raw_material_mismatches),
        "cleveland_technique_without_extracted_material_examples": raw_material_mismatches[:40],
        "evidence_file": str((output / "catalogue_issues.jsonl").relative_to(ROOT)).replace("\\", "/"),
        "limitations": ["A heuristic finding is a review item, not proof that the catalogue statement is false.",
                        "No image pixels were inspected for this catalogue-only audit; image evidence would be a separate supplement.",
                        "The audit reads the frozen snapshot and raw museum records only."],
    }
    (output / "catalogue_issues.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in issues), encoding="utf-8")
    (output / "enriched_labels.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.txt").write_text(
        "Generated by scripts/audit_cv_v4_catalogue.py.\n"
        "report.json: aggregate findings and conclusions.\n"
        "catalogue_issues.jsonl: item-level evidence and deterministic recommendations.\n"
        "enriched_labels.jsonl: in-memory builder output joined to raw field excerpts.\n"
        "baseline_rebuild/: direct, unchanged builder output for reproducibility.\n",
        encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(items), "finding_instances": len(issues),
                      "unique_items_with_findings": unique_item_count, "issue_counts": issue_counts}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
