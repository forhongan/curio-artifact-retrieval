"""Build the catalogue-first CV v4 evaluation labels without overwriting v2.

The raw museum records are the primary source.  This script parses explicit
catalogue fields into reviewable domains and retains provenance.  Values that
cannot be determined stay null; AI supplements are stored separately.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/cv_v4_20261108/official"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(text(x) for x in value if text(x))
    if isinstance(value, dict):
        return " ".join(text(x) for x in value.values() if text(x))
    return re.sub(r"<[^>]+>", " ", str(value)).strip()


def raw_record(item: dict) -> dict:
    path = ROOT / "data/raw" / item["source"] / f"{item['source_id']}.json"
    return json.loads(path.read_text(encoding="utf-8"))


TYPE_PATTERNS = (
    ("vase", r"\b(?:vases?|meiping|bottle vase)\b"),
    ("bottle", r"\bbottles?\b|\bsnuff bottle\b"),
    ("jar", r"\bjars?\b"),
    ("bowl", r"\bbowls?\b"),
    ("dish", r"\b(?:dishes|dish)\b"),
    ("plate", r"\b(?:plates?|platters?)\b"),
    ("cup", r"\bcu(?:p|ps)\b"),
    ("teapot", r"\bteapots?\b"),
    ("ewer", r"\bewers?\b"),
    ("jug", r"\bjug\b"),
    ("flask", r"\bflasks?\b"),
    ("censer", r"\b(?:censer|incense burner|incense vessel)\b"),
    ("figure", r"\b(?:figures?|figurines?|sculptures?|statues?|buddha|bodhisattva)\b"),
    ("mirror", r"\bmirrors?\b"),
    ("box", r"\bbox(?:es)?\b"),
    ("textile", r"\b(?:textiles?|garments?|robes?|quilts?|fabric)\b"),
    ("buckle", r"\bbuckles?\b"),
    ("stopper", r"\bstoppers?\b"),
    ("pendant", r"\bpendants?\b"),
    ("ornament", r"\b(?:ornaments?|jewelry|jewellery)\b"),
    ("tile", r"\btiles?\b"),
    ("fragment", r"\bfragments?\b"),
    ("rubbing", r"\brubbings?\b"),
    ("painting", r"\bpaintings?\b|\bink on (?:paper|silk)\b|\bhanging scroll\b"),
    ("book", r"\bbooks?\b|\balbums?\b|\bwoodblock printed\b"),
)


MATERIAL_PATTERNS = (
    ("porcelain", r"\bporcelain\b"),
    ("earthenware", r"\b(?:earthenware|pottery)\b"),
    ("stoneware", r"\bstoneware\b"),
    ("ceramic", r"\b(?:ceramic|glaze[ds]?|glazed)\b"),
    ("bronze", r"\b(?:bronze|copper alloy)\b"),
    ("jade", r"\b(?:jade|nephrite)\b"),
    ("glass", r"\bglass\b"),
    ("lacquer", r"\blacquer(?:ed)?\b"),
    ("silk", r"\bsilk\b"),
    ("textile", r"\b(?:textile|cotton|wool|satin|brocade|embroid(?:ery|ered)|woven|weave)\b"),
    ("gold", r"\b(?:gold|gilt)\b"),
    ("silver", r"\bsilver\b"),
    ("copper", r"\bcopper\b"),
    ("iron", r"\biron\b"),
    ("wood", r"\bwood\b"),
    ("stone", r"\b(?:stone|marble|agate|nephrite|jade)\b"),
    ("ink", r"\bink\b"),
    ("paper", r"\bpaper\b"),
)


MOTIF_PATTERNS = (
    ("dragon", r"\bdragons?\b"),
    ("lotus", r"\blotus(?:es)?\b"),
    ("floral", r"\b(?:flowers?|floral|blossoms?|peony|peonies|plum blossoms?|prunus|chrysanthemum|orchid)\b"),
    ("bird", r"\bbirds?\b"),
    ("phoenix", r"\bphoenix\b"),
    ("fish", r"\bfish(?:es)?\b"),
    ("horse", r"\bhorses?\b"),
    ("tiger", r"\btigers?\b"),
    ("lion", r"\blions?\b"),
    ("butterfly", r"\bbutterfl(?:y|ies)\b"),
    ("cloud", r"\bclouds?\b"),
    ("landscape", r"\blandscape\b"),
    ("figures", r"\b(?:figures?|buddhist|buddha|bodhisattva)\b"),
    ("geometric", r"\b(?:geometric|scrollwork|diapers?|lozenges?)\b"),
)


COLOR_PATTERNS = (
    ("blue", r"\b(?:blue|cobalt)\b"),
    ("white", r"\bwhite\b"),
    ("green", r"\b(?:green|celadon)\b"),
    ("red", r"\b(?:red|cinnabar|rouge)"),
    ("black", r"\b(?:black|dark)"),
    ("yellow", r"\byellow\b"),
    ("brown", r"\bbrown\b"),
    ("gray", r"\b(?:gray|grey)"),
    ("purple", r"\bpurple\b"),
    ("pink", r"\bpink\b"),
)


ERAS = (
    ("neolithic", r"neolithic|yangshao|longshan|liangzhu"),
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


def find(text_value: str, patterns) -> list[str]:
    return [name for name, pattern in patterns if re.search(pattern, text_value, re.I)]


def build_one(item: dict) -> dict:
    raw = raw_record(item)
    if item["source"] == "met":
        title = text(raw.get("title"))
        identity = " ".join(text(raw.get(k)) for k in ("title", "objectName", "classification"))
        medium = text(raw.get("medium"))
        era_text = ". ".join(text(raw.get(k)) for k in ("period", "dynasty", "reign", "objectDate"))
        tags = [text(x.get("term") if isinstance(x, dict) else x) for x in (raw.get("tags") or [])]
        description = ""
        source_fields = {"title": "title", "object_name": "objectName", "classification": "classification", "medium": "medium", "period": "period", "dynasty": "dynasty", "reign": "reign", "date": "objectDate", "description": None, "tags": "tags"}
    else:
        title = text(raw.get("title"))
        identity = " ".join(text(raw.get(k)) for k in ("title", "type", "collection"))
        medium = text(raw.get("technique") or raw.get("medium"))
        era_text = ". ".join(text(raw.get(k)) for k in ("culture", "creation_date", "date_text"))
        tags = [text(x) for x in (raw.get("artists_tags") or [])]
        description = " ".join(text(raw.get(k)) for k in ("description", "did_you_know", "early_education_description"))
        source_fields = {"title": "title", "type": "type", "medium": "technique", "culture": "culture", "date": "creation_date", "description": "description/did_you_know", "tags": "artists_tags"}
    type_evidence = identity + " " + title
    material_evidence = medium
    motif_evidence = " ".join([title, *tags, description])
    color_evidence = " ".join([title, medium, description])
    era_evidence = era_text
    broad_type = find(type_evidence, TYPE_PATTERNS)
    # A title for a "stopper" of a snuff bottle is a component, not a bottle.
    if re.search(r"\bstoppers?\b", title, re.I):
        broad_type = [value for value in broad_type if value != "bottle"]
        if "stopper" not in broad_type:
            broad_type.append("stopper")
    material = find(material_evidence, MATERIAL_PATTERNS)
    motifs = find(motif_evidence, MOTIF_PATTERNS)
    colors = find(color_evidence, COLOR_PATTERNS)
    era = find(era_evidence, ERAS)
    if any(x in material for x in ("porcelain", "earthenware", "stoneware", "ceramic")):
        broad_material = "ceramic"
    elif any(x in material for x in ("silk", "textile")):
        broad_material = "textile"
    elif any(x in material for x in ("bronze", "copper", "iron", "gold", "silver")):
        broad_material = "metal"
    elif any(x in material for x in ("jade", "stone")):
        broad_material = "stone"
    else:
        broad_material = material[0] if material else ""
    # A catalogue title can state a material; medium has priority, then title.
    if not material:
        material = find(title + " " + description, MATERIAL_PATTERNS)
    return {
        "item_id": item["item_id"],
        "source": item["source"],
        "split": item["split"],
        "object_group": item["object_group"],
        "image_path": item["image_path"],
        "image_sha256": item["image_sha256"],
        "official_catalog": {
            "title": title,
            "object_name": text(raw.get("objectName") if item["source"] == "met" else raw.get("type")),
            "classification": text(raw.get("classification") if item["source"] == "met" else raw.get("collection")),
            "medium": medium,
            "era": era_text,
            "tags": tags,
            "description": description,
            "source_url": item["source_url"],
            "field_provenance": source_fields,
        },
        "derived_ontology": {
            "object_type": broad_type,
            "material": material,
            "broad_material": broad_material,
            "motifs": motifs,
            "colors": colors,
            "era": era,
            "label_presence": {
                "object_type": bool(broad_type),
                "material": bool(material),
                "motifs_checked": bool(motifs or description or tags),
                "colors_checked": bool(colors or title and medium),
                "era_checked": bool(era_text),
            },
        },
        "ai_visible_supplement": None,
        "label_source": "official_catalog_derived",
    }


def main(output: Path = DEFAULT_OUTPUT) -> None:
    catalog_path = ROOT / "data/snapshots/collection_v2.jsonl"
    rows = read_jsonl(catalog_path)
    output.mkdir(parents=True, exist_ok=True)
    result = [build_one(row) for row in rows]
    (output / "labels.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in result), encoding="utf-8"
    )
    frequencies = {}
    for slot in ("object_type", "material", "broad_material", "motifs", "colors", "era"):
        if slot == "broad_material":
            values = [row["derived_ontology"][slot] for row in result]
        else:
            values = [value for row in result for value in row["derived_ontology"][slot]]
        frequencies[slot] = Counter(values).most_common()
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "catalog_sha256": sha(catalog_path),
        "labels_sha256": sha(output / "labels.jsonl"),
        "items": len(result),
        "splits": dict(Counter(row["split"] for row in result)),
        "coverage": {
            slot: sum(bool(row["derived_ontology"][slot]) for row in result)
            for slot in ("object_type", "material", "broad_material", "motifs", "colors", "era")
        },
        "frequencies": frequencies,
        "warning": "Official catalogue-derived labels. A colour or motif mention is not an independent image annotation. Descriptions can discuss other objects; ambiguous values require audit.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    main(parser.parse_args().output)
