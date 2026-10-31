"""Auditable museum references, with a bilingual rule planner (not an LLM).

References are deliberately separate from the frozen retrieval gallery.  The
builder uses official museum records, checks reuse rights and excludes gallery
objects, their recorded families, and exact/visually similar images.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlparse

from PIL import Image, ImageOps
import imagehash


# Each alias is a literal bilingual rule. English phrases are museum search
# vocabulary, not a generated translation or a claim of semantic understanding.
RULES = {
    "underglaze_polychrome": (("青花五彩", "五彩", "青花加彩", "blue and green", "blue-green", "wucai", "polychrome"),
                             ("underglaze blue polychrome porcelain",)),
    "blue_white": (("青花", "blue and white", "blue-and-white", "blue white"),
                   ("blue and white porcelain", "underglaze blue porcelain")),
    "celadon": (("青瓷", "龙泉", "龍泉", "粉青", "梅子青", "celadon", "longquan"),
                ("celadon", "Longquan ware")),
    "meiping": (("梅瓶", "meiping"), ("meiping", "plum vase")),
    "gourd": (("葫芦", "葫蘆", "gourd", "double gourd"), ("double gourd vase", "gourd vase")),
    "bronze": (("青铜", "青銅", "铜器", "銅器", "bronze", "饕餮", "鼎", "觚", "觯", "尊"),
               ("Chinese ritual bronze", "bronze ritual vessel")),
    "jade": (("玉", "jade", "nephrite"), ("Chinese jade", "jade carving")),
    "lacquer": (("漆", "lacquer", "cinnabar"), ("Chinese carved lacquer", "Chinese lacquer")),
    "textile": (("织物", "織物", "纺织", "絲", "丝", "刺绣", "刺繡", "缂丝", "袍", "textile", "silk", "embroidery"),
                ("Chinese silk textile", "Chinese embroidery")),
    "porcelain": (("瓷", "porcelain", "ceramic"), ("Chinese porcelain",)),
    "buddhist": (("佛", "菩萨", "菩薩", "buddha", "bodhisattva"), ("Chinese Buddhist sculpture",)),
    "painting": (("绘画", "繪畫", "山水", "画卷", "畫", "painting", "landscape"), ("Chinese landscape painting",)),
}

ERA_RULES = {
    "Wanli": ("万历", "萬曆", "wanli"), "Kangxi": ("康熙", "kangxi"),
    "Yongzheng": ("雍正", "yongzheng"), "Qianlong": ("乾隆", "qianlong"),
    "Xuande": ("宣德", "xuande"), "Yongle": ("永乐", "永樂", "yongle"),
    "Jiajing": ("嘉靖", "jiajing"), "Zhengde": ("正德", "zhengde"),
    "Chongzhen": ("崇祯", "崇禎", "chongzhen"),
    "Ming": ("明代", "明朝", "ming dynasty"), "Qing": ("清代", "清朝", "qing dynasty"),
    "Song": ("宋代", "宋朝", "song dynasty"), "Yuan": ("元代", "元朝", "yuan dynasty"),
}

FORM_RULES = {
    "vase": (("瓶", "vase", "bottle", "meiping", "flask"), r"\bvase\b|\bbottle\b|meiping|\bflask\b"),
    "bowl": (("碗", "盌", "bowl"), r"\bbowl\b"),
    "dish": (("盘", "盤", "碟", "dish", "plate"), r"\bdish\b|\bplate\b|\bsaucer\b"),
    "jar": (("罐", "jar"), r"\bjar\b"),
    "box": (("盒", "箱", "box", "casket"), r"\bbox\b|\bcasket\b"),
    "tripod": (("鼎", "tripod", "ding"), r"\bding\b|\btripod\b|\bcauldron\b"),
    "ritual": (("礼器", "禮器", "ritual"), r"\britual\b|\b(jue|ding|gui|gu|you|zun)\b|wine vessel"),
    "robe": (("袍", "robe"), r"\brobe\b"),
}


def plan_reference_query(query: str) -> dict:
    lowered = query.casefold().strip()
    matches = [key for key, (aliases, _) in RULES.items() if any(a in lowered for a in aliases)]
    if "underglaze_polychrome" in matches and "blue_white" in matches:
        matches.remove("blue_white")
    # A broad ceramic alias must not drown out the requested glaze/form.
    if "porcelain" in matches and any(k in matches for k in ("blue_white", "celadon", "meiping", "gourd")):
        matches.remove("porcelain")
    terms = list(dict.fromkeys(term for key in matches for term in RULES[key][1]))
    if not terms:
        # English queries can go straight to museum search; unknown Chinese is
        # explicit and abstains rather than pretending to translate it.
        terms = [query.strip()] if query.strip() and not re.search(r"[\u3400-\u9fff]", query) else []
    eras = [era for era, aliases in ERA_RULES.items() if any(a in lowered for a in aliases)]
    forms = [key for key, (aliases, _) in FORM_RULES.items() if any(
        a in lowered if re.search(r"[\u3400-\u9fff]", a) else bool(re.search(r"\b" + re.escape(a) + r"\b", lowered))
        for a in aliases)]
    form_terms = {"vase": "vase", "bowl": "bowl", "dish": "dish", "jar": "jar",
                  "box": "box", "tripod": "ding tripod vessel", "ritual": "ritual vessel", "robe": "robe"}
    for form in forms:
        terms = [term if re.search(FORM_RULES[form][1], term, re.I) else f"{term} {form_terms[form]}"
                 for term in terms]
    if eras:
        terms = [f"{era} {term}" for era in eras for term in (terms or ["Chinese art"])]
    return {
        "planner": "bilingual_rules_v1", "is_llm": False, "input_query": query,
        "category_keys": matches, "query_terms": terms, "required_eras": eras, "required_forms": forms,
        "museum_queries": [{"source": source, "query": term} for term in terms
                           for source in ("cleveland", "met")],
        "explanation": "Literal bilingual museum-vocabulary rules; no language model was called.",
        "unmapped": not bool(terms),
    }


def load_reference_bank(root: Path | str) -> list[dict]:
    root = Path(root)
    path = root / "bank.jsonl" if root.name == "references" else root / "data/references/bank.jsonl"
    sources = [path, path.with_name("supplemental.jsonl")]
    rows = []
    seen = set()
    for source in sources:
        if source.exists():
            for line in source.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["reference_id"] not in seen:
                    rows.append(row)
                    seen.add(row["reference_id"])
    return rows


def select_references(query: str, root: Path | str, limit: int = 4) -> dict:
    plan = plan_reference_query(query)
    wanted = set(plan["category_keys"])
    scored = []
    for row in load_reference_bank(root):
        if not (Path(root) / row["image_path"]).is_file():
            continue
        overlap = wanted.intersection(row.get("categories", []))
        if wanted and not wanted.issubset(row.get("categories", [])):
            continue
        attribution = " ".join(str(row.get(k, "")) for k in ("date", "culture", "period", "reign")).casefold()
        if any(era.casefold() not in attribution for era in plan["required_eras"]):
            continue
        object_identity = " ".join(str(row.get(k, "")) for k in ("title", "object_name", "classification")).casefold()
        if any(not re.search(FORM_RULES[form][1], object_identity) for form in plan["required_forms"]):
            continue
        fragment = bool(re.search(r"\b(shard|fragment|fragmentary|sherd)\b", object_identity))
        if fragment and not re.search(r"碎|残|殘|片|shard|fragment|sherd", query, re.I):
            continue
        text = (row.get("title", "") + " " + row.get("medium", "")).casefold()
        score = len(overlap) * 10 + sum(term.casefold() in text for term in plan["query_terms"])
        if score:
            source = row.get("source", "unknown")
            family = row.get("source_family") or f"{source}:{accession_root(row.get('accession', ''))}"
            # Repeated titles in a museum accession series receive a diversity
            # group for selection, without asserting physical object identity.
            series = ".".join(row.get("accession", "").split(".")[:2])
            diversity = f"{source}:{series}:{row.get('title', '').casefold()}"
            scored.append((score, row["reference_id"], {**row, "source_family": family,
                           "selection_diversity_group": diversity, "constraint_status": "verified_metadata"}))
    scored.sort(key=lambda entry: (-entry[0], entry[1]))
    references, deferred, families = [], [], set()
    for _, _, row in scored:
        group = row["selection_diversity_group"]
        if group in families:
            deferred.append(row)
        else:
            references.append(row)
            families.add(group)
    references = (references + deferred)[:max(0, limit)]
    return {"plan": plan, "references": references,
            "constraint_status": "verified_metadata" if references else "unverified_no_reference",
            "qualifications": (["Source medium describes underglaze blue and the photograph includes green decoration; the source does not explicitly verify a wucai technique attribution."]
                               if references and "underglaze_polychrome" in wanted else [])}


def metadata_sha256(record: dict) -> str:
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def accession_root(value: str) -> str:
    return re.sub(r"(?<=\d)\.?[a-z][^0-9]*$", "", value.strip(), flags=re.I)


def official_related_ids(record: dict) -> list[str]:
    entries = []
    for field in ("related_works", "inseparable_parts"):
        value = record.get(field) or []
        if isinstance(value, dict):
            value = list(value.values())
        if isinstance(value, list):
            for entry in value:
                sid = entry.get("id") if isinstance(entry, dict) else entry
                if isinstance(sid, int) or (isinstance(sid, str) and sid.isdigit()):
                    entries.append(f"cma_{sid}")
    return list(dict.fromkeys(entries))


def normalize_record(source: str, record: dict) -> dict:
    """Accept only reuse-authorized records and official museum image hosts."""
    if source == "cleveland":
        if record.get("share_license_status") != "CC0":
            raise ValueError("Cleveland record lacks explicit CC0 rights")
        image_url = (record.get("images", {}).get("print") or {}).get("url", "")
        source_id = str(record["id"])
        result = {"title": record.get("title", ""), "medium": record.get("technique", ""),
                  "accession": record.get("accession_number", ""), "source_url": record.get("url", ""),
                  "related_ids": official_related_ids(record),
                  "culture": record.get("culture", []), "date": record.get("creation_date", ""),
                  "description": record.get("description", ""), "period": "", "reign": "",
                  "object_name": record.get("type", ""), "classification": record.get("department", "")}
        allowed = {"openaccess-cdn.clevelandart.org"}
    elif source == "met":
        if record.get("isPublicDomain") is not True:
            raise ValueError("Met record lacks explicit public-domain rights")
        image_url = record.get("primaryImageSmall") or record.get("primaryImage", "")
        source_id = str(record["objectID"])
        result = {"title": record.get("title", ""), "medium": record.get("medium", ""),
                  "accession": record.get("accessionNumber", ""), "source_url": record.get("objectURL", ""),
                  "related_ids": [], "culture": record.get("culture", ""), "date": record.get("objectDate", ""),
                  "period": record.get("period", ""), "reign": record.get("reign", ""), "description": "",
                  "object_name": record.get("objectName", ""), "classification": record.get("classification", "")}
        allowed = {"images.metmuseum.org"}
    else:
        raise ValueError("untrusted museum source")
    if urlparse(image_url).scheme != "https" or urlparse(image_url).hostname not in allowed:
        raise ValueError("missing image or untrusted image host")
    prefix = "cma" if source == "cleveland" else "met"
    return {**result, "reference_id": f"{prefix}_{source_id}", "source_id": source_id,
            "source": source, "license": "CC0", "image_url": image_url,
            "metadata_sha256": metadata_sha256(record)}


def image_fingerprints(image: Image.Image) -> dict:
    image = image.convert("RGB")
    gray = ImageOps.grayscale(image).resize((16, 16))
    values = list(gray.getdata())
    mean = sum(values) / len(values)
    return {"ahash256": "".join("1" if v >= mean else "0" for v in values),
            "phash64": str(imagehash.phash(image, hash_size=8))}


class GalleryExclusions:
    """Whole-gallery exclusion, independent of train/dev/test split."""

    def __init__(self, rows: list[dict], raw_records: dict[str, dict] | None = None):
        self.ids = {r["item_id"] for r in rows}
        self.family_ids = {r.get("object_group") for r in rows if r.get("object_group")}
        self.hashes = {r.get(key): r["item_id"] for r in rows
                       for key in ("sha256", "image_sha256", "thumb_sha256") if r.get(key)}
        self.ahashes = [(r["item_id"], int(r["phash"], 2)) for r in rows
                        if len(r.get("phash", "")) == 256]
        self.phashes: list[tuple[str, int]] = []
        self.accessions: dict[tuple[str, str], str] = {}
        self.related: dict[str, str] = {}
        for row in rows:
            raw = (raw_records or {}).get(row["item_id"], {})
            acc = raw.get("accessionNumber" if row["source"] == "met" else "accession_number", "")
            if acc:
                self.accessions[(row["source"], accession_root(acc))] = row["item_id"]
            for related in official_related_ids(raw):
                self.related[related] = row["item_id"]

    def add_gallery_phash(self, item_id: str, value: str) -> None:
        self.phashes.append((item_id, int(value, 16)))

    def reason(self, candidate: dict) -> dict | None:
        rid = candidate["reference_id"]
        for reason, hit in (("gallery_item_id", rid if rid in self.ids else None),
                            ("gallery_object_family", rid if rid in self.family_ids else None),
                            ("official_related_work", self.related.get(rid)),
                            ("official_accession_family", self.accessions.get((candidate["source"], accession_root(candidate.get("accession", "")))))):
            if hit:
                return {"reason": reason, "gallery_item_id": hit}
        for related in candidate.get("related_ids", []):
            if related in self.ids:
                return {"reason": "official_related_work", "gallery_item_id": related}
        if candidate["source"] == "cleveland" and candidate.get("accession"):
            candidate_root = accession_root(candidate["accession"])
            for (source, root), item_id in self.accessions.items():
                if source == "cleveland" and (candidate_root.startswith(root + ".") or root.startswith(candidate_root + ".")):
                    return {"reason": "official_accession_parent_record", "gallery_item_id": item_id}
        for key in ("image_sha256", "original_image_sha256"):
            if candidate.get(key) in self.hashes:
                return {"reason": "exact_image_sha256", "gallery_item_id": self.hashes[candidate[key]]}
        if candidate.get("ahash256"):
            value = int(candidate["ahash256"], 2)
            for item_id, other in self.ahashes:
                distance = (value ^ other).bit_count()
                if distance <= 4:
                    return {"reason": "perceptual_ahash256", "gallery_item_id": item_id, "distance": distance}
        if candidate.get("phash64"):
            value = int(candidate["phash64"], 16)
            for item_id, other in self.phashes:
                distance = (value ^ other).bit_count()
                if distance <= 6:
                    return {"reason": "perceptual_phash64", "gallery_item_id": item_id, "distance": distance}
        return None
