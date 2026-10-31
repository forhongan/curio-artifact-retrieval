"""Build a separate, rights-checked official-museum reference bank.

Never writes raw ingestion records or the frozen gallery. All network responses,
accepted references and explicit rejection decisions live in data/references.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sys
from urllib.parse import urlencode

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from curio.retrieve.references import (  # noqa: E402
    GalleryExclusions, RULES, image_fingerprints, normalize_record,
)


PATTERNS = {
    "blue_white": r"blue[ -]and[ -]white|underglaze blue|cobalt.{0,40}(blue|decoration)",
    "celadon": r"celadon|longquan|yue ware|green.glaze",
    "meiping": r"meiping|plum vase",
    "gourd": r"gourd",
    "bronze": r"bronze|copper alloy",
    "jade": r"jade|nephrite",
    "lacquer": r"lacquer|cinnabar",
    "textile": r"silk|textile|embroidery|embroidered|tapestry",
    "porcelain": r"porcelain",
    "buddhist": r"buddha|bodhisattva",
    "painting": r"painting|ink on (paper|silk)|hanging scroll|handscroll",
    "calligraphy": r"calligraphy|calligrapher",
    "book": r"woodblock.printed|printed book|album of|book of",
}


def categories(record: dict, source: str) -> list[str]:
    # A painting's prose may mention a jade ornament or a bronze inscription.
    # Material categories must describe the object itself, not its subject.
    medium = str(record.get("technique" if source == "cleveland" else "medium", "")).casefold()
    title = str(record.get("title", "")).casefold()
    kind = str(record.get("type" if source == "cleveland" else "classification", "")).casefold()
    actual = title + " " + medium + " " + kind
    cats = []
    for key, pattern in PATTERNS.items():
        text = medium if key in {"bronze", "jade", "lacquer", "textile", "porcelain"} else actual
        if re.search(pattern, text):
            cats.append(key)
    # Blue/white and celadon evidence may be catalogued in the description,
    # but only on an actual ceramic object.
    if re.search(r"porcelain|stoneware|earthenware|ceramic", medium + " " + kind):
        for key in ("blue_white", "celadon"):
            if key not in cats and re.search(PATTERNS[key], str(record.get("description", "")).casefold()):
                cats.append(key)
    if "painting" in cats:
        # Silk support and lacquered rollers are not standalone textile or
        # lacquer artifacts for visual-reference selection.
        cats = [c for c in cats if c not in {"textile", "lacquer", "jade", "bronze"}]
    if "calligraphy" in cats:
        # A manual's title can name painting and calligraphy without describing
        # the visible page as calligraphy. Require explicit page/work evidence.
        explicit = bool(re.search(r"calligraphy in|calligraphy\s*(?:\)|$)|seal script|running.{0,15}script|standard script|cursive script", title))
        if re.search(r"manual|handbook|studio collection|page from", title):
            explicit = False
        if not explicit and "calligraphy" not in kind:
            cats.remove("calligraphy")
        elif "painting" in cats:
            cats.remove("painting")
    return cats


def chinese(record: dict) -> bool:
    return bool(re.search(r"chin(a|ese)", str(record.get("culture", "")) + " " + str(record.get("country", "")), re.I))


class Builder:
    def __init__(self, root: Path, offline=False, output: Path | None = None):
        self.root, self.offline = root, offline
        self.out = output or root / "data/references"
        for name in ("images", "metadata", "http_cache"):
            (self.out / name).mkdir(parents=True, exist_ok=True)
        self.ledger, self.bank, self.failures = [], [], []
        self.http = requests.Session()
        self.http.headers["Accept"] = "application/json,image/*,*/*"
        self.queried = []
        self.seen = set()
        self.counts = Counter()
        self.gallery_path = root / "data/snapshots/collection_v2.jsonl"
        rows = [json.loads(line) for line in self.gallery_path.read_text(encoding="utf-8").splitlines()]
        frozen_rows = list(rows)
        expansion_path = root / "data/snapshots/expansion_v3.jsonl"
        expansion_rows = ([json.loads(line) for line in expansion_path.read_text(encoding="utf-8").splitlines()]
                          if expansion_path.exists() else [])
        rows.extend(expansion_rows)
        raw = {}
        for row in rows:
            path = (root / row["metadata_path"] if row.get("metadata_path") else
                    root / "data/raw" / row["source"] / f"{row['source_id']}.json")
            if path.exists():
                raw[row["item_id"]] = json.loads(path.read_bytes())
        self.exclusions = GalleryExclusions(rows, raw)
        fingerprints = self.out / "gallery_fingerprints.json"
        gallery_digest = hashlib.sha256(self.gallery_path.read_bytes()).hexdigest()
        saved = json.loads(fingerprints.read_text()) if fingerprints.exists() else {}
        if saved.get("gallery_sha256") != gallery_digest:
            saved = {"gallery_sha256": gallery_digest, "phash64": {}}
            for row in frozen_rows:
                path = root / row["image_path"]
                if path.exists():
                    with Image.open(path) as image:
                        saved["phash64"][row["item_id"]] = image_fingerprints(image)["phash64"]
            fingerprints.write_text(json.dumps(saved), encoding="utf-8")
        for item_id, value in saved["phash64"].items():
            self.exclusions.add_gallery_phash(item_id, value)
        for row in expansion_rows:
            if row.get("phash64"):
                self.exclusions.add_gallery_phash(row["item_id"], row["phash64"])
        self.gallery_digest = gallery_digest
        self.expansion_digest = hashlib.sha256(expansion_path.read_bytes()).hexdigest() if expansion_path.exists() else None

    def get_json(self, url: str) -> dict:
        key = hashlib.sha256(url.encode()).hexdigest()
        path = self.out / "http_cache" / f"{key}.json"
        if path.exists():
            return json.loads(path.read_bytes())["response"]
        if self.offline:
            raise ValueError("offline cache miss")
        response = self.http.get(url, timeout=(8, 35))
        response.raise_for_status()
        data = response.json()
        path.write_text(json.dumps({"request_url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                                    "response": data}, ensure_ascii=False), encoding="utf-8")
        return data

    def attempt(self, source: str, record: dict, terms: list[str], per_category: int):
        rid = f"{'cma' if source == 'cleveland' else 'met'}_{record.get('id' if source == 'cleveland' else 'objectID')}"
        if rid in self.seen:
            return
        self.seen.add(rid)
        cats = categories(record, source)
        if not cats or not chinese(record) or all(self.counts[c] >= per_category for c in cats):
            return
        try:
            row = normalize_record(source, record)
            exclusion = self.exclusions.reason(row)
            if exclusion:
                self.ledger.append({"reference_id": rid, "categories": cats, **exclusion})
                return
            # Every accepted candidate has an official API response cached here,
            # even when discovery came from an older, read-only raw snapshot.
            url = (f"https://openaccess-api.clevelandart.org/api/artworks/{row['source_id']}"
                   if source == "cleveland" else f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{row['source_id']}")
            live = self.get_json(url)
            record = live.get("data", live)
            row = normalize_record(source, record)
            cats = categories(record, source)
            if not cats or not chinese(record):
                raise ValueError("official metadata no longer confirms Chinese category")
            exclusion = self.exclusions.reason(row)
            if exclusion:
                self.ledger.append({"reference_id": rid, **exclusion})
                return
            image_path = self.out / "images" / f"{rid}.jpg"
            if image_path.exists():
                binary = image_path.read_bytes()
            else:
                if self.offline:
                    raise ValueError("offline image cache miss")
                response = self.http.get(row["image_url"], timeout=(8, 45))
                response.raise_for_status()
                binary = response.content
            with Image.open(io.BytesIO(binary)) as image:
                image = image.convert("RGB")
                if min(image.size) < 256 or max(image.size) < 512:
                    raise ValueError(f"image below quality floor: {image.size}")
                original_hash = hashlib.sha256(binary).hexdigest()
                row.update(image_fingerprints(image))
                row["original_image_sha256"] = original_hash
                exclusion = self.exclusions.reason(row)
                if exclusion:
                    self.ledger.append({"reference_id": rid, "categories": cats, **exclusion})
                    return
                # Preserve fetched image bytes so offline rebuilds retain SHA.
                image_path.write_bytes(binary)
                row.update(width=image.width, height=image.height)
            meta_path = self.out / "metadata" / f"{rid}.json"
            meta_path.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            row.update(image_path=image_path.relative_to(self.root).as_posix(), image_sha256=original_hash,
                       metadata_path=meta_path.relative_to(self.root).as_posix(),
                       metadata_api_url=url, categories=cats, query_terms=terms,
                       collection_query="; ".join(terms),
                       verified_at=datetime.now(timezone.utc).isoformat(),
                       exclusion_policy="whole_gallery_ids_accession_families_related_sha_ahash256_le4_phash64_le6",
                       gallery_sha256=self.gallery_digest)
            # Exclude duplicates among accepted references as well.
            for other in self.bank:
                if (row["image_sha256"] == other["image_sha256"] or
                    (int(row["phash64"], 16) ^ int(other["phash64"], 16)).bit_count() <= 6):
                    self.ledger.append({"reference_id": rid, "reason": "duplicate_reference_image", "reference_of": other["reference_id"]})
                    return
            self.bank.append(row)
            self.counts.update(cats)
            print(f"accepted {rid}: {row['title']} {cats}", flush=True)
        except Exception as exc:
            self.failures.append({"reference_id": rid, "source": source, "error": str(exc)})

    def run(self, per_category=5, online=True):
        # Existing raw metadata is only discovery; never modified. Prefer rare
        # forms before generic porcelain/textiles fill category caps.
        cached = []
        for source in ("cleveland", "met"):
            for path in sorted((self.root / "data/raw" / source).glob("*.json")):
                record = json.loads(path.read_bytes())
                cats = categories(record, source)
                if cats and chinese(record):
                    cached.append((source, record, cats))
        cached.sort(key=lambda entry: (not bool(set(entry[2]) & {"meiping", "gourd", "celadon", "lacquer", "calligraphy", "book"}), str(entry[1].get("id", entry[1].get("objectID")))))
        for source, record, cats in cached:
            self.attempt(source, record, [RULES[c][1][0] if c in RULES else f"Chinese {c}" for c in cats], per_category)
        if online:
            for category, pattern in PATTERNS.items():
                if self.counts[category] >= per_category:
                    continue
                term = RULES[category][1][0] if category in RULES else f"Chinese {category}"
                for source in ("cleveland", "met"):
                    if self.counts[category] >= per_category:
                        break
                    base = "https://openaccess-api.clevelandart.org/api/artworks/?" if source == "cleveland" else "https://collectionapi.metmuseum.org/public/collection/v1.1/search?"
                    params = ({"q": term, "cc0": 1, "has_image": 1, "culture": "China", "limit": 50}
                              if source == "cleveland" else {"q": term, "hasImages": "true", "geoLocation": "China", "limit": 50, "offset": 0})
                    url = base + urlencode(params)
                    try:
                        result = self.get_json(url)
                        self.queried.append({"source": source, "category": category, "query": term, "url": url, "total": result.get("total", result.get("info", {}).get("total"))})
                        records = result.get("data", []) if source == "cleveland" else []
                        if source == "met":
                            for sid in (result.get("objectIDs") or [])[:50]:
                                if f"met_{sid}" in self.seen:
                                    continue
                                try:
                                    records.append(self.get_json(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{sid}"))
                                except Exception as exc:
                                    self.failures.append({"reference_id": f"met_{sid}", "error": str(exc)})
                        for record in records:
                            self.attempt(source, record, [term], per_category)
                    except Exception as exc:
                        self.failures.append({"query_url": url, "error": str(exc)})
        self.save(per_category)

    def save(self, target):
        def jsonl(name, rows):
            (self.out / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        jsonl("bank.jsonl", self.bank)
        jsonl("exclusions.jsonl", self.ledger)
        jsonl("failures.jsonl", self.failures)
        report = {"gallery_sha256": self.gallery_digest, "reference_count": len(self.bank),
                  "expansion_sha256": self.expansion_digest,
                  "bank_sha256": hashlib.sha256((self.out / "bank.jsonl").read_bytes()).hexdigest(),
                  "category_counts": dict(self.counts), "target_per_category": target,
                  "gaps": [c for c in PATTERNS if not self.counts[c]],
                  "below_target": [c for c in PATTERNS if self.counts[c] < target],
                  "source_counts": dict(Counter(r["source"] for r in self.bank)),
                  "exclusion_counts": dict(Counter(r["reason"] for r in self.ledger)),
                  "failure_count": len(self.failures), "queries": self.queried,
                  "role": "candidate extension and retrieval references only; frozen production gallery unchanged",
                  "metadata_policy": "official API snapshots; CC0 or Met isPublicDomain=true; Chinese culture; image quality floor",
                  "perceptual_policy": "ahash256 Hamming <=4 or phash64 Hamming <=6 is rejected conservatively; this is not proof of no semantic identity"}
        (self.out / "coverage.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        jsonl("extension_candidates.jsonl", [r for r in self.bank if set(r["categories"]) & {"painting", "calligraphy", "book", "bronze", "jade", "buddhist"}])
        print(json.dumps({"reference_count": len(self.bank), "categories": dict(self.counts), "failures": len(self.failures)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--per-category", type=int, default=5)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--cached-discovery-only", action="store_true")
    parser.add_argument("--output", type=Path, help="Separate acquisition output directory; defaults to data/references")
    args = parser.parse_args()
    Builder(args.root, args.offline, output=args.output).run(args.per_category, not args.cached_discovery_only)


if __name__ == "__main__":
    main()
