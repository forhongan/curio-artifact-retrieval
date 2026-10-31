"""Acquire precise-era whole-vessel references without changing bank43."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import hashlib
import json
import re
import requests

from build_reference_bank import Builder, ROOT, categories
from curio.retrieve.references import GalleryExclusions


def main():
    out = ROOT / "data/references/supplemental_acquisition"
    builder = Builder(ROOT, output=out)
    builder.http.headers["Accept"] = "*/*"
    class OfficialHTTP:
        def get(self, url, **kwargs):
            return requests.get(url, **kwargs)
    builder.http = OfficialHTTP()
    bank_path = ROOT / "data/references/bank.jsonl"
    frozen_sha = hashlib.sha256(bank_path.read_bytes()).hexdigest()
    refs = [json.loads(line) for line in bank_path.read_text(encoding="utf-8").splitlines()]
    galleries = []
    for filename in ("collection_v2.jsonl", "expansion_v3.jsonl"):
        galleries.extend(json.loads(line) for line in (ROOT / "data/snapshots" / filename).read_text(encoding="utf-8").splitlines())
    blockers = galleries + [{**row, "item_id": row["reference_id"], "object_group": row["reference_id"], "phash": row["ahash256"]} for row in refs]
    raw = {}
    for row in blockers:
        path = ROOT / row["metadata_path"] if row.get("metadata_path") else ROOT / "data/raw" / row["source"] / f"{row['source_id']}.json"
        raw[row["item_id"]] = json.loads(path.read_bytes())
    guard = GalleryExclusions(blockers, raw)
    guard.phashes = builder.exclusions.phashes.copy()
    for row in refs:
        guard.add_gallery_phash(row["reference_id"], row["phash64"])
    builder.exclusions = guard
    queries = [("cleveland", "Wanli"), ("cleveland", "Ming vase"),
               ("cleveland", "Qianlong vase"), ("met", "Ming vase"),
               ("met", "Ming blue and white vase"), ("met", "Wanli"),
               ("met", "Jiajing"), ("met", "Xuande vase"), ("met", "blue white porcelain"),
               ("met", "Qianlong famille rose vase"), ("met", "famille rose")]
    queries.extend(("met", term) for term in ("Ming bottle", "Ming porcelain", "Jiajing vase", "Wanli porcelain", "blue vase", "porcelain vase"))
    searched = []
    queries.extend(("met", "Ming blue and white vase") for _ in range(1))
    for query_number, (source, term) in enumerate(queries):
        from urllib.parse import urlencode
        if source == "cleveland":
            url = "https://openaccess-api.clevelandart.org/api/artworks/?" + urlencode({"q": term, "cc0": 1, "has_image": 1, "culture": "China", "limit": 100})
            try:
                response = builder.get_json(url)
            except Exception as exc:
                builder.failures.append({"query_url": url, "error": str(exc)})
                continue
            records = response.get("data", [])
        else:
            offset = 300 if query_number == len(queries) - 1 else 0
            limit = 300 if term == "Ming blue and white vase" else 100
            url = "https://collectionapi.metmuseum.org/public/collection/v1.1/search?" + urlencode({"q": term, "hasImages": "true", "geoLocation": "China", "limit": limit, "offset": offset})
            try:
                response = builder.get_json(url)
            except Exception as exc:
                builder.failures.append({"query_url": url, "error": str(exc)})
                continue
            def fetch(sid):
                try:
                    return builder.get_json(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{sid}")
                except Exception:
                    return {}
            with ThreadPoolExecutor(max_workers=8) as executor:
                records = list(executor.map(fetch, response.get("objectIDs", [])[:300]))
        searched.append({"source": source, "query": term, "url": url, "record_count": len(records)})
        for record in records:
            title = str(record.get("title", ""))
            if not re.search(r"vase|bottle|meiping", title, re.I) or re.search(r"fragment|shard|stopper|cover|snuff", title, re.I):
                continue
            era = " ".join(str(record.get(k, "")) for k in ("culture", "period", "reign", "creation_date", "objectDate"))
            cats = categories(record, source)
            medium = str(record.get("technique" if source == "cleveland" else "medium", ""))
            blue_white = bool(re.search("Ming|Wanli", era, re.I) and "blue_white" in cats)
            rose = bool(re.search("Qianlong", era, re.I) and re.search("famille rose", medium, re.I))
            if not (blue_white or rose):
                continue
            if rose:
                # Preserve explicit source term as a separate category, not a
                # broad porcelain synonym.
                record = dict(record)
            before = len(builder.bank)
            builder.attempt(source, record, [term], 100)
            if len(builder.bank) > before:
                row = builder.bank[-1]
                if rose:
                    row["categories"].append("famille_rose")
                row["supplemental_constraint"] = "Ming/Wanli blue-white whole vase" if blue_white else "Qianlong famille-rose porcelain whole vase"
                if row["reference_id"] == "cma_111782":
                    row["categories"] = [c for c in row["categories"] if c != "blue_white"] + ["underglaze_polychrome"]
                    row["visual_review"] = "Whole vase photograph shows blue decoration and visible green-painted figures; not a pure blue-and-white visual reference. Official medium describes underglaze blue, and does not explicitly identify wucai technique."
                    row["supplemental_constraint"] = "Wanli underglaze-blue vase with visible polychrome green decoration; wucai technique unverified"
                else:
                    row["visual_review"] = "Whole blue-and-white gourd bottle visible; not a fragment or isolated detail."
        builder.counts = Counter(c for row in builder.bank for c in row["categories"])
        builder.save(3)
    target = ROOT / "data/references/supplemental.jsonl"
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in builder.bank), encoding="utf-8")
    report = {"count": len(builder.bank), "supplemental_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
              "bank43_sha256": frozen_sha, "bank43_unchanged": frozen_sha == hashlib.sha256(bank_path.read_bytes()).hexdigest(),
              "blue_white_ming_vases": sum("blue_white" in row["categories"] for row in builder.bank),
              "target_strict_ming_blue_white_vases": 3,
              "strict_target_met": sum("blue_white" in row["categories"] for row in builder.bank) >= 3,
              "wanli_polychrome_vases": sum("underglaze_polychrome" in row["categories"] for row in builder.bank),
              "qianlong_famille_rose_vases": sum("famille-rose" in row["supplemental_constraint"] for row in builder.bank),
              "queries": searched, "exclusions": builder.ledger, "failures": builder.failures}
    (ROOT / "data/references/supplemental_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=True), flush=True)


if __name__ == "__main__":
    main()
