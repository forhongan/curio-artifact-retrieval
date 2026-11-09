"""Museum text search with explicit image availability and source qualification.

Only reference records are inspected here. Target gallery captions never enter
ranking. Source qualification is not a substitute for checking the picture.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
from urllib.parse import quote, urlencode

from PIL import Image, ImageOps

from curio.retrieve.dynamic_references import CONCEPT_BY_KEY, ERA_YEAR_RANGES, plan_dynamic_query
from curio.retrieve.references import image_fingerprints, normalize_record


def api_search_text(description: str) -> str:
    """Expose a small bilingual API vocabulary; never claim general translation."""
    if not re.search(r"[\u4e00-\u9fff]", description):
        return description
    plan = plan_dynamic_query(description)
    words = [r["label"] for r in plan["era_constraints"]]
    keys = set(plan["concept_keys"])
    for key in plan["concept_keys"]:
        if key == "polychrome" and "famille_rose" in keys:
            continue
        if key == "porcelain" and keys & {"blue_white", "famille_rose", "celadon"}:
            continue
        if key in {"vase", "jar", "bowl", "dish"}:
            words.append(key)
        else:
            words.append(CONCEPT_BY_KEY[key].query_phrases[0])
    return " ".join(words) if words else description


def source_support(row: dict, plan: dict) -> dict:
    """Conjunctive source evidence; a mark alone cannot establish its period."""
    text = json.dumps(
        {k: row.get(k) for k in ["title", "medium", "description", "styles"]}, ensure_ascii=False
    ).lower()
    checks = []
    if "origin" in row:
        checks.append(
            {
                "constraint": "Chinese_origin",
                "supported": bool(
                    re.search(
                        r"China|Chinese|Jingdezhen|Dehua|Longquan|中国|景德镇",
                        str(row["origin"]),
                        re.I,
                    )
                ),
            }
        )
    for key in plan["required_concepts"]:
        rule = CONCEPT_BY_KEY[key]
        value = row.get("title", "") if rule.family == "form" else text
        # Museums often put the specialist form in the description and use
        # only "Vase" or "Ewer" for objectType. Keep the object-family gate so
        # a painted gourd on a plate does not become a gourd-shaped vessel.
        if key in {'gourd', 'meiping', 'leather_flask'} and re.search(
            r'vase|bottle|flask|ewer|vessel|瓶|壶', str(row.get('title', '')), re.I
        ):
            value = text
        checks.append(
            {"constraint": key, "supported": bool(re.search(rule.evidence_pattern, value, re.I))}
        )
    for era in plan["era_constraints"]:
        bounds = ERA_YEAR_RANGES.get(era["label"])
        interval = row.get("date_interval")
        supported = bool(
            bounds and interval and bounds[0] <= interval[0] <= interval[1] <= bounds[1]
        )
        checks.append(
            {
                "constraint": era["label"],
                "supported": supported,
                "date_interval": interval,
                "required_interval": bounds,
                "policy": "production interval contained in requested period; inscriptions ignored",
            }
        )
    if plan["required_material"]:
        checks.append(
            {
                "constraint": "ceramic_body",
                "supported": bool(
                    re.search(
                        r"porcelain|stoneware|earthenware|ceramic|\bpottery\b",
                        str(row.get("medium", "")),
                        re.I,
                    )
                ),
            }
        )
    return {
        "status": "source_supported"
        if checks and all(c["supported"] for c in checks)
        else "insufficient_source_support",
        "checks": checks,
        "picture_checked": False,
        "scope": "recognized reference constraints only; source wording is not visual proof",
    }


def normalize_vam(hit: dict, detail: dict) -> dict:
    record = detail["record"]
    intervals = []
    for date in record.get("productionDates", []):
        d = date.get("date", {})
        first, last = str(d.get("earliest", ""))[:4], str(d.get("latest", ""))[:4]
        if first.isdigit() and last.isdigit():
            intervals.append((int(first), int(last)))
    sid = hit["systemNumber"]
    image_id = hit.get("_primaryImageId")
    return {
        "reference_id": "vam_" + sid,
        "source": "vam",
        "source_id": sid,
        "source_url": f"https://collections.vam.ac.uk/item/{sid}",
        "title": record.get("objectType", ""),
        "accession": record.get("accessionNumber", ""),
        "date": " / ".join(
            str(r.get("date", {}).get("text", "")) for r in record.get("productionDates", [])
        ),
        "date_interval": [min(a for a, b in intervals), max(b for a, b in intervals)]
        if intervals
        else None,
        "medium": record.get("materialsAndTechniques", ""),
        "description": " ".join(
            str(record.get(k, ""))
            for k in ["briefDescription", "physicalDescription", "summaryDescription"]
        ),
        "styles": record.get("styles"),
        "origin": record.get("placesOfOrigin"),
        "api_reported_has_image": bool(image_id),
        "api_detail_image_count": len(record.get("images", [])),
        "image_url": f"https://framemark.vam.ac.uk/collections/{image_id}/full/!768,768/0/default.jpg"
        if image_id
        else None,
        "license": "V&A educational/noncommercial website terms; not CC0",
        "license_url": "https://www.vam.ac.uk/info/va-websites-terms-conditions",
        "attribution": "© Victoria and Albert Museum, London",
        "original_metadata": record,
    }


def candidates(acquirer, source: str, text: str, limit: int):
    if source == "vam":
        url = "https://api.vam.ac.uk/v2/objects/search?" + urlencode(
            {"q": text, "images_exist": 1, "page_size": min(100, limit)}
        )
        result = acquirer.get_json(url, "search", "qualified-vam-search", retries=0)
        for hit in result.get("records", []):
            detail = acquirer.get_json(
                "https://api.vam.ac.uk/v2/museumobject/" + hit["systemNumber"],
                "object",
                "qualified-vam-" + hit["systemNumber"],
                retries=0,
            )
            yield normalize_vam(hit, detail)
    elif source == "aic":
        fields = "id,title,date_display,date_start,date_end,medium_display,place_of_origin,image_id,is_public_domain,main_reference_number,description"
        url = "https://api.artic.edu/api/v1/artworks/search?" + urlencode(
            {"q": text, "limit": min(limit, 100), "fields": fields}
        )
        result = acquirer.get_json(url, "search", "qualified-aic-search", retries=0)
        for r in result.get("data", []):
            iid = r.get("image_id")
            yield {
                "reference_id": f"aic_{r['id']}",
                "source": "aic",
                "source_id": str(r["id"]),
                "title": r.get("title", ""),
                "origin": r.get("place_of_origin"),
                "medium": r.get("medium_display", ""),
                "description": r.get("description"),
                "date": r.get("date_display"),
                "date_interval": [r["date_start"], r["date_end"]]
                if r.get("date_start") is not None and r.get("date_end") is not None
                else None,
                "accession": r.get("main_reference_number"),
                "source_url": f"https://www.artic.edu/artworks/{r['id']}",
                "api_reported_has_image": bool(iid),
                "image_url": f"https://www.artic.edu/iiif/2/{iid}/full/843,/0/default.jpg"
                if iid
                else None,
                "license": "CC0" if r.get("is_public_domain") else "rights_not_confirmed",
                "original_metadata": r,
            }
    elif source == "mia":
        url = (
            "https://search.artsmia.org/" + quote(text, safe="") + "?" + urlencode({"size": limit})
        )
        result = acquirer.get_json(url, "search", "qualified-mia-search", retries=0)
        for hit in result.get("hits", {}).get("hits", []):
            r = hit["_source"]
            dated = str(r.get("dated", ""))
            years = [int(v) for v in re.findall(r"\b(?:1[0-9]{3}|20[0-9]{2})\b", dated)]
            century = re.fullmatch(r"(\d+)(?:st|nd|rd|th) century", dated.strip(), re.I)
            interval = (
                [min(years), max(years)]
                if years
                else [100 * (int(century[1]) - 1), 100 * int(century[1]) - 1]
                if century
                else None
            )
            sid = str(r["id"])
            yield {
                "reference_id": "mia_" + sid,
                "source": "mia",
                "source_id": sid,
                "title": r.get("object_name") or r.get("title", ""),
                "origin": r.get("country"),
                "date": dated,
                "date_interval": interval,
                "medium": r.get("medium", ""),
                "description": r.get("text", ""),
                "source_url": "https://collections.artsmia.org/art/" + sid,
                "accession": r.get("accession_number"),
                "api_reported_has_image": r.get("image") == "valid",
                "image_url": f"https://6.api.artsmia.org/800/{sid}.jpg",
                "license": "Public Domain"
                if r.get("rights_type") == "Public Domain"
                else "rights_not_confirmed",
                "original_metadata": r,
            }
    elif source == "cleveland":
        hits = acquirer.search(source, text, limit=limit)
        for sid in hits:
            r = acquirer.object(source, str(sid))
            row = normalize_record(source, r)
            row.update(
                origin=r.get("culture"),
                date_interval=[r["creation_date_earliest"], r["creation_date_latest"]]
                if r.get("creation_date_earliest") is not None
                and r.get("creation_date_latest") is not None
                else None,
                description=r.get("description", ""),
                api_reported_has_image=bool(row.get("image_url")),
                original_metadata=r,
            )
            yield row
    else:
        raise ValueError("Unknown reference source")


def acquire_qualified(
    acquirer,
    description: str,
    search_text: str,
    *,
    limit: int = 8,
    per_source: int = 24,
    sources=("vam", "mia", "aic", "cleveland"),
) -> dict:
    plan = plan_dynamic_query(description)
    trace = {
        "query": description,
        "api_search_text": search_text,
        "mode": "qualified",
        "plan": plan,
        "accepted": [],
        "candidate_decisions": [],
        "failures": [],
        "searches": [],
        "image_availability_contract": "API image ID -> HTTP download -> decode -> source constraints; picture review remains separate",
        "semantic_scope": "fixed recognized concepts and production dates; unrecognized details are not certified",
    }
    seen = set()
    seen_phashes = []
    def source_candidates(source):
        # Original API query first, then at most two concise museum synonyms.
        # Reference qualification still applies the complete original plan.
        phrases = [search_text]
        for key in plan['concept_keys']:
            if key in {'gourd', 'meiping', 'leather_flask'}:
                phrases.append(CONCEPT_BY_KEY[key].query_phrases[0].replace('Chinese ', ''))
        shorter = re.sub(r'\bChinese\s+', '', search_text, flags=re.I)
        phrases.append(shorter)
        candidates_seen = set()
        for phrase in list(dict.fromkeys(phrases))[:3]:
            trace['searches'].append({'source':source, 'text':phrase})
            for row in candidates(acquirer, source, phrase, per_source):
                if row['reference_id'] not in candidates_seen:
                    candidates_seen.add(row['reference_id'])
                    row['discovery_text'] = phrase
                    yield row
    for source in sources:
        if len(trace["accepted"]) >= limit:
            break
        try:
            for position, row in enumerate(
                source_candidates(source), 1
            ):
                decision = {
                    "reference_id": row["reference_id"],
                    "source": source,
                    "source_api_rank": position,
                    "api_reported_has_image": row["api_reported_has_image"],
                    "download_verified": False,
                }
                trace["candidate_decisions"].append(decision)
                row["source_support"] = source_support(row, plan)
                decision["source_support"] = row["source_support"]
                reason = acquirer.exclusions.reason(row)
                if (
                    reason
                    or not row["api_reported_has_image"]
                    or row["source_support"]["status"] != "source_supported"
                    or row.get("license") == "rights_not_confirmed"
                ):
                    decision.update(
                        status="rejected",
                        reason=reason or "missing_image_or_source_support_or_rights",
                    )
                    continue
                try:
                    binary, request = acquirer.get_image(
                        row, "qualified-image-" + row["reference_id"]
                    )
                    row["image_sha256"] = hashlib.sha256(binary).hexdigest()
                    with Image.open(io.BytesIO(binary)) as opened:
                        image = ImageOps.exif_transpose(opened).convert("RGB")
                        image.load()
                        row.update(
                            width=image.width, height=image.height, **image_fingerprints(image)
                        )
                except Exception as exc:
                    decision.update(
                        status="rejected",
                        reason="image_download_or_decode_failed",
                        failure=str(exc),
                    )
                    continue
                decision["download_verified"] = True
                reason = acquirer.exclusions.reason(row)
                near_duplicate = any(
                    (int(row["phash64"], 16) ^ v).bit_count() <= 6 for v in seen_phashes
                )
                if (
                    reason
                    or row["image_sha256"] in seen
                    or near_duplicate
                    or min(row["width"], row["height"]) < 128
                ):
                    decision.update(status="rejected", reason=reason or "duplicate_or_small_image")
                    continue
                path = acquirer.output / "images" / f"{row['reference_id']}.jpg"
                path.write_bytes(binary)
                metadata = acquirer.output / "metadata" / f"{row['reference_id']}.json"
                metadata.write_text(
                    json.dumps(row["original_metadata"], ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                row.update(
                    image_path=path.relative_to(acquirer.root).as_posix(),
                    metadata_path=metadata.relative_to(acquirer.root).as_posix(),
                    image_request=request,
                    download_verified=True,
                    source_api_rank=position,
                )
                seen.add(row["image_sha256"])
                seen_phashes.append(int(row["phash64"], 16))
                trace["accepted"].append(row)
                decision.update(status="selected", image_sha256=row["image_sha256"])
                if len(trace["accepted"]) >= limit:
                    break
        except Exception as exc:
            trace["failures"].append({"source": source, "error": f"{type(exc).__name__}: {exc}"})
    trace["status"] = "references_available" if trace["accepted"] else "no_qualified_references"
    trace["request_summary"] = {
        "live": sum(not r.get("from_cache", False) for r in acquirer.request_log),
        "cached": sum(r.get("from_cache", False) for r in acquirer.request_log),
    }
    return trace
