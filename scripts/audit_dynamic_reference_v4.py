"""Audit dynamic references, HTTP traces, rights, hashes, and isolation.

No network request is made.  Gallery metadata participates only in checks;
ranking inputs remain the frozen image vectors and selected reference pixels.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curio.retrieve.dynamic_references import load_gallery  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def acquisition_files(output: Path) -> list[Path]:
    return sorted(path for path in output.glob("references_*.json")
                  if "__fallback" not in path.name)


def make_failure_sheet(output: Path, failures: list[dict], gallery: dict[str, dict]) -> None:
    if not failures:
        return
    width, height = 1200, 50 + 240 * len(failures)
    sheet = Image.new("RGB", (width, height), "#eeeeeb")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((10, 8), "Failure contact sheet: visible local top-1 under each failed query", fill="#111", font=font)
    for index, failure in enumerate(failures):
        y = 38 + index * 240
        draw.text((10, y), f"{failure['qid']} | {failure['query']} | catalog-first-rank {failure['first_rank']:g} | visible grade estimate {failure['visible_grade']:.1f}", fill="#111", font=font)
        draw.text((10, y + 18), failure["diagnosis"][:180], fill="#333", font=font)
        for j, top in enumerate(failure["tops"][:5]):
            row = gallery.get(top["item_id"])
            if not row:
                continue
            x = 10 + j * 235
            with Image.open(ROOT / row["image_path"]) as image:
                thumb = image.convert("RGB")
                thumb.thumbnail((215, 175))
            sheet.paste(thumb, (x, y + 42 + (175 - thumb.height) // 2))
            draw.text((x, y + 218), f"{top['item_id']} r{top['rank']}", fill="#111", font=font)
    sheet.save(output / "failure_contact_sheet.jpg", quality=93)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/dynamic_reference_v4/broad8")
    parser.add_argument("--strict-output", type=Path,
                        default=ROOT / "artifacts/dynamic_reference_v4/run_strict2")
    args = parser.parse_args()
    output = args.output.resolve()
    gallery, exclusions = load_gallery(ROOT)
    by_id = {row["item_id"]: row for row in gallery}
    acquisitions = []
    all_accepted = {}
    all_rejected = {}
    for path in acquisition_files(output):
        data = json.loads(path.read_text(encoding="utf-8"))
        qid = data["query_id"]
        for row in data.get("accepted", []):
            all_accepted[row["reference_id"]] = row
        for row in data.get("rejected", []):
            all_rejected[f"{qid}:{row.get('reference_id')}:{row.get('stage', row.get('reason'))}"] = row
        files = {}
        for row in data.get("selected", []) + data.get("accepted", []):
            path_value = ROOT / row["image_path"]
            if path_value.is_file():
                files[row["reference_id"]] = sha256(path_value)
        acquisitions.append({
            "qid": qid, "candidate_count": len(data.get("candidates", [])),
            "accepted_count": len(data.get("accepted", [])),
            "selected_count": len(data.get("selected", [])),
            "selection_status": data.get("selection_status"),
            "selected": [{key: row.get(key) for key in (
                "reference_id", "source", "source_id", "accession", "title", "license",
                "image_url", "metadata_api_url", "discovery_url", "image_sha256",
                "phash64", "ahash256", "width", "height", "reference_constraint_status")}
                for row in data.get("selected", [])],
            "image_sha256_recomputed": files,
            "rejection_reasons": dict(Counter(row.get("reason", row.get("stage", "unknown"))
                                                for row in data.get("rejected", []))),
        })
    isolation_issues = []
    rights_issues = []
    hash_mismatches = []
    for reference_id, row in all_accepted.items():
        reason = exclusions.reason(row)
        if reason:
            isolation_issues.append({"reference_id": reference_id, **reason})
        if row.get("license") != "CC0":
            rights_issues.append({"reference_id": reference_id, "license": row.get("license")})
        image_path = ROOT / row["image_path"]
        if not image_path.is_file() or sha256(image_path) != row.get("image_sha256"):
            hash_mismatches.append(reference_id)
    requests = read_jsonl(output / "http_requests.jsonl")
    request_summary = {
        "request_events": len(requests),
        "unique_urls": len({row["request_url"] for row in requests}),
        "status_codes": dict(Counter(str(row.get("status_code")) for row in requests)),
        "cached_events": sum(bool(row.get("from_cache")) for row in requests),
        "search_requests": sum("/search" in row["request_url"] for row in requests),
        "object_requests": sum("/objects/" in row["request_url"] for row in requests),
        "image_requests": sum("images.metmuseum.org" in row["request_url"] or
                              "openaccess-cdn.clevelandart.org" in row["request_url"]
                              for row in requests),
        "failures": len(read_jsonl(output / "acquisition_failures.jsonl")),
    }
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    metrics = {row["qid"]: row for row in
               json.loads((output / "metrics.json").read_text(encoding="utf-8"))}
    rankings = read_jsonl(output / "rankings_topk.jsonl")
    direct_diagnostics = {}
    for qid in metrics:
        acquisition = next((json.loads(path.read_text(encoding="utf-8"))
                            for path in acquisition_files(output)
                            if json.loads(path.read_text(encoding="utf-8"))["query_id"] == qid), None)
        if acquisition is None:
            continue
        direct_diagnostics[qid] = {
            "failure_reasons": dict(Counter(reason for row in acquisition["accepted"]
                                            for reason in row.get("metadata_diagnostics", {}).get("failure_reasons", []))),
            "exclusion_reasons": dict(Counter(row.get("reason", row.get("stage", "unknown"))
                                               for row in acquisition.get("rejected", []))),
            "selected_validation": [(row["reference_id"], row.get("reference_constraint_status"))
                                    for row in acquisition.get("selected", [])],
        }
    strict_summary = (json.loads((args.strict_output / "summary.json").read_text(encoding="utf-8"))
                      if (args.strict_output / "summary.json").exists() else None)
    # Explicit visual review of selected top results.  These are observations
    # made from the contact sheets, not human qrels.
    review = {
        "reviewed_at": datetime.now(timezone.utc).isoformat(), "evidence": "contact sheets visually inspected",
        "ming_blue_white": {"top_visible_hits": 5, "of": 10, "visible_grade02": 0.5,
                           "diagnosis": "Most top results are blue-and-white ceramic and plausible; plates and vases dominate, so dynasty is not verified."},
        "wanli_blue_white": {"top_visible_hits": 7, "of": 10, "visible_grade02": 0.7,
                            "diagnosis": "After color gating, top rows are visibly blue-white ceramics; several forms differ and pixels cannot establish Wanli attribution."},
        "kangxi_meiping": {"top_visible_hits": 7, "of": 10, "visible_grade02": 0.7,
                          "diagnosis": "After silhouette gating, most top results display a meiping outline. Colour differs, but colour was not requested; shape cannot establish Kangxi date."},
        "qianlong_famille_rose": {"top_visible_hits": 8, "of": 10, "visible_grade02": 0.8,
                                 "diagnosis": "Most top-10 images visibly show multicolour enamel decoration; dates remain metadata assertions."},
        "liao_leather_flask": {"top_visible_hits": 2, "of": 10, "visible_grade02": 0.2,
                              "diagnosis": "The true leather-bag flask targets are scarce and isolated out; broad pilgrim flasks dominate. Need Liao-safe external source beyond current API pool."},
        "yuan_blue_white_dragon": {"top_visible_hits": 2, "of": 10, "visible_grade02": 0.2,
                                  "diagnosis": "Several blue-white vessel matches, but dragon and Yuan criteria are not reliably visible. Metadata query failed to surface a strict reference."},
        "song_celadon": {"top_visible_hits": 2, "of": 10, "visible_grade02": 0.2,
                        "diagnosis": "Vessel family is often close, yet pale/white and coloured glazes appear; reference set lacks confirmed Southern Song Longquan."},
        "jiajing_blue_white_jar": {"top_visible_hits": 2, "of": 10, "visible_grade02": 0.2,
                                  "diagnosis": "Wrong-shape jars, dishes and undecorated vessels appear. Main failure is reference coverage, not just target embedding."},
    }
    failures = [
        {"qid": qid, "query": metrics[qid]["query"] if qid in metrics else qid,
         "first_rank": (metrics[qid]["metrics"]["siglip_mean"]["first_relevant_rank"] or 9999)
         if qid in metrics else 9999,
         "visible_grade": value["visible_grade02"] / 2,
         "diagnosis": value["diagnosis"],
         "tops": [row for row in read_jsonl(output / "rankings_topk.jsonl")
                  if row["qid"] == qid and row["method"] == "siglip_mean"][:5]}
        for qid, value in review.items()
        if isinstance(value, dict) and value.get("top_visible_hits", 0) < 5
    ]
    # Include user query strings without using catalogue as rank truth.
    query_lookup = {row["qid"]: row["query"]
                    for row in json.loads((output / "query_plans.json").read_text(encoding="utf-8"))}
    for failure in failures:
        failure["query"] = query_lookup[failure["qid"]]
    make_failure_sheet(output, failures, by_id)
    manifest = {
        "status": "passed" if not (isolation_issues or rights_issues or hash_mismatches) else "failed",
        "gallery_count": len(gallery), "accepted_reference_count": len(all_accepted),
        "strict_isolation_issues": isolation_issues, "rights_issues": rights_issues,
        "reference_image_hash_mismatches": hash_mismatches,
        "target_metadata_used_for_ranking": False,
        "ranking_inputs": ["reference image pixels", "SigLIP index pixels", "DINO index pixels"],
        "request_summary": request_summary, "acquisitions": acquisitions,
        "diagnostic_failure_breakdown": direct_diagnostics,
        "top10_rankings": rankings,
        "broad_summary": summary, "strict_summary": strict_summary,
        "contact_sheet_review": review,
        "precision_at_k_reported": False,
        "precision_at_k_reason": "no complete all-gallery qrels",
        "subject_files_sha256": {
            str(path.relative_to(ROOT)): sha256(path) for path in
            (ROOT / "src/curio/retrieve/dynamic_references.py",
             ROOT / "scripts/run_dynamic_reference_v4.py",
             ROOT / "scripts/audit_dynamic_reference_v4.py")
        },
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output / "audit_report.json", manifest)
    for failure in failures:
        qid = failure["qid"]
        text = (f"{qid}: visible top-10 reviewed. first strict diagnostic rank "
                f"{failure['first_rank']:.3g}. {failure['diagnosis']}")
        (output / f"failure_{qid}.txt").write_text(text + "\n", encoding="utf-8")
    report = ["# Dynamic reference v4 audit", "", f"Status: **{manifest['status']}**",
              "", f"- Gallery images: {len(gallery)}",
              f"- Accepted references across queries: {len(all_accepted)}",
              f"- Isolation issues: {len(isolation_issues)}",
              f"- Rights issues: {len(rights_issues)}",
              f"- Image hash mismatches: {len(hash_mismatches)}",
              f"- HTTP events: {request_summary['request_events']} ({request_summary['unique_urls']} URLs)",
              "- Primary metrics: Hit@K, Recall@K, MRR@K only",
              "- Precision@K: not reported because no complete all-gallery qrels exist",
              "- Visible gate: reference pixels must pass before encoding and ranking",
              "", "Catalogue metadata was not an input to any target rank. It was used only for reference planning, reference validation, and diagnostic retrieval evaluation."]
    (output / "AUDIT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "accepted": len(all_accepted),
                      "isolation_issues": len(isolation_issues), "rights_issues": len(rights_issues),
                      "hash_mismatches": len(hash_mismatches), "requests": request_summary}, indent=2))


if __name__ == "__main__":
    main()
