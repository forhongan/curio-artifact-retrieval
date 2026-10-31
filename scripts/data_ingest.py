"""Download and normalize CC0 Chinese object images from Met and Cleveland.

The command is resumable: raw API records and downloaded images are cached by
source/id, and the processed JSONL is rebuilt from the cache.  It intentionally
uses only official open-access endpoints and records rights and failures.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
META = PROC / "items.jsonl"
FAIL = PROC / "download_failures.jsonl"
UA = "Curio/0.1 research open-access-ingest (COMP5523)"

_session_local = threading.local()
_request_lock = threading.Lock()
_last_request = 0.0


def session() -> requests.Session:
    if not hasattr(_session_local, "s"):
        s = requests.Session()
        # Met's image CDN returns 406 for a custom User-Agent; default requests UA works.
        s.headers.update({"Accept": "application/json,image/*,*/*"})
        _session_local.s = s
    return _session_local.s


def get(url: str, *, binary: bool = False, tries: int = 5) -> requests.Response:
    global _last_request
    last: Exception | None = None
    for attempt in range(tries):
        try:
            with _request_lock:
                wait = max(0.0, 0.15 - (time.monotonic() - _last_request))
                if wait:
                    time.sleep(wait)
                _last_request = time.monotonic()
            r = session().get(url, timeout=(15, 90))
            if r.status_code in (429, 500, 502, 503, 504):
                raise requests.HTTPError(f"retryable status {r.status_code}")
            r.raise_for_status()
            return r
        except Exception as e:  # network failures are recorded by caller
            last = e
            if isinstance(e, requests.HTTPError) and e.response is not None and e.response.status_code not in (429, 500, 502, 503, 504):
                break
            time.sleep(min(20, 1.5 * (2**attempt)))
    raise RuntimeError(f"GET failed {url}: {last}")


def ensure_dirs() -> None:
    for p in (RAW / "met", RAW / "cleveland", PROC / "images", PROC / "thumbs"):
        p.mkdir(parents=True, exist_ok=True)


def load_json(p: Path) -> dict[str, Any] | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def text(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()


def years(*vals: Any) -> tuple[int | None, int | None]:
    nums: list[int] = []
    for v in vals:
        if isinstance(v, int):
            if -10000 < v < 2100 and v != 0:
                nums.append(v)
            continue
        sv = text(v)
        # Convert century labels such as "18th century" to a useful range anchor.
        for m in re.finditer(r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)\s*[-–]\s*(?:early|late|mid)?\s*(\d{1,2})(?:st|nd|rd|th)\s*century", sv, flags=re.I):
            nums.append((int(m.group(1)) - 1) * 100)
            nums.append((int(m.group(2)) - 1) * 100)
        for m in re.finditer(r"(?<!\d)(\d{1,2})(?:st|nd|rd|th)\s*century", sv, flags=re.I):
            nums.append((int(m.group(1)) - 1) * 100)
        for m in re.findall(r"(?<!\d)(-?\d{3,4})(?!\d)", sv):
            n = int(m)
            if -10000 < n < 2100 and n != 0:
                nums.append(n)
    return (min(nums), max(nums)) if nums else (None, None)


def terms(v: Any) -> list[str]:
    if isinstance(v, list):
        return [text(x.get("term") if isinstance(x, dict) else x) for x in v if text(x.get("term") if isinstance(x, dict) else x)]
    if isinstance(v, str):
        return [x.strip() for x in re.split(r"[,;|]", v) if x.strip()]
    return []


def ahash(im: Image.Image) -> str:
    g = ImageOps.grayscale(im).resize((16, 16))
    px = list(g.getdata())
    avg = sum(px) / len(px)
    return "".join("1" if p >= avg else "0" for p in px)


def split_for(item_id: str) -> str:
    n = int(hashlib.sha256(item_id.encode()).hexdigest()[:8], 16) % 10
    return "train" if n < 8 else "dev" if n == 8 else "test"


def save_image(item_id: str, image_url: str, source: str) -> tuple[str, str, int, int, str, str, str, str | None]:
    """Return paths, dimensions, sha and perceptual hash."""
    ip = PROC / "images" / f"{item_id}.jpg"
    tp = PROC / "thumbs" / f"{item_id}.jpg"
    source_id = item_id.split("_", 1)[1]
    rawp = RAW / source / f"{source_id}.jpg"
    if not rawp.exists():
        rawp.write_bytes(get(image_url).content)
    raw = rawp.read_bytes()
    if not ip.exists():
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        if min(im.size) < 256 or max(im.size) < 512:
            raise ValueError(f"image below raw quality floor (short edge>=256,long edge>=512): {im.size}")
        im.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        w, h = im.size
        im.save(ip, format="JPEG", quality=94, optimize=True)
    else:
        im = Image.open(ip).convert("RGB")
        w, h = im.size
        im = Image.open(ip).convert("RGB")
    if not tp.exists():
        thumb = im.copy()
        thumb.thumbnail((320, 320), Image.Resampling.LANCZOS)
        thumb.save(tp, format="JPEG", quality=88, optimize=True)
    sha = hashlib.sha256(raw).hexdigest()
    return (f"data/processed/images/{item_id}.jpg", f"data/processed/thumbs/{item_id}.jpg", w, h, sha,
            hashlib.sha256(ip.read_bytes()).hexdigest(), hashlib.sha256(tp.read_bytes()).hexdigest(), ahash(im))


def met_ids(limit: int) -> list[int]:
    out: list[int] = []
    browser_files = []
    for pat in ("met_browser_ceramics_*.json", "met_browser_jade_*.json", "met_browser_metalwork_*.json", "met_browser_textiles_*.json", "met_browser_china_offset*.json"):
        browser_files.extend(sorted((ROOT / "data" / "raw").glob(pat)))
    for cached in browser_files:
        try:
            d = json.loads(cached.read_text(encoding="utf-8"))
            out.extend(int(x) for x in (d.get("objectIDs") or []))
        except Exception:
            pass
    for cached in (ROOT / "data" / "raw" / "met_focused_1.json", ROOT / "data" / "raw" / "met_focused_2.json"):
      if cached.exists():
        try:
            out.extend(int(x) for x in (json.loads(cached.read_text(encoding="utf-8")).get("objectIDs") or []))
        except Exception:
            pass
    offset = 0
    while len(out) < limit:
        size = min(500, limit - len(out))
        u = f"https://collectionapi.metmuseum.org/public/collection/v1.1/search?geoLocation=China&hasImages=true&departmentId=6&q=China&limit={size}&offset={offset}"
        try:
            data = get(u).json()
        except RuntimeError as e:
            print(f"Met candidate pagination stopped at {offset}: {e}", flush=True)
            break
        ids = data.get("objectIDs") or []
        if not ids:
            break
        out.extend(int(x) for x in ids)
        offset += len(ids)
        if offset >= int(data.get("total") or offset):
            break
    return list(dict.fromkeys(out))[:limit]


def cleveland_ids(limit: int) -> list[int]:
    u = f"https://openaccess-api.clevelandart.org/api/artworks/?cc0=1&has_image=1&culture=China&limit={min(limit,100)}"
    out: list[int] = []
    cached = ROOT / "data" / "raw" / "cma_first_page.json"
    if cached.exists():
        try:
            rows = json.loads(cached.read_text(encoding="utf-8")).get("data") or []
            for row in rows:
                if row.get("id"):
                    out.append(int(row["id"]))
                    (RAW / "cleveland" / f"{row['id']}.json").write_text(json.dumps(row, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    if len(out) >= limit:
        return out[:limit]
    skip = 0
    while len(out) < limit:
        d = get(u + f"&skip={skip}").json()
        rows = d.get("data") or []
        if not rows:
            break
        out.extend(int(r["id"]) for r in rows if r.get("id"))
        skip += len(rows)
        if len(rows) < 100:
            break
    return out[:limit]


def normalize_met(d: dict[str, Any]) -> tuple[dict[str, Any], str]:
    iid = f"met_{d['objectID']}"
    image_url = d.get("primaryImage") or d.get("primaryImageSmall")
    title = text(d.get("title"))
    y0, y1 = years(d.get("objectBeginDate"), d.get("objectEndDate"))
    if y0 is None:
        y0, y1 = years(d.get("objectDate"), d.get("period"))
    obj = {
        "item_id": iid, "source": "met", "source_id": str(d["objectID"]), "title": title,
        "object_name": text(d.get("objectName")), "classification": text(d.get("classification")),
        "medium": text(d.get("medium")), "culture": text(d.get("culture")), "period": text(d.get("period")),
        "year_from": y0, "year_to": y1, "tags": terms(d.get("tags")), "description": "",
        "image_url": image_url, "source_url": f"https://www.metmuseum.org/art/collection/search/{d['objectID']}",
        "license": "CC0", "credit_line": text(d.get("creditLine")),
        "object_type": text(d.get("objectName")), "material": text(d.get("medium")),
        "motifs": terms(d.get("tags")), "colors": [], "split": split_for(iid),
    }
    return obj, image_url or ""


def normalize_cle(d: dict[str, Any]) -> tuple[dict[str, Any], str]:
    iid = f"cma_{d['id']}"
    imgs = d.get("images") or {}
    image_url = (imgs.get("print") or imgs.get("web") or {}).get("url")
    y0, y1 = years(d.get("creation_date_earliest"), d.get("creation_date_latest"))
    if y0 is None:
        y0, y1 = years(d.get("creation_date"))
    obj = {
        "item_id": iid, "source": "cleveland", "source_id": str(d["id"]), "title": text(d.get("title")),
        "object_name": text(d.get("type")), "classification": text(d.get("classification")),
        "medium": text(d.get("technique") or d.get("medium")), "culture": "; ".join(terms(d.get("culture"))), "period": text(d.get("creation_date")),
        "year_from": y0, "year_to": y1, "tags": terms(d.get("technique")),
        "description": text(d.get("description")), "image_url": image_url,
        "source_url": text(d.get("url")) or f"https://www.clevelandart.org/art/{d.get('accession_number', '')}", "license": "CC0",
        "credit_line": text(d.get("creditline")), "object_type": text(d.get("type")),
        "material": text(d.get("medium")), "motifs": [], "colors": [],
        "split": split_for(iid),
    }
    return obj, image_url or ""


def fetch_one(source: str, sid: int, cached_only: bool = False) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    base = RAW / source
    mp = base / f"{sid}.json"
    try:
        d = load_json(mp)
        if d is None:
            if cached_only:
                raise ValueError("raw metadata not cached")
            if source == "met":
                d = get(f"https://collectionapi.metmuseum.org/public/collection/v1/objects/{sid}").json()
            else:
                d = get(f"https://openaccess-api.clevelandart.org/api/artworks/{sid}").json().get("data", {})
            mp.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        if source == "met" and d.get("isPublicDomain") is not True:
            raise ValueError("rights: Met isPublicDomain is not true")
        if source == "cleveland" and d.get("share_license_status") != "CC0":
            raise ValueError("rights: Cleveland share_license_status is not CC0")
        item, image_url = normalize_met(d) if source == "met" else normalize_cle(d)
        if item["year_from"] is None or item["year_from"] >= 1912:
            raise ValueError("date: missing or modern (1912 onward)")
        origin = " ".join(text(d.get(k)) for k in ("country", "culture", "geographyType", "find_spot"))
        country = text(d.get("country")).lower()
        if source == "met" and country and country not in ("china", "china (?)"):
            raise ValueError(f"origin: explicit country {country}")
        if source == "met" and not re.search(r"\b(china|chinese)\b", origin, re.I):
            raise ValueError("origin: missing explicit China/Chinese evidence")
        item["origin_basis"] = origin or ("Met geoLocation=China search" if source == "met" else "Cleveland culture=China search")
        content = " ".join(text(d.get(k)) for k in ("title", "objectName", "classification", "type")).lower()
        if re.search(r"\b(painting|calligraphy|photograph|book|album|print|drawing|rubbing|scroll|manuscript)\b", content):
            raise ValueError("type: excluded flat media")
        def match(choices: list[tuple[str, str]], value: str) -> str:
            return next((canonical for pattern, canonical in choices if re.search(pattern, value)), "")
        objtypes = [(r"\bbowls?\b", "bowl"), (r"\bplates?|platters?|dishes?\b", "plate"), (r"\bvases?|bottles?\b", "vase"), (r"\bjars?\b", "jar"), (r"\bteapots?\b", "teapot"), (r"\bcups?\b", "cup"), (r"\bpots?\b", "pot"), (r"\bfigur(?:e|ines?)|bodhisattva|buddha|statues?\b", "figure"), (r"\brobes?|garments?\b", "textile"), (r"\bpendants?|pendant|jade pendants?\b", "pendant"), (r"\bcong|jade cong\b", "cong"), (r"\bbi disk|bi disc|jade bi\b", "bi"), (r"\bboxes?\b", "box"), (r"\bvessels?\b", "ritual vessel"), (r"\bmirrors?\b", "mirror"), (r"\bstelae?|steles?\b", "stele"), (r"\bjewelry|ornament|scepter\b", "ornament"), (r"\bincense burners?\b", "censer")]
        mats = [(r"\bporcelain\b", "porcelain"), (r"\bjade\b", "jade"), (r"\bbronze\b", "bronze"), (r"\bceramic|pottery|stoneware|earthenware\b", "ceramic"), (r"\bsilk|textile|cotton|wool\b", "textile"), (r"\bwood\b", "wood"), (r"\bgold\b", "gold"), (r"\bsilver\b", "silver"), (r"\bglass\b", "glass"), (r"\blacquer\b", "lacquer"), (r"\bmarble|stone\b", "stone")]
        item["object_type"] = match(objtypes, content)
        item["material"] = match(mats, item["medium"].lower())
        motifs = [("dragon", "dragon"), ("lotus", "lotus"), ("flower", "floral"), ("floral", "floral"), ("bird", "bird"), ("cloud", "cloud"), ("phoenix", "phoenix"), ("fish", "fish")]
        item["motifs"] = [canonical for word, canonical in motifs if re.search(r"\b" + word + r"s?\b", (" ".join(item["tags"]) + " " + item["title"]).lower())]
        item["mapping_basis"] = "object_type: title/object name/classification; material: medium; motifs: title/tags"
        if not image_url:
            raise RuntimeError("no image URL")
        if cached_only and not (base / f"{sid}.jpg").exists():
            raise ValueError("raw image not cached")
        paths = save_image(item["item_id"], image_url, source)
        item.update({"image_path": paths[0], "thumb_path": paths[1], "width": paths[2], "height": paths[3], "sha256": paths[4], "image_sha256": paths[5], "thumb_sha256": paths[6], "phash": paths[7]})
        return item, None
    except Exception as e:
        return None, {"source": source, "source_id": str(sid), "error": repr(e), "time": time.time()}


def finalize(items: list[dict[str, Any]], target: int = 0) -> list[dict[str, Any]]:
    """Deterministically deduplicate raw bytes and group identical perceptual hashes."""
    selected = []
    shas: set[str] = set()
    groups: dict[str, str] = {}
    for row in sorted(items, key=lambda x: x["item_id"]):
        if row["sha256"] in shas:
            continue
        shas.add(row["sha256"])
        row.pop("near_duplicate_of", None)
        row["split"] = split_for(row["item_id"])
        row["object_group"] = row.get("object_group") or row["item_id"]
        ph = row.get("phash")
        if ph and ph in groups:
            row["flags"] = ["near_duplicate_phash"]
            row["near_duplicate_of"] = groups[ph]
            row["split"] = split_for(groups[ph])
        elif ph:
            groups[ph] = row["item_id"]
        selected.append(row)
        if target and len(selected) >= target:
            break
    return selected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=5000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--met-candidates", type=int, default=9000)
    ap.add_argument("--cleveland", type=int, default=0, help="also download up to this many Cleveland objects")
    args = ap.parse_args()
    ensure_dirs()
    ids = [("met", x) for x in met_ids(args.met_candidates)]
    if args.cleveland:
        ids += [("cleveland", x) for x in cleveland_ids(args.cleveland)]
    existing: dict[str, dict[str, Any]] = {}
    # Resume from the last published table when a previous run already downloaded records.
    if META.exists():
        for line in META.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if row.get("item_id"):
                    existing[row["item_id"]] = row
            except Exception:
                continue
    failures: list[dict[str, Any]] = []
    todo = ids
    todo = [(s, i) for s, i in todo if f"{'met' if s == 'met' else 'cma'}_{i}" not in existing]
    # Interleave sources so a useful text-bearing subset appears early.
    if args.cleveland:
        m = [(s, i) for s, i in ids if s == "met"]
        c = [(s, i) for s, i in ids if s == "cleveland"]
        todo = [x for j in range(max(len(m), len(c))) for x in (([m[j]] if j < len(m) else []) + ([c[j]] if j < len(c) else []))]
    def publish() -> None:
        selected = finalize(list(existing.values()), args.target)
        tmp = META.with_suffix(".tmp")
        tmp.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in selected) + ("\n" if selected else ""), encoding="utf-8")
        os.replace(tmp, META)
        FAIL.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in failures) + ("\n" if failures else ""), encoding="utf-8")
    print(f"candidates={len(ids)} cached={len(existing)} todo={len(todo)}", flush=True)
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for n, (item, failure) in enumerate(ex.map(lambda x: fetch_one(*x), todo), 1):
            if item:
                existing[item["item_id"]] = item
            if failure:
                failures.append(failure)
            if n % 50 == 0:
                print(f"processed={n}/{len(todo)} items={len(existing)} failed={len(failures)}", flush=True)
                publish()
            if len(existing) >= args.target and not args.cleveland:
                # Continue the executor map so already submitted requests finish; output is trimmed below.
                pass
    publish()
    print(f"wrote {len(existing)} validated records to {META}; failures={len(failures)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
