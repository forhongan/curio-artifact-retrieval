"""Description -> isolated museum photographs -> image-only gallery retrieval.

The scorer deliberately accepts arrays and IDs, never target catalogue records.
Reference source metadata validates a photograph's identity/attribution only.
No score is a probability or a visual certification of the requested period.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from curio.represent.visual_parts import extract_visual_parts, load_visual_parts
from curio.retrieve.dynamic_references import (
    DynamicMuseumAcquirer,
    SOURCE_BASES,
    load_gallery,
    plan_dynamic_query,
    reference_metadata_score,
    select_reference_set,
)
from curio.retrieve.image_search import unit_rows
from curio.retrieve.references import accession_root, image_fingerprints, normalize_record


VERSION = "dynamic-reference-runtime-v1"
FORM_KEYS = {"meiping", "leather_flask", "gourd", "jar", "bowl", "dish", "vase"}
COLOR_KEYS = {"blue_white", "famille_rose", "polychrome", "celadon"}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def visual_reference_evidence(path: Path, plan: dict) -> tuple[dict, dict]:
    """Only reliable visible contradictions exclude source-qualified references.

    Unknown masks remain unknown. A museum photograph on a textured background
    must not silently become an incorrect form/color negative.
    """
    parts = extract_visual_parts(path)
    checks = []
    valid = parts["diagnostics"]["status"] == "valid"
    if valid:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image = image.resize(tuple(parts["diagnostics"]["image_size"]))
        hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
        hue, saturation, value = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
        mask = parts["mask"] > 0
        for key in plan["concept_keys"]:
            if key not in COLOR_KEYS:
                continue
            if key == "blue_white":
                signal = (hue >= 180) & (hue <= 280) & (saturation > .22) & (value > .12)
                threshold, name = .002, "foreground_blue_fraction"
            elif key == "celadon":
                signal = (hue >= 70) & (hue <= 200) & (saturation > .12) & (value > .2)
                threshold, name = .015, "foreground_green_fraction"
            else:
                signal = (saturation > .28) & (value > .22)
                threshold, name = .02, "foreground_saturated_fraction"
            fraction = float(signal[mask].mean())
            checks.append({"concept": key, "signal": name, "value": fraction,
                           "threshold": threshold, "status": "supports" if fraction >= threshold
                           else "contradicts"})
        if "meiping" in plan["concept_keys"]:
            yy, _ = np.where(mask)
            widths = mask[yy.min():yy.max() + 1].sum(axis=1)
            height = len(widths)
            shoulder = float(np.median(widths[int(.20 * height):int(.35 * height) + 1]))
            neck = float(np.median(widths[int(.02 * height):int(.14 * height) + 1]))
            waist = float(np.median(widths[int(.55 * height):int(.72 * height) + 1]))
            lower = float(np.median(widths[int(.74 * height):int(.94 * height) + 1]))
            # A low studio shadow can double the total mask bounding-box
            # width. Use the upper/body profile for aspect; keep lower-body
            # checks separate. Otherwise a good green meiping is rejected as
            # "wide" solely because of its photographed shadow.
            body_width = float(np.quantile(widths[int(.18 * height):int(.70 * height) + 1], .95))
            aspect = height / max(1, body_width)
            passed = (aspect >= 1.15 and .42 <= waist / max(1, shoulder) <= 1.5
                      and neck / max(1, shoulder) <= .72 and lower / max(1, shoulder) <= 1.08)
            # A nominally "valid" planar-background mask can include lateral
            # lighting halos. The inspected green reference had two broad
            # wings across its body. Treat conflicting profile estimates as
            # uncertain segmentation, rather than asserting a wrong form.
            unstable_width = (body_width > 1.5 * max(1, shoulder)
                              and aspect < 1.15 <= height / max(1, shoulder))
            if unstable_width:
                parts["diagnostics"] = {**parts["diagnostics"], "status": "unknown",
                                        "runtime_reasons": ["lateral_mask_width_instability"],
                                        "valid": {name: False for name in ("shape", "color", "texture")}}
                parts["validity"] = parts["valid"] = {name: False for name in ("shape", "color", "texture")}
            checks.append({"concept": "meiping", "signal": "foreground_profile",
                           "status": "unknown" if unstable_width else "supports" if passed else "contradicts",
                           "aspect": float(aspect), "neck_to_shoulder": neck / max(1, shoulder),
                           "aspect_width_policy": "upper_body_profile_excludes_base_shadow",
                           "waist_to_shoulder": waist / max(1, shoulder),
                           "lower_to_shoulder": lower / max(1, shoulder)})
    # A failed mask does not prohibit a directly visible color corroboration.
    # Conversely, absence of white throughout a strongly blue central region
    # is an observable contradiction even without knowing the exact outline.
    # A grayscale archival photograph stays unknown; zero blue is not a
    # negative when the photograph itself contains no chroma information.
    if "blue_white" in plan["concept_keys"] and not valid:
        with Image.open(path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((320, 320))
        hsv = np.asarray(image.convert("HSV"), dtype=np.float32)
        height, width = hsv.shape[:2]
        hsv = hsv[int(.1 * height):int(.9 * height), int(.1 * width):int(.9 * width)]
        hue, saturation, value = hsv[..., 0] * 360 / 255, hsv[..., 1] / 255, hsv[..., 2] / 255
        blue = float(((hue >= 180) & (hue <= 280) & (saturation > .22) & (value > .12)).mean())
        white = float(((saturation < .20) & (value > .65)).mean())
        if blue > .5 and white < .002:
            checks.append({"concept": "blue_white", "signal": "unsegmented_color_contradiction",
                           "status": "contradicts", "blue_fraction": blue, "white_fraction": white,
                           "rule": "strongly blue central photograph with no visible light neutral body/decoration"})
        elif blue > .002 and white > .01:
            checks.append({"concept": "blue_white", "signal": "unsegmented_color_corroboration",
                           "status": "supports", "blue_fraction": blue, "white_fraction": white,
                           "limitation": "visible colors only; no foreground mask or technique validation"})
    status = ("contradicted" if any(row["status"] == "contradicts" for row in checks)
              else "supported" if any(row["status"] == "supports" for row in checks) else "unknown")
    return {"status": status, "checks": checks, "segmentation": parts["diagnostics"],
            "rule": "reject reliable visible contradictions; retain unknown with disclosure",
            "limitation": "coarse color/profile assistance; cannot verify technique, motif or date"}, parts


def _order(values: np.ndarray, ids: list[str]) -> np.ndarray:
    return np.array(sorted(range(len(ids)), key=lambda i: (-float(values[i]), ids[i])))


def _rrf(values: np.ndarray, ids: list[str]) -> np.ndarray:
    # Equal similarities get equal rank; lexical tie breaking is presentation only.
    order = _order(values, ids)
    ranks = np.empty(len(ids), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return 61 / (60 + ranks)


def reference_consensus(siglip: np.ndarray, dino: np.ndarray) -> dict:
    """Downweight visually redundant references, preserve distinct modes.

    Clusters express image similarity, not independent historical evidence.
    The conservative complete-link rule avoids chaining dissimilar endpoints.
    """
    siglip, dino = unit_rows(siglip), unit_rows(dino)
    if len(siglip) != len(dino):
        raise ValueError("Reference channel lengths differ")
    sm, dm = siglip @ siglip.T, dino @ dino.T
    clusters: list[list[int]] = []
    for index in range(len(siglip)):
        for cluster in clusters:
            if all(sm[index, other] >= .80 and dm[index, other] >= .65 for other in cluster):
                cluster.append(index)
                break
        else:
            clusters.append([index])
    weights = np.empty(len(siglip), dtype=np.float32)
    for cluster in clusters:
        weights[cluster] = 1 / (len(clusters) * len(cluster))
    pairs = [(i, j) for i in range(len(siglip)) for j in range(i)]
    return {"reference_count": len(siglip), "visual_clusters": clusters,
            "cluster_count": len(clusters), "reference_weights": weights.tolist(),
            "siglip_pairwise": sm.tolist(), "dino_pairwise": dm.tolist(),
            "minimum_pair_cosine": {"siglip": min((float(sm[i, j]) for i, j in pairs), default=None),
                                    "dino": min((float(dm[i, j]) for i, j in pairs), default=None)},
            "state": "single_reference" if len(siglip) == 1 else
                     "visually_concentrated" if len(clusters) == 1 else "visually_diverse",
            "interpretation": "visual agreement/diversity only; no calibrated correctness"}


def score_gallery(ids: list[str], siglip_gallery: np.ndarray, dino_gallery: np.ndarray,
                  siglip_references: np.ndarray, dino_references: np.ndarray,
                  *, auxiliary: dict | None = None, assist_window: int = 100) -> dict:
    """Pure image scorer; target labels/descriptions cannot enter this contract.

    Shape/color assistance only exchanges the baseline slots of candidates with
    valid masks inside a bounded recall window. Unknown masks keep their slots.
    All gallery images are scored by both full-image channels first.
    """
    if len(ids) != len(set(ids)) or len(ids) != len(siglip_gallery) or len(ids) != len(dino_gallery):
        raise ValueError("Gallery lengths/IDs differ")
    sg, dg = unit_rows(siglip_gallery), unit_rows(dino_gallery)
    sr, dr = unit_rows(siglip_references), unit_rows(dino_references)
    if sg.shape[1] != sr.shape[1] or dg.shape[1] != dr.shape[1]:
        raise ValueError("Gallery/reference dimensions differ")
    consensus = reference_consensus(sr, dr)
    weights = np.array(consensus["reference_weights"], dtype=np.float32)
    sm, dm = sg @ sr.T, dg @ dr.T
    # The best-reference term keeps legitimate visual modes from cancelling in
    # a broad description; equal-cluster weighting bounds repeated-view votes.
    channels = {"siglip_mean": sm.mean(axis=1), "dino_mean": dm.mean(axis=1),
                "siglip_diverse": .75 * (sm @ weights) + .25 * sm.max(axis=1),
                "dino_diverse": .75 * (dm @ weights) + .25 * dm.max(axis=1)}
    channels["fusion"] = .70 * _rrf(channels["siglip_diverse"], ids) + .30 * _rrf(channels["dino_diverse"], ids)
    baseline = _order(channels["fusion"], ids)
    assisted = baseline.copy()
    assistance = {"status": "not_requested", "window": min(len(ids), assist_window),
                  "unknown_policy": "retain baseline slot", "channels": {}}
    if auxiliary:
        window = baseline[:max(0, assist_window)]
        combined, count = np.zeros(len(ids)), np.zeros(len(ids))
        for name, raw in auxiliary.items():
            values = np.asarray(raw, dtype=np.float64)
            if values.shape != (len(ids),):
                raise ValueError("Auxiliary length differs from gallery")
            accepted = window[np.isfinite(values[window])]
            assistance["channels"][name] = {"valid_gallery_count": int(np.isfinite(values).sum()),
                                             "valid_window_count": len(accepted)}
            if len(accepted) < 2:
                continue
            # Relative rank among the same known candidates; disparate color
            # histogram/profile cosine scales are never averaged directly.
            relative = _rrf(values[accepted], [ids[i] for i in accepted])
            combined[accepted] += relative
            count[accepted] += 1
        known = window[count[window] > 0]
        if len(known) >= 2:
            mixed = .80 * channels["fusion"][known] + .20 * combined[known] / count[known]
            replacements = known[_order(mixed, [ids[i] for i in known])]
            slots = np.flatnonzero(np.isin(assisted, known))
            assisted[slots] = replacements
            assistance.update(status="applied", reordered_candidate_count=len(known),
                              rule="0.8 full-image RRF + 0.2 auxiliary RRF within accepted slots")
        else:
            assistance.update(status="insufficient_valid_pairs", reordered_candidate_count=0)
    orders = {name: _order(values, ids) for name, values in channels.items()}
    orders["assisted"] = assisted
    tops = [[ids[i] for i in _order(sm[:, j], ids)[:5]] for j in range(sm.shape[1])]
    consensus["individual_siglip_top5"] = tops
    consensus["top5_pairwise_overlap"] = [
        {"first": i, "second": j, "intersection": len(set(tops[i]) & set(tops[j]))}
        for i in range(len(tops)) for j in range(i)]
    return {"orders": orders, "channels": channels, "consensus": consensus,
            "assistance": assistance, "siglip_pairs": sm, "dino_pairs": dm}


def load_image_index(directory: Path, data_sha256: str) -> tuple[np.ndarray, list[str], dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    values = np.load(directory / "image_embeddings.npy", allow_pickle=False)
    ids = [str(value) for value in json.loads((directory / "item_ids.json").read_text(encoding="utf-8"))]
    digest = hashlib.sha256(values.tobytes() + json.dumps(ids).encode()).hexdigest()
    if digest != manifest["index_sha256"] or manifest["data_sha256"] != data_sha256:
        raise ValueError(f"Gallery image index fingerprint mismatch: {directory}")
    if len(ids) != len(values) or len(ids) != len(set(ids)):
        raise ValueError("Image index IDs are not unique/parallel")
    return unit_rows(values), ids, manifest


def make_sheet(path: Path, title: str, cells: list[tuple[str, Path]], columns: int = 6) -> None:
    width, height = 225, 230
    rows = max(1, (len(cells) + columns - 1) // columns)
    canvas = Image.new("RGB", (columns * width, rows * height + 44), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 10), title, fill="black")
    for index, (label, image_path) in enumerate(cells):
        x, y = index % columns * width, index // columns * height + 44
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail((width - 12, height - 43))
            canvas.paste(image, (x + (width - image.width) // 2, y))
        draw.text((x + 5, y + height - 40), label[:33], fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


class DynamicReferenceRuntime:
    """Reusable lazy runtime. Retain an instance to reuse its frozen encoders.

    ``run`` defaults to exact museum API text retrieval, writes an audit
    directory and returns a JSON-compatible result. The optional curated mode
    retains the older metadata-filtered experiment and same-description replay.
    """

    def __init__(self, root: Path | str, *, device: str = "auto"):
        self.root = Path(root).resolve()
        self.device = device
        self._gallery = None
        self._siglip = None
        self._dino = None
        self._indices = None
        self._parts = None

    def _load(self):
        if self._gallery is None:
            self._gallery = load_gallery(self.root)
        if self._indices is None:
            digest = hashlib.sha256((self.root / "data/snapshots/collection_v2.jsonl").read_bytes()).hexdigest()
            siglip = load_image_index(self.root / "artifacts/siglip", digest)
            dino = load_image_index(self.root / "artifacts/dinov2", digest)
            if siglip[1] != dino[1]:
                raise ValueError("SigLIP/DINO gallery IDs or ordering differ")
            self._indices = siglip, dino

    def _encode(self, selected: list[dict], output: Path) -> tuple[np.ndarray, np.ndarray, dict]:
        from curio.represent.encoders import Encoder
        from curio.represent.image_encoder import DinoEncoder
        siglip, dino = self._indices
        signature = {"images": [row["image_sha256"] for row in selected],
                     "siglip": siglip[2]["encoder_metadata"], "dino": dino[2]["encoder_metadata"]}
        checkpoint = output / "reference_embeddings.npz"
        manifest = output / "reference_embeddings.json"
        if checkpoint.exists() and manifest.exists():
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            if saved["signature"] == signature and saved["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest():
                with np.load(checkpoint, allow_pickle=False) as values:
                    return values["siglip"], values["dino"], saved["encoders"]
        if self._siglip is None:
            self._siglip = Encoder("siglip", device=self.device)
        if self._dino is None:
            self._dino = DinoEncoder(device=self.device)
        for encoder, index in ((self._siglip, siglip), (self._dino, dino)):
            expected = index[2]["encoder_metadata"]
            if encoder.metadata["revision"] != expected["revision"] or expected.get("adapter_path"):
                raise ValueError("Reference encoder must match the unadapted gallery checkpoint")
        paths = [self.root / row["image_path"] for row in selected]
        sr, dr = self._siglip.encode_images(paths), self._dino.encode_images(paths)
        np.savez_compressed(checkpoint, siglip=sr, dino=dr)
        encoders = {"siglip": self._siglip.metadata, "dino": self._dino.metadata}
        write_json(manifest, {"signature": signature, "encoders": encoders,
                              "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()})
        return sr, dr, encoders

    def run(self, description: str, output: Path | str, *, mode: str = "direct",
            search_text: str | None = None, **options) -> dict:
        """Default: exact API text hits -> usable external photos -> gallery pixels.

        ``search_text`` is an explicit caller-supplied translation/override. No
        hidden rule planner or automatic metadata taxonomy filter runs in the
        direct mode. ``mode='curated'`` retains the older stricter experiment.
        """
        if mode == "direct":
            return self.run_direct(description, output, search_text=search_text, **options)
        if mode == "qualified":
            return self.run_qualified(description, output, search_text=search_text, **options)
        if mode == "curated":
            if search_text is not None:
                raise ValueError("search_text override applies to direct mode only")
            return self.run_curated(description, output, **options)
        raise ValueError("mode must be direct, qualified or curated")

    def run_qualified(self, description: str, output: Path | str, *, search_text: str | None = None,
                      cache_dir: Path | str | None = None, offline: bool = False,
                      reference_limit: int = 8, top_k: int = 24,
                      aggregation: str = 'uniform_mean', sources=('vam','mia','aic','cleveland')) -> dict:
        """Verified available source-supported photos, followed by image matching."""
        from curio.retrieve.qualified_references import acquire_qualified, api_search_text
        from curio.retrieve.reference_aggregation import METHODS, aggregate_references

        if not description.strip() or not (search_text or description).strip():
            raise ValueError('Description/search text must be nonempty')
        if not 1 <= reference_limit <= 12 or not 1 <= top_k <= 100 or aggregation not in METHODS:
            raise ValueError('Invalid qualified reference options')
        self._load()
        output=Path(output).resolve()
        if not output.is_relative_to(self.root):
            raise ValueError('Audit output must be inside workspace')
        acquirer=DynamicMuseumAcquirer(self.root,output,self._gallery[1],offline=offline,
                                      cache_dir=Path(cache_dir) if cache_dir else None)
        effective_text=search_text or api_search_text(description)
        trace=acquire_qualified(acquirer,description,effective_text,limit=reference_limit,sources=sources)
        # Shared acquisition contract also serves only these hash-verified images.
        write_json(output/'direct_acquisition.json',trace)
        write_json(output/'request_log.json',acquirer.request_log)
        selected=trace['accepted']
        result={'mode':'qualified_api_text', 'description':description, 'query':description, 'search_text':effective_text,
                'plan':{'query_terms':[effective_text], 'required_eras':[r['label'] for r in trace['plan']['era_constraints']], 'is_llm':False},
                'status':'ranked' if selected else 'no_qualified_references',
                'gallery_count':len(self._indices[0][1]),'reference_count':len(selected),
                'reference_ids':[r['reference_id'] for r in selected], 'results':[],
                'acquisition':{'request_summary':trace['request_summary'],'failures':trace['failures']},
                'warnings':['Source qualification covers recognized terms; actual picture relevance is not guaranteed.',
                            'Image similarity does not verify the dates or authenticity of target objects.'],
                'candidate_catalogue_used_for_ranking':False}
        if selected:
            sr,dr,encoders=self._encode(selected,output)
            sg,dg=self._indices
            # A reference embedding cache hit does not instantiate encoders.
            # Load only the text-query encoder here; never recompute the gallery.
            if self._siglip is None:
                from curio.represent.encoders import Encoder
                self._siglip=Encoder('siglip',device=self.device)
            query=self._siglip.encode_texts([description])[0]
            combined=aggregate_references(sg[0],dg[0],sr,dr,query,gallery_ids=sg[1])
            method=combined['methods'][aggregation]
            result.update(aggregation=aggregation, encoders=encoders,
                          text_assisted_reference_weighting=method['text_assisted_reference_selection_or_weighting'],
                          rankings={name:[sg[1][i] for i in value['orders']['fusion']]
                                    for name,value in combined['methods'].items()},
                          ranking_inputs=['reference image pixels','gallery image pixels'] +
                          (['query text for reference selection/weighting'] if method['text_assisted_reference_selection_or_weighting'] else []))
            np.savez_compressed(output/'gallery_scores.npz',item_ids=np.array(sg[1]),
                                **{name:value['scores']['fusion'] for name,value in combined['methods'].items()})
            for rank,i in enumerate(method['orders']['fusion'][:top_k],1):
                result['results'].append({'item_id':sg[1][i], 'rank':rank,
                                          'score':float(method['scores']['fusion'][i])})
            make_sheet(output/'reference_review.jpg','Source-supported API references; image review separate',
                       [(r['reference_id'],self.root/r['image_path']) for r in selected],columns=4)
            by_id={r['item_id']:r for r in self._gallery[0]}
            make_sheet(output/'results.jpg','Qualified reference image matching',
                       [(r['item_id'],self.root/by_id[r['item_id']]['image_path']) for r in result['results']],columns=4)
        write_json(output/'result.json',result)
        return result

    def run_direct(self, description: str, output: Path | str, *, search_text: str | None = None,
                   cache_dir: Path | str | None = None, offline: bool = False,
                   acquisition: Path | str | None = None, reference_limit: int = 8,
                   per_query: int = 30, page_depth: int = 1,
                   max_metadata_candidates: int = 120, top_k: int = 24) -> dict:
        """Use API hit relevance/order, without hand-enforced form/era taxonomy."""
        if acquisition is not None:
            raise ValueError("Direct mode does not replay filtered acquisitions; use its HTTP cache")
        if not description.strip() or not (search_text or description).strip():
            raise ValueError("Description/search text must be nonempty")
        if not 1 <= reference_limit <= 12 or not 1 <= top_k <= 100:
            raise ValueError("reference_limit must be 1..12; top_k must be 1..100")
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        self._load()
        items, exclusions = self._gallery
        by_id = {row["item_id"]: row for row in items}
        api_text = (search_text if search_text is not None else description).strip()
        acquirer = DynamicMuseumAcquirer(self.root, output, exclusions, offline=offline,
                                         cache_dir=Path(cache_dir).resolve() if cache_dir else output)
        trace = {"mode": "direct_api_text", "original_description": description,
                 "search_text": api_text,
                 "query_mapping": "explicit caller override" if search_text is not None else "identity; exact original text",
                 "automatic_translation": False, "automatic_query_expansion": False,
                 "metadata_relevance_filters": [], "raw_searches": [], "candidate_decisions": [],
                 "selection_order": "Cleveland then Met, each in raw API hit order; bounded inspection",
                 "metadata_role": "rights, official identity/provenance and duplicate isolation; no form/date taxonomy rejection",
                 "accepted": [], "rejected": []}
        hits = []
        for source in ("cleveland", "met"):
            pages = range(max(1, page_depth)) if source == "cleveland" else range(1)
            for page in pages:
                params = ({"q": api_text, "cc0": 1, "has_image": 1, "limit": per_query,
                           "skip": page * per_query} if source == "cleveland" else
                          {"q": api_text, "hasImages": "true"})
                url = SOURCE_BASES[source] + ("?" + urlencode(params))
                try:
                    data = acquirer.get_json(url, "search", f"direct-{source}-{page}")
                    ids = ([row["id"] for row in data.get("data", [])] if source == "cleveland"
                           else data.get("objectIDs") or [])
                    trace["raw_searches"].append({"source": source, "request_url": url,
                        "api_text": api_text, "returned_hit_ids_in_order": ids,
                        "returned_count": len(ids), "response_info": data.get("info")})
                    hits.extend((source, str(source_id), rank + 1 + page * per_query)
                                for rank, source_id in enumerate(ids))
                except Exception as exc:
                    trace["raw_searches"].append({"source": source, "request_url": url,
                                                  "api_text": api_text, "error": str(exc)})
        seen, seen_families = set(), set()
        for source, source_id, api_rank in hits:
            if len(trace["accepted"]) >= reference_limit or len(seen) >= max_metadata_candidates:
                break
            identity = (source, source_id)
            if identity in seen:
                continue
            seen.add(identity)
            decision = {"source": source, "source_id": source_id, "source_api_rank": api_rank,
                        "image_downloaded": False}
            trace["candidate_decisions"].append(decision)
            try:
                record = acquirer.object(source, source_id)
                row = normalize_record(source, record)
                metadata_path = output / "metadata" / f"{row['reference_id']}.json"
                write_json(metadata_path, {"record": record})
                decision.update(reference_id=row["reference_id"], title=row.get("title"),
                                source_metadata_path=metadata_path.relative_to(self.root).as_posix(),
                                source_date=row.get("date"), source_culture=row.get("culture"),
                                source_medium=row.get("medium"), source_image_url=row["image_url"])
                reason = exclusions.reason(row)
                accession = accession_root(row.get("accession", ""))
                # Cleveland exposes a set and its photographed components as
                # parent / numeric-child accessions. Those are the same source
                # object family, not independent reference evidence. Met's
                # numeric accession components do not have that semantics.
                family = (source, ".".join(accession.split(".")[:2])
                          if source == "cleveland" else accession)
                if reason:
                    decision.update(status="excluded", exclusion=reason)
                    trace["rejected"].append({"reference_id": row["reference_id"], **reason})
                    continue
                if family[1] and family in seen_families:
                    decision.update(status="excluded", exclusion={"reason": "same_reference_accession_family"})
                    continue
                binary, request = acquirer.get_image(row, f"direct-img-{row['reference_id']}")
                decision["image_downloaded"] = True
                row.update(image_sha256=hashlib.sha256(binary).hexdigest(), image_request=request)
                with Image.open(io.BytesIO(binary)) as opened:
                    image = ImageOps.exif_transpose(opened).convert("RGB")
                    if min(image.size) < 128 or max(image.size) < 256:
                        decision.update(status="excluded", exclusion={"reason": "below_usable_resolution", "size": list(image.size)})
                        continue
                    row.update(image_fingerprints(image))
                    row.update(width=image.width, height=image.height)
                reason = exclusions.reason(row)
                if reason:
                    decision.update(status="excluded", exclusion=reason)
                    trace["rejected"].append({"reference_id": row["reference_id"], **reason})
                    continue
                duplicate = next((prior for prior in trace["accepted"]
                    if prior["image_sha256"] == row["image_sha256"]
                    or (int(prior["phash64"], 16) ^ int(row["phash64"], 16)).bit_count() <= 6
                    or sum(a != b for a, b in zip(prior["ahash256"], row["ahash256"])) <= 4), None)
                if duplicate:
                    decision.update(status="excluded", exclusion={"reason": "duplicate_reference_photo", "duplicate_of": duplicate["reference_id"]})
                    continue
                image_path = output / "images" / f"{row['reference_id']}.jpg"
                image_path.write_bytes(binary)
                row.update(image_path=image_path.relative_to(self.root).as_posix(),
                           metadata_path=metadata_path.relative_to(self.root).as_posix(),
                           original_metadata=record, source_api_rank=api_rank)
                decision.update(status="selected", image_path=row["image_path"], image_sha256=row["image_sha256"])
                trace["accepted"].append(row)
                seen_families.add(family)
            except Exception as exc:
                decision.update(status="unusable", error=str(exc))
        acquirer.save_manifest(output / "http_requests.jsonl")
        trace["request_summary"] = {"total": len(acquirer.request_log),
            "live": sum(not row["from_cache"] for row in acquirer.request_log),
            "cached": sum(row["from_cache"] for row in acquirer.request_log)}
        trace["inspection_budget"] = {"max_metadata_candidates": max_metadata_candidates,
                                       "inspected": len(seen), "reference_limit": reference_limit}
        selected = trace["accepted"]
        write_json(output / "direct_acquisition.json", trace)
        cells = [(f"SELECTED {row['reference_id']}", self.root / row["image_path"]) for row in selected]
        excluded = {row.get("reference_id") for row in trace["rejected"]}
        cells += [(f"ISOLATED {key}", self.root / by_id[key]["image_path"])
                  for key in sorted(excluded & by_id.keys())[:6]]
        if cells:
            make_sheet(output / "reference_review.jpg", "Raw API usable photos / sampled isolated target objects", cells)
        result = {"runtime_version": VERSION, "mode": "direct_api_text", "description": description,
                  "original_description": description, "search_text": api_text,
                  "status": "ranked" if selected else "no_usable_external_reference",
                  "evidence_level": "no_external_reference" if not selected else
                                    "single_api_reference" if len(selected) == 1 else "multiple_api_references",
                  "reference_count": len(selected), "reference_ids": [row["reference_id"] for row in selected],
                  "reference_metadata_validation": "rights/official identity only; API relevance and source attribution retained as provenance",
                  "request_summary": trace["request_summary"], "gallery_count": len(items), "results": [],
                  "limits": ["Raw API text relevance may be broad or wrong; no hand-enforced era/form taxonomy filter is applied.",
                             "Image similarity cannot certify dates, authenticity or precise technique.",
                             "Single/sparse reference photographs may bias photographed form and composition."],
                  "created_at": datetime.now(timezone.utc).isoformat()}
        if selected:
            sr, dr, encoders = self._encode(selected, output)
            siglip, dino = self._indices
            ids = siglip[1]
            scored = score_gallery(ids, siglip[0], dino[0], sr, dr)
            result.update(encoders=encoders, consensus=scored["consensus"],
                          assistance={"status": "disabled_direct_mode", "mask_prerequisite": False},
                          ranking_inputs=["reference image embeddings", "gallery image embeddings"],
                          index_versions={"siglip": siglip[2]["index_sha256"], "dino": dino[2]["index_sha256"]},
                          score_semantics="image cosine and disclosed reference/rank aggregation; not probability",
                          rankings={name: [ids[i] for i in order] for name, order in scored["orders"].items()})
            for rank, index in enumerate(scored["orders"]["fusion"][:top_k], 1):
                result["results"].append({"item_id": ids[index], "rank": rank,
                    "score": float(scored["channels"]["fusion"][index]),
                    "channels": {name: float(values[index]) for name, values in scored["channels"].items()},
                    "reference_cosines": {"siglip": scored["siglip_pairs"][index].tolist(),
                                          "dino": scored["dino_pairs"][index].tolist()}})
            np.savez_compressed(output / "gallery_scores.npz", item_ids=np.array(ids),
                                **scored["channels"], siglip_pairs=scored["siglip_pairs"], dino_pairs=scored["dino_pairs"])
            for method in ("siglip_mean", "dino_mean", "fusion"):
                cells = [(f"{rank + 1} {ids[i]}", self.root / by_id[ids[i]]["image_path"])
                         for rank, i in enumerate(scored["orders"][method][:top_k])]
                make_sheet(output / f"results_{method}.jpg", f"Direct API photos -> {method}; gallery pixels only", cells)
        write_json(output / "result.json", result)
        return result

    def run_curated(self, description: str, output: Path | str, *, cache_dir: Path | str | None = None,
            offline: bool = False, acquisition: Path | str | None = None,
            reference_limit: int = 8, per_query: int = 20, page_depth: int = 2,
            max_metadata_candidates: int = 150, top_k: int = 24) -> dict:
        if not 1 <= reference_limit <= 12 or not 1 <= top_k <= 100:
            raise ValueError("reference_limit must be 1..12; top_k must be 1..100")
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        self._load()
        items, exclusions = self._gallery
        by_id = {row["item_id"]: row for row in items}
        plan = plan_dynamic_query(description)
        # Unknown concepts are exposed. Arbitrary English can be discovered,
        # but a no-constraint source hit cannot count as a validated reference.
        if not plan["concept_keys"]:
            result = {"status": "unsupported_description", "plan": plan, "results": [],
                      "reason": "No supported visual identity constraint; add a form/material/decoration."}
            write_json(output / "result.json", result)
            return result
        if acquisition:
            acquired = json.loads(Path(acquisition).read_text(encoding="utf-8"))
            old_description = acquired.get("plan", {}).get("input_query")
            if old_description != description:
                raise ValueError("Acquisition replay must match the exact input description")
            acquired["replay_source"] = str(Path(acquisition).resolve())
            acquired["plan"] = plan
        else:
            acquirer = DynamicMuseumAcquirer(self.root, output, exclusions, offline=offline,
                                             cache_dir=Path(cache_dir).resolve() if cache_dir else output)
            acquired = acquirer.fetch("description", plan, per_query=per_query,
                                      page_depth=page_depth, max_metadata_candidates=max_metadata_candidates)
            acquirer.save_manifest(output / "http_requests.jsonl")
            acquirer.save_discovery(output / "discovery_diagnostics.jsonl")
            write_json(output / "acquisition_failures.json", acquirer.failures)
            acquired["request_summary"] = {"total": len(acquirer.request_log),
                "live": sum(not row["from_cache"] for row in acquirer.request_log),
                "cached": sum(row["from_cache"] for row in acquirer.request_log)}
        selected, rejected, parts = [], [], []
        # Revalidate old acquisitions against the current strict planner and
        # image bytes; cached selection_status is never a trust boundary.
        rows = []
        for row in acquired["accepted"]:
            metadata_path = self.root / row["metadata_path"]
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["record"]
            normalized = normalize_record(row["source"], metadata)
            if normalized["reference_id"] != row["reference_id"]:
                raise ValueError("Reference metadata identity differs from acquisition")
            if hashlib.sha256((self.root / row["image_path"]).read_bytes()).hexdigest() != row["image_sha256"]:
                raise ValueError("Reference image bytes differ from recorded acquisition")
            score, diagnostics = reference_metadata_score(metadata, row["source"], plan)
            rows.append({**row, **normalized, "metadata_score": score, "metadata_diagnostics": diagnostics})
        for row in select_reference_set(rows, plan, limit=12):
            path = self.root / row["image_path"]
            with Image.open(path) as image:
                current = {**row, "image_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                           **image_fingerprints(image)}
            exclusion = exclusions.reason(current)
            if exclusion:
                rejected.append({**current, "runtime_rejection": exclusion})
                continue
            evidence, extracted = visual_reference_evidence(path, plan)
            current["visual_evidence"] = evidence
            if evidence["status"] == "contradicted":
                rejected.append({**current, "runtime_rejection": "visible_constraint_contradiction"})
                continue
            selected.append(current)
            parts.append(extracted)
            if len(selected) >= reference_limit:
                break
        acquired.update(selected=selected, runtime_rejected=rejected)
        write_json(output / "acquisition.json", acquired)
        reference_cells = [(f"KEEP {row['reference_id']}", self.root / row["image_path"]) for row in selected]
        reference_cells += [(f"REJECT {row['reference_id']}", self.root / row["image_path"]) for row in rejected]
        # Explicitly review some excluded local-object photographs; these never
        # enter reference vectors. This also documents sparse-source failures.
        excluded_ids = {row.get("reference_id") for row in acquired.get("rejected", [])}
        reference_cells += [(f"ISOLATED {key}", self.root / by_id[key]["image_path"])
                            for key in sorted(excluded_ids & by_id.keys())[:6]]
        if reference_cells:
            make_sheet(output / "reference_review.jpg", "Retained / contradicted / target-isolated references", reference_cells)
        evidence_level = ("no_external_reference" if not selected else
                          "single_source_qualified" if len(selected) == 1 else "multiple_source_qualified")
        result = {"runtime_version": VERSION, "description": description, "plan": plan,
                  "status": "ranked" if selected else "no_qualified_reference",
                  "evidence_level": evidence_level, "reference_count": len(selected),
                  "reference_ids": [row["reference_id"] for row in selected],
                  "reference_visual_support": [row["visual_evidence"]["status"] for row in selected],
                  "description_coverage": {
                      "planner": "literal bilingual concept/era rules; no language model",
                      "recognized_visual_concepts": plan["concept_keys"],
                      "recognized_eras": [row["label"] for row in plan["era_constraints"]],
                      "qualifiers": "Only these recognized constraints are enforced on reference metadata; other wording is not independently validated."},
                  "reference_composition": [
                      {"reference_id": row["reference_id"], "width": row.get("width"),
                       "height": row.get("height"),
                       "wide_composition": row.get("width", 0) > 3 * row.get("height", 1)}
                      for row in selected],
                  "gallery_count": len(items), "results": [],
                  "limits": ["Image similarity cannot certify period, provenance, authenticity or precise technique.",
                             "Reference agreement is not independent validation.",
                             "A sparse reference set can impose its photographed form, color and viewpoint on a broader request.",
                             "No gallery labels/text are supplied to the image scorer."],
                  "created_at": datetime.now(timezone.utc).isoformat()}
        if not selected:
            write_json(output / "result.json", result)
            return result
        sr, dr, encoders = self._encode(selected, output)
        siglip, dino = self._indices
        ids = siglip[1]
        auxiliary = {}
        requested_features = []
        if FORM_KEYS.intersection(plan["concept_keys"]):
            requested_features.append("shape")
        if COLOR_KEYS.intersection(plan["concept_keys"]):
            requested_features.append("color")
        if requested_features:
            if self._parts is None:
                self._parts = load_visual_parts(self.root / "artifacts/visual_parts")
            if self._parts["manifest"]["data_sha256"] != siglip[2]["data_sha256"]:
                raise ValueError("Visual parts gallery differs from image index")
            lookup = {key: i for i, key in enumerate(self._parts["item_ids"])}
            positions = np.array([lookup[key] for key in ids])
            for feature in requested_features:
                usable = [value[feature] for value in parts if value["validity"][feature]]
                if not usable:
                    continue
                valid = self._parts["features"][f"valid_{feature}"][positions].astype(bool)
                values = np.full(len(ids), np.nan, dtype=np.float32)
                gallery = self._parts["features"][feature][positions][valid]
                values[valid] = (unit_rows(gallery) @ unit_rows(np.stack(usable)).T).mean(axis=1)
                auxiliary[feature] = values
        scored = score_gallery(ids, siglip[0], dino[0], sr, dr, auxiliary=auxiliary)
        result.update(consensus=scored["consensus"], assistance=scored["assistance"], encoders=encoders,
                      index_versions={"siglip": siglip[2]["index_sha256"], "dino": dino[2]["index_sha256"]},
                      ranking_inputs=["reference image embeddings", "gallery image embeddings",
                                      "valid foreground shape/color descriptors"],
                      score_semantics="rank fusion and bounded auxiliary slot reranking; not probability")
        result["rankings"] = {name: [ids[i] for i in order] for name, order in scored["orders"].items()}
        for rank, index in enumerate(scored["orders"]["assisted"][:top_k], 1):
            result["results"].append({"item_id": ids[index], "rank": rank,
                "baseline_score": float(scored["channels"]["fusion"][index]),
                "channels": {name: float(values[index]) for name, values in scored["channels"].items()},
                "auxiliary": {name: float(values[index]) if np.isfinite(values[index]) else None
                              for name, values in auxiliary.items()},
                "reference_cosines": {"siglip": scored["siglip_pairs"][index].tolist(),
                                      "dino": scored["dino_pairs"][index].tolist()}})
        np.savez_compressed(output / "gallery_scores.npz", item_ids=np.array(ids),
                            **scored["channels"], **{f"aux_{key}": value for key, value in auxiliary.items()},
                            siglip_pairs=scored["siglip_pairs"], dino_pairs=scored["dino_pairs"])
        for method in ("siglip_mean", "dino_mean", "fusion", "assisted"):
            cells = [(f"{rank + 1} {ids[i]}", self.root / by_id[ids[i]]["image_path"])
                     for rank, i in enumerate(scored["orders"][method][:top_k])]
            make_sheet(output / f"results_{method}.jpg", f"{method}; image-only gallery ranking", cells)
        write_json(output / "result.json", result)
        return result
