"""Reference image matching over image features, with auditable specialist ablations.

Catalogue words select external reference photographs only. Local gallery
ranking never reads catalogue text, dates, or category tags.
"""

import hashlib
import json
from pathlib import Path
import threading
import time
import uuid
from collections import OrderedDict

import numpy as np


METHODS = ("siglip", "dinov2", "dinov2_patch", "shape", "color", "texture", "siglip_shape", "siglip_color", "siglip_texture", "ensemble")


def unit_rows(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Expected finite nonempty feature matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms < 1e-10):
        raise ValueError("Cannot normalize zero feature")
    return values / norms


def reference_similarity(gallery, references, aggregation="mean"):
    """Compare each prototype separately; preserve multimodal reference sets."""
    gallery, references = unit_rows(gallery), unit_rows(references)
    if gallery.shape[1] != references.shape[1]:
        raise ValueError("Reference/gallery feature dimensions differ")
    similarities = gallery @ references.T
    if aggregation == "mean":
        return similarities.mean(axis=1), similarities
    if aggregation == "max":
        return similarities.max(axis=1), similarities
    if aggregation == "centroid":
        centroid = unit_rows(references.mean(axis=0, keepdims=True))[0]
        return gallery @ centroid, similarities
    raise ValueError("aggregation must be mean, max, or centroid")


class ImageSearchService:
    def __init__(self, service, *, bank_dir=None, parts_dir=None, dino_dir=None):
        self.service = service
        self.root = service.root
        self.bank_dir = Path(bank_dir or self.root / "data/references")
        self.parts_dir = Path(parts_dir or self.root / "artifacts/visual_parts")
        self.dino_dir = Path(dino_dir or self.root / "artifacts/dinov2")
        self._parts = None
        self._dino = None
        self._dino_vectors = None
        self._dino_ids = None
        self._dino_manifest = None
        self._patch_matcher = None
        self._patch_cache = OrderedDict()
        self._lock = threading.Lock()

    def _siglip(self, images):
        # Reuse the exact image tower/adapter of the loaded text-image index.
        if self.service.embeddings is None:
            raise RuntimeError("Image index unavailable; no catalogue fallback")
        if self.service.encoder is None:
            self.service._dense("image reference initialization")
        if not self.service.encoder or not hasattr(self.service.encoder, "encode_images"):
            raise RuntimeError("Matching image encoder unavailable")
        with self.service._encoder_lock:
            return unit_rows(self.service.encoder.encode_images(images))

    def _dinov2(self, images):
        with self._lock:
            if self._dino is None:
                from curio.represent.image_encoder import DinoEncoder
                manifest = json.loads((self.dino_dir / "manifest.json").read_text(encoding="utf-8"))
                if manifest.get("data_sha256") != self.service.manifest.get("data_sha256"):
                    raise ValueError("DINO index/catalogue mismatch")
                self._dino_vectors = unit_rows(np.load(self.dino_dir / "image_embeddings.npy", allow_pickle=False))
                self._dino_ids = json.loads((self.dino_dir / "item_ids.json").read_text(encoding="utf-8"))
                # Validate before normalization, since float renormalization may alter bytes.
                raw = np.load(self.dino_dir / "image_embeddings.npy", allow_pickle=False)
                fingerprint = hashlib.sha256(raw.tobytes() + json.dumps(self._dino_ids).encode()).hexdigest()
                if fingerprint != manifest.get("index_sha256") or set(self._dino_ids) != set(self.service.by_id):
                    raise ValueError("DINO index fingerprint or IDs mismatch")
                encoder = DinoEncoder()
                if encoder.metadata["revision"] != manifest.get("encoder_metadata", {}).get("revision"):
                    raise ValueError("DINO checkpoint revision mismatch")
                self._dino = encoder
                self._dino_manifest = manifest
            return self._dino.encode_images(images), self._dino_vectors, self._dino_ids

    def _specialist(self, images, feature):
        from curio.represent.visual_parts import extract_visual_parts, load_visual_parts
        if self._parts is None:
            self._parts = load_visual_parts(self.parts_dir)
            if self._parts["manifest"].get("data_sha256") != self.service.manifest.get("data_sha256"):
                self._parts = None
                raise ValueError("Specialist index/catalogue mismatch")
        return [extract_visual_parts(image) for image in images], self._parts

    def search(self, images, *, top_k=12, method="siglip", aggregation="mean", exclude_ids=None):
        if method not in METHODS:
            raise ValueError(f"Unknown image method {method}")
        if not images or not 1 <= len(images) <= 12:
            raise ValueError("Use 1..12 reference images")
        if not 1 <= top_k <= 50:
            raise ValueError("top_k must be 1..50")
        if method == "dinov2_patch":
            return self._patch_search(images, top_k=top_k, aggregation=aggregation, exclude_ids=exclude_ids)
        started = time.perf_counter()
        if self.service.embeddings is None:
            raise RuntimeError("Image index unavailable; no catalogue fallback")
        ids = [str(self.service.items[row]["item_id"]) for row in self.service.vector_rows]
        excluded = set(exclude_ids or [])
        channels, matrices, warnings = {}, {}, []
        if method.startswith("siglip") or method == "ensemble":
            vectors = self._siglip(images)
            channels["siglip"], matrices["siglip"] = reference_similarity(self.service.embeddings, vectors, aggregation)
        if method in {"dinov2", "ensemble"}:
            vectors, gallery, dino_ids = self._dinov2(images)
            raw, details = reference_similarity(gallery, vectors, aggregation)
            lookup = {key: row for row, key in enumerate(dino_ids)}
            positions = [lookup[key] for key in ids]
            channels["dinov2"], matrices["dinov2"] = raw[positions], details[positions]
        feature = method.removeprefix("siglip_")
        if feature in {"shape", "color", "texture"}:
            extracted, parts = self._specialist(images, feature)
            # load_visual_parts returns a feature dictionary with parallel IDs.
            valid_refs = [row for row in extracted if row["validity"][feature]]
            if not valid_refs:
                if not channels:
                    raise ValueError(f"Reference {feature} extraction rejected: insufficient foreground/mask evidence")
                warnings.append(f"Reference {feature} rejected; using full-image channel only")
            else:
                features, part_ids = parts["features"], parts["item_ids"]
                lookup = {key: row for row, key in enumerate(part_ids)}
                positions = np.array([lookup[key] for key in ids])
                gallery = features[feature][positions]
                valid = features[f"valid_{feature}"][positions].astype(bool)
                vectors = np.stack([row[feature] for row in valid_refs])
                values = np.full(len(ids), np.nan, dtype=np.float32)
                if valid.any():
                    values[valid], _ = reference_similarity(gallery[valid], vectors, aggregation)
                channels[feature] = values
                warnings.append("Specialist features are classical mask/color/texture descriptors; no learned motif verification")
                warnings.append(f"{feature}: {len(valid_refs)}/{len(images)} references and {int(valid.sum())}/{len(ids)} gallery images accepted; unknown masks are not verified negatives")
        if not channels:
            raise ValueError("No usable visual channel")
        # Single channel keeps actual cosine. Mixed channels use disclosed
        # rank fusion so unlike feature scales never masquerade as probability.
        if len(channels) == 1:
            final = next(iter(channels.values())).copy()
            scoring = "cosine_similarity"
        else:
            final = np.zeros(len(ids), dtype=np.float32)
            dense_name = "siglip" if "siglip" in channels else next(iter(channels))
            for name, values in channels.items():
                weight = .75 if name == dense_name else .25 / (len(channels) - 1)
                finite = np.flatnonzero(np.isfinite(values))
                ranks = self.service._average_ranks(finite, values, np.isfinite)
                for idx, rank in ranks.items():
                    final[idx] += weight * 61 / (60 + rank)
            scoring = "research_rrf_0.75_0.25"
        order = sorted((row for row in range(len(ids)) if ids[row] not in excluded and np.isfinite(final[row])),
                       key=lambda row: (-float(final[row]), ids[row]))[:top_k]
        results = []
        for rank, row in enumerate(order, 1):
            key = ids[row]
            item = self.service.by_id[key]
            results.append({"item_id": key, "rank": rank, "score": float(final[row]), "title": item.get("title", ""),
                            "source": item.get("source"), "source_url": item.get("source_url"),
                            "image_url": f"/api/images/{key}", "thumb_url": f"/api/images/{key}?thumb=true",
                            "matched_slots": [], "evidence": [{"kind": "image_similarity", "text": "参考图片与图库像素特征相似；未验证年代、真实性或具体纹饰。"}],
                            "score_breakdown": {name: float(values[row]) if np.isfinite(values[row]) else None for name, values in channels.items()},
                            "reference_similarities": {name: [float(value) for value in matrix[row]] for name, matrix in matrices.items()}})
        primary = "siglip" if "siglip" in channels else next(iter(channels))
        primary_rows = [idx for idx in range(len(ids)) if ids[idx] not in excluded and np.isfinite(channels[primary][idx])]
        primary_top = ids[max(primary_rows, key=lambda idx: float(channels[primary][idx]))] if primary_rows else None
        finite = sorted((float(value) for idx, value in enumerate(channels[primary])
                         if ids[idx] not in excluded and np.isfinite(value)), reverse=True)
        agreement = {"status": "not_measured", "compared": 0, "agreeing": 0}
        reference_diagnostics = {}
        if primary in matrices and len(images) >= 2 and order:
            matrix = matrices[primary]
            eligible = [idx for idx in range(len(ids)) if ids[idx] not in excluded]
            individual_tops = [ids[eligible[int(np.argmax(matrix[eligible, ref]))]] for ref in range(matrix.shape[1])]
            winner = ids[max(eligible, key=lambda idx: float(channels[primary][idx]))]
            agreeing = sum(key == winner for key in individual_tops)
            agreement = {"status": "agree" if agreeing == len(images) else "disagree" if not agreeing else "mixed",
                         "compared": len(images), "agreeing": agreeing, "independent_validation": False,
                         "definition": "aggregate_primary_winner_also_top_for_each_reference"}
            reference_diagnostics = {"primary_channel": primary, "individual_reference_top_ids": individual_tops,
                                     "note": "References can agree on an incorrect result; disagreement may reflect genuinely diverse views."}
        confidence = {"level": "unknown", "label": "未校准",
                      "scope": "reference_image_before_fusion", "dense_top_item_id": primary_top,
                      "final_top_item_id": ids[order[0]] if order else None,
                      "ranking_separation": {"score_space": "cosine_similarity", "candidate_count": len(finite),
                                             "top1": finite[0] if finite else None, "top2": finite[1] if len(finite) > 1 else None,
                                             "top1_top2_margin": finite[0] - finite[1] if len(finite) > 1 else None},
                      "signal_agreement": agreement, "calibration": {"status": "uncalibrated", "correctness_probability": None},
                      "basis": "图图相似度及多参考一致性不等于专业属性正确率；研究融合权重尚未验证。"}
        channel_provenance = {}
        if "siglip" in channels:
            channel_provenance["siglip"] = {"encoder": self.service.manifest.get("encoder_metadata"),
                                             "index_sha256": self.service.index_sha256,
                                             "index_version": self.service.index_version}
        if "dinov2" in channels:
            channel_provenance["dinov2"] = {"encoder": self._dino.metadata,
                                            "index_sha256": self._dino_manifest["index_sha256"],
                                            "index_version": self._dino_manifest["index_version"]}
        for name in {"shape", "color", "texture"}.intersection(channels):
            channel_provenance[name] = {"method": self._parts["manifest"]["version"],
                                        "index_sha256": self._parts["manifest"]["index_sha256"],
                                        "learned_segmentation": False}
        model_version = (self.service.model_version if "siglip" in channels else
                         self._dino.metadata["revision"] if "dinov2" in channels else self._parts["manifest"]["version"])
        index_version = self.service.index_version if "siglip" in channels else self._dino_manifest["index_version"] if "dinov2" in channels else self._parts["manifest"]["index_sha256"][:16]
        return {"request_id": str(uuid.uuid4()), "query": "image reference", "variant": "reference_image", "effective_variant": method,
                "retrieval_mode": "image_to_image", "reference_count": len(images), "aggregation": aggregation,
                "gallery_count": len(ids), "scoring": scoring, "results": results, "warnings": warnings,
                "model_version": model_version, "index_version": index_version, "channel_provenance": channel_provenance,
                "timings": {"total_ms": round((time.perf_counter() - started) * 1000, 3)},
                "reference_diagnostics": reference_diagnostics, "confidence": confidence}

    def _patch_search(self, images, *, top_k, aggregation, exclude_ids):
        """Recall100 via DINO CLS, then rerank only reliable patch candidates."""
        from curio.represent.patch_match import DinoPatchMatcher, match_patch_features
        started = time.perf_counter()
        baseline = self.search(images, top_k=top_k, method="dinov2", aggregation=aggregation, exclude_ids=exclude_ids)
        vectors, gallery, ids = self._dinov2(images)
        scores, _ = reference_similarity(gallery, vectors, aggregation)
        excluded = set(exclude_ids or [])
        recall = sorted((idx for idx in range(len(ids)) if ids[idx] not in excluded),
                        key=lambda idx: (-float(scores[idx]), str(ids[idx])))[:100]
        candidate_ids = [ids[idx] for idx in recall]
        with self._lock:
            if self._patch_matcher is None:
                self._patch_matcher = DinoPatchMatcher(encoder=self._dino)
            reference_features = self._patch_matcher.encode_images(images)
            missing = [key for key in candidate_ids if key not in self._patch_cache]
            paths = [self.root / self.service.by_id[key]["image_path"] for key in missing]
            for key, feature in zip(missing, self._patch_matcher.encode_images(paths)):
                self._patch_cache[key] = feature
            features = [self._patch_cache[key] for key in candidate_ids]
            for key in candidate_ids:
                self._patch_cache.move_to_end(key)
            while len(self._patch_cache) > 512:
                self._patch_cache.popitem(last=False)
        details = []
        for feature in features:
            pairs = [match_patch_features(ref, feature) for ref in reference_features]
            valid = [row for row in pairs if row["status"] == "available"]
            details.append({"status": "available" if valid else "unknown",
                            "combined_score": float(np.mean([r["combined_score"] for r in valid])) if valid else None,
                            "accepted_references": len(valid)})
        # Unknown segmentation does not become a relevance penalty: only
        # accepted candidates exchange their original CLS ranking slots.
        slots = [idx for idx, row in enumerate(details) if row["status"] == "available"]
        accepted = sorted(slots, key=lambda idx: (-details[idx]["combined_score"], -float(scores[recall[idx]]), candidate_ids[idx]))
        order = list(range(len(recall)))
        for slot, idx in zip(slots, accepted):
            order[slot] = idx
        results = []
        for rank, idx in enumerate(order[:top_k], 1):
            key = candidate_ids[idx]
            item = self.service.by_id[key]
            results.append({"item_id": key, "rank": rank, "score": float(1 / rank),
                            "title": item.get("title", ""), "source": item.get("source"), "source_url": item.get("source_url"),
                            "image_url": f"/api/images/{key}", "thumb_url": f"/api/images/{key}?thumb=true", "matched_slots": [],
                            "score_breakdown": {"dinov2_cls": float(scores[recall[idx]]), "local_patch_score": details[idx]["combined_score"]},
                            "patch_evidence": details[idx], "evidence": [{"kind": "image_similarity", "text": "DINO 图像召回后比较前景 patch 与软轮廓；未知前景保留原位置，不核验年代或具体纹饰。"}]})
        baseline.update(results=results, effective_variant="dinov2_patch", scoring="accepted_patch_rerank_with_unknown_slots_preserved",
                        patch_diagnostics={"recall_count": len(recall), "accepted_candidates": len(accepted),
                                           "patch_aggregation": "mean_of_available_reference_pair_scores",
                                           "accepted_references": sum(r["diagnostics"]["status"] == "valid" for r in reference_features),
                                           "total_references": len(images), "status": "reranked" if accepted else "fallback_cls",
                                           "metadata": self._patch_matcher.metadata})
        baseline["confidence"]["scope"] = "dinov2_cls_recall_before_patch_rerank"
        baseline["confidence"]["final_top_item_id"] = results[0]["item_id"] if results else None
        baseline["warnings"].append("局部匹配为冻结 DINO patch 加启发式前景的研究方法；不是学习式分割，也未训练语义纹饰识别。")
        if not accepted:
            baseline["warnings"].append("本次前景证据不足，明确保留 DINO 全图排序。")
        baseline["timings"]["total_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return baseline

    def search_query(self, query, *, reference_ids=None, top_k=12, method="siglip", aggregation="mean", limit=4):
        from curio.retrieve.references import load_reference_bank, select_references
        from curio.query import parse_query
        started = time.perf_counter()
        if reference_ids:
            bank = load_reference_bank(self.bank_dir)
            lookup = {row["reference_id"]: row for row in bank}
            if set(reference_ids) - lookup.keys():
                raise ValueError("Unknown reference ID")
            selected = {"plan": {"planner": "explicit_references", "query": query}, "references": [lookup[key] for key in reference_ids]}
        else:
            selected = select_references(query, self.root, limit=limit)
        rows = selected["references"]
        if not rows:
            return {"query": query, "results": [], "variant": "reference_image", "retrieval_mode": "reference_unavailable",
                    "plan": selected["plan"], "references": [],
                    "timings": {"total_ms": round((time.perf_counter() - started) * 1000, 3)},
                    "warnings": ["没有符合查询年代、器型等已映射条件的馆外参考图；未放宽条件或自动改用文本检索。"]}
        paths = []
        for row in rows:
            path = (self.root / row["image_path"]).resolve()
            if not path.is_relative_to(self.bank_dir.resolve()) or not path.is_file():
                raise ValueError("Reference image is outside verified reference bank")
            if row.get("image_sha256") and hashlib.sha256(path.read_bytes()).hexdigest() != row["image_sha256"]:
                raise ValueError("Reference image fingerprint changed")
            paths.append(path)
        response = self.search(paths, top_k=top_k, method=method, aggregation=aggregation)
        response.update(query=query, plan=selected["plan"], references=[{key: value for key, value in row.items() if key != "image_path"} for row in rows])
        response["reference_constraint_status"] = selected.get("constraint_status", "explicit_unverified")
        if selected.get("qualifications"):
            response["reference_qualifications"] = selected["qualifications"]
            response["reference_constraint_status"] = "qualified_metadata"
            response["warnings"].append("来源记录写釉下蓝，但参考照片含绿色彩绘；未由来源明确确认“五彩”工艺归属，也不视为纯青花。")
        parsed = parse_query(query)
        unsupported = [{"slot": key, "value": value, "status": "unknown"}
                       for key, values in parsed.slots.items() if key not in {"object_type", "material"}
                       for value in values]
        unsupported += [{"slot": "unsupported", "value": value, "status": "unknown"} for value in parsed.unsupported]
        response["unverified_query_constraints"] = unsupported
        if unsupported or parsed.negations:
            response["warnings"].append("参考图规划只约束已映射的类别、器型和来源年代；复杂纹饰、否定、数量及未映射专名尚未逐项核验。")
        response["timings"]["image_matching_ms"] = response["timings"]["total_ms"]
        response["timings"]["total_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return response
