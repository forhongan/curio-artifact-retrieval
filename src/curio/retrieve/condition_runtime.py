"""Index-bound condition boundary, with bounded prompt cache and explicit fallback."""

from collections import OrderedDict
import hashlib
import json
from pathlib import Path
import threading

import numpy as np

from .condition_boundary import (
    CONDITIONS,
    FEATURE_NAMES,
    condition_features,
    plan_texts,
    prompt_plan,
    support_scores,
)
from .condition_parser import parse_conditions

DEFAULT_BUNDLE = Path(__file__).with_name("condition_boundary_v7.json")


def vocabulary_sha256():
    return hashlib.sha256(
        json.dumps([CONDITIONS, FEATURE_NAMES], sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


class ConditionBoundary:
    def __init__(self, service, bundle_path=DEFAULT_BUNDLE):
        self.service = service
        self.bundle_path = Path(bundle_path)
        self._bundle = None
        self._text_vectors = {}
        self._cache = OrderedDict()
        self._lock = threading.Lock()

    def _load(self):
        if self._bundle is None:
            bundle = json.loads(self.bundle_path.read_text(encoding="utf-8"))
            if bundle.get("schema") != "condition-boundary-v7":
                raise ValueError("Unsupported condition boundary bundle schema")
            if bundle.get("vocabulary_sha256") != vocabulary_sha256():
                raise ValueError("Condition vocabulary differs from fitted boundary")
            if bundle.get("feature_names") != FEATURE_NAMES:
                raise ValueError("Condition feature schema differs from fitted boundary")
            for name in ["conditions", "refit_base"]:
                width = 9 if name == "conditions" else 3
                for lang in ["en", "zh"]:
                    model = bundle["models"].get(f"{name}/{lang}", {})
                    for field in ["mean", "scale", "coefficients"]:
                        values = np.asarray(model.get(field), dtype=float)
                        if values.shape != (width,) or not np.isfinite(values).all():
                            raise ValueError("Invalid condition model parameters")
                    if np.any(np.array(model["scale"]) <= 0) or not np.isfinite(model["intercept"]):
                        raise ValueError("Invalid condition model scaling/intercept")
                    if not 0 <= model["threshold"] <= 1:
                        raise ValueError("Invalid condition threshold")
            if "text_cache" in bundle:
                spec = bundle["text_cache"]
                if Path(spec["file"]).name != spec["file"]:
                    raise ValueError("Invalid frozen text cache path")
                path = self.bundle_path.parent / spec["file"]
                if hashlib.sha256(path.read_bytes()).hexdigest() != spec["sha256"]:
                    raise ValueError("Frozen text cache changed after export")
                with np.load(path, allow_pickle=False) as cache:
                    texts, vectors = cache["texts"].tolist(), cache["vectors"]
                if len(texts) != len(set(texts)) or vectors.shape != (
                    len(texts),
                    self.service.embeddings.shape[1],
                ):
                    raise ValueError("Invalid frozen text cache dimensions or duplicate prompts")
                if not np.isfinite(vectors).all() or not np.allclose(
                    np.linalg.norm(vectors, axis=1), 1, atol=1e-5
                ):
                    raise ValueError("Invalid frozen text vectors")
                self._text_vectors = dict(zip(texts, vectors))
            self._bundle = bundle
        if self.service.index_sha256 != self._bundle["index_sha256"]:
            raise ValueError("Condition boundary belongs to a different image index")
        if len(self.service.items) != self._bundle["gallery_count"]:
            raise ValueError("Condition boundary needs the complete fitted gallery")
        return self._bundle

    def _score(self, text):
        if text not in self._cache:
            if text in self._text_vectors:
                # The fitted pipeline already normalized these vectors. Reuse
                # their bytes; re-encoding across devices can move edge cases.
                score = np.empty(len(self.service.items), dtype=np.float32)
                score[self.service.vector_rows] = self.service.embeddings @ self._text_vectors[text]
            else:
                score = self.service._dense(text)
            self._cache[text] = score
            if len(self._cache) > 256:
                self._cache.popitem(last=False)
        self._cache.move_to_end(text)
        return self._cache[text]

    def decide(self, query, generic, raw):
        with self._lock:
            bundle = self._load()
            metadata = getattr(self.service.encoder, "metadata", {})
            for field, expected in bundle["text_encoder"].items():
                if metadata.get(field) != expected:
                    raise ValueError(f"Condition text encoder {field} differs from fitted model")
            parsed = parse_conditions(query)
            plan = prompt_plan(parsed["conditions"], parsed["language"])
            fallback = bool(parsed["fallback_reasons"]) or not plan["groups"]
            effective = "refit_base" if fallback else "conditions"
            model = bundle["models"][f"{effective}/{parsed['language']}"]
            # Unknown syntax never becomes a fabricated positive clause.
            used_plan = prompt_plan([], parsed["language"]) if fallback else plan
            if query.strip() in self._text_vectors:
                raw = self._score(query.strip())
            if "a photograph of a museum object" in self._text_vectors:
                generic = self._score("a photograph of a museum object")
            prompt_scores = {}
            for text in dict.fromkeys(plan_texts(used_plan)):
                prompt_scores[text] = self._score(text)
            features = condition_features(raw, generic, used_plan, prompt_scores)
            scores = support_scores(features[:, :3] if fallback else features, model)
            report = {
                "status": "condition_boundary_applied",
                "version": bundle["schema"],
                "effective_model": effective,
                "threshold": model["threshold"],
                "decision_score_semantics": "uncalibrated logistic support",
                "fit_split": "60 dev images; grouped out-of-fold threshold selection",
                "scope": "same-intent fit; query transfer is not certified",
                **parsed,
                "missing_conditions": plan["missing"],
                "supported_conditions": [g["key"] for g in plan["groups"]],
                "raw_query_used": True,
                "frozen_query_vector": query.strip() in self._text_vectors,
                "index_sha256": bundle["index_sha256"],
            }
            notices = []
            if fallback:
                notices.append(
                    "这条描述有暂未支持的条件，已按完整原句匹配；入选不代表逐项条件都已核验。"
                )
            if plan["missing"]:
                notices.append(
                    "部分条件（如年代、工艺）无法仅凭图像确认，入选结果仍需核对馆藏记录。"
                )
            report["notices"] = notices
            return scores >= model["threshold"], scores, raw - generic, report, raw
