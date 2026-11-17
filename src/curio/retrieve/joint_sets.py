"""Whole-gallery asynchronous binary decisions. Partial jobs are never complete sets."""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import threading
import time

from .joint_vlm import LocalVQA

REFERENCE_SEEDS = {
    "form/gourd": [("met_48284", True), ("met_46727", False)],
    "form/meiping": [("met_46727", True), ("met_48284", False)],
}


def binary_decision(score, *, calibration=None):
    if score["reason"] == "color_not_observable":
        return False, 0.0, "color_not_observable"
    probability = float(score["score"])
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Invalid visual score; no binary decision was made")
    threshold = 0.5
    if calibration:
        if not math.isfinite(score["yes_no_logit"]):
            raise ValueError("Invalid visual logit; no binary decision was made")
        logit = calibration["coefficient"] * score["yes_no_logit"] + calibration["intercept"]
        probability = 1 / (1 + math.exp(-max(-60, min(60, logit))))
        threshold = calibration["threshold"]
    yes = probability >= threshold
    return yes, probability, "visual_match" if yes else "insufficient_visual_support"


class JointSetJobs:
    def __init__(self, service, scorer=None, *, method="joint_vlm", pool=None):
        self.service = service
        self.method = method
        self.scorer = scorer or LocalVQA(service.root)
        self.directory = Path(service.root) / (
            "artifacts/joint_vlm_v9/jobs"
            if method == "joint_vlm"
            else "artifacts/catalogue_visual_v15/jobs"
        )
        self.directory.mkdir(parents=True, exist_ok=True)
        self._owns_pool = pool is None
        self.pool = pool or ThreadPoolExecutor(max_workers=1, thread_name_prefix="joint-gallery")
        self.lock = threading.RLock()
        self.jobs = {}
        self.futures = {}

    def _validate_calibration(self, bundle, plan):
        if (
            bundle.get("schema") != "joint-calibration-v1"
            or bundle.get("model") != self.scorer.identity
            or plan.key not in bundle.get("query_keys", [])
        ):
            return None
        for key in ["coefficient", "intercept", "threshold"]:
            if not math.isfinite(bundle[key]):
                raise ValueError("Invalid joint calibration bundle")
        if bundle["coefficient"] <= 0 or not 0 < bundle["threshold"] <= 1:
            raise ValueError("Invalid joint calibration direction or threshold")
        return bundle

    def _reference_adaptation(self, plan):
        directory = self.directory.parent
        bank_path = directory / "reference_bank.json"
        calibration_path = directory / "reference_calibration.json"
        if not bank_path.exists() or not calibration_path.exists():
            return None
        bundle = self._validate_calibration(json.loads(calibration_path.read_text()), plan)
        # A saved research bundle alone must not promote a regressing policy.
        if (
            not bundle
            or not bundle.get("release_approved")
            or not bundle.get("calibration_goal_met")
        ):
            return None
        if (
            not bundle
            or bundle.get("reference_bank_sha256")
            != hashlib.sha256(bank_path.read_bytes()).hexdigest()
        ):
            return None
        references = json.loads(bank_path.read_text())["references"].get(plan.key, [])
        for ref in references:
            item = self.service.by_id.get(ref["item_id"])
            if not item or item["image_path"] != ref["image_path"] or type(ref["yes"]) is not bool:
                raise ValueError("Reference bank does not match this gallery")
        return references, bundle

    def _calibration(self, plan, references):
        if self.method == "catalogue_visual":
            return None  # This conjunction uses its own per-condition rules.
        path = Path(self.service.root) / "artifacts/joint_vlm_v9/calibration.json"
        if not path.exists() or references:
            return None
        return self._validate_calibration(json.loads(path.read_text()), plan)

    def create(self, query, *, use_references=False):
        plan = self.scorer.plan(query)
        seeds = []
        if use_references:
            for key in plan.clauses:
                seeds.extend(REFERENCE_SEEDS.get(key, []))
        # An added color/material changes reference relevance. Only use these
        # concept exemplars for the plain specialist form query.
        if len(plan.clauses) != 1:
            seeds = []
        references = [
            {"item_id": ident, "image_path": self.service.by_id[ident]["image_path"], "yes": yes}
            for ident, yes in seeds
            if ident in self.service.by_id
        ]
        calibration = self._calibration(plan, references)
        adaptation = self._reference_adaptation(plan) if use_references else None
        if adaptation is not None:
            references, calibration = adaptation
        gallery_stamp = []
        for item in self.service.items:
            path = Path(self.service.root) / item["image_path"]
            stat = path.stat()
            gallery_stamp.append((item["item_id"], stat.st_size, stat.st_mtime_ns))
        signature = {
            "plan": plan.payload(),
            "model": self.scorer.identity,
            "gallery": gallery_stamp,
            "references": references,
            "calibration": calibration,
            "policy": "catalogue-visual-v15"
            if self.method == "catalogue_visual"
            else "joint-set-v1",
        }
        ident = ("cv15_" if self.method == "catalogue_visual" else "") + hashlib.sha256(
            json.dumps(signature, sort_keys=True).encode()
        ).hexdigest()
        with self.lock:
            previous = self.jobs.get(ident)
            if (
                previous
                and previous["status"] in {"failed", "cancelled"}
                and self.futures.get(ident)
                and self.futures[ident].done()
            ):
                del self.jobs[ident]
            if ident not in self.jobs:
                saved = self.directory / (ident + ".json")
                if saved.exists():
                    self.jobs[ident] = json.loads(saved.read_text())
                else:
                    if sum(j["status"] in {"queued", "running"} for j in self.jobs.values()) >= 8:
                        raise RuntimeError("正在排队的全库判别任务较多，请先完成或取消现有任务。")
                    job = {
                        "job_id": ident,
                        "status": "queued",
                        "query": query,
                        "plan": plan.payload(),
                        "processed": 0,
                        "total": len(self.service.items),
                        "matched_count": 0,
                        "complete": False,
                        "results": [],
                        "decisions": [],
                        "started": time.time(),
                        "calibration_applied": bool(calibration),
                        "calibration_goal_met": bool(
                            calibration and calibration.get("calibration_goal_met")
                        ),
                        "adaptation": "fit_reference_bank"
                        if adaptation is not None
                        else "concept_examples"
                        if references
                        else "no_references",
                        "references": references,
                        "experimental": True,
                        "model": self.scorer.identity,
                    }
                    self.jobs[ident] = job
                    self.futures[ident] = self.pool.submit(
                        self._run, ident, plan, references, calibration
                    )
            return self.get(ident)

    def _run(self, ident, plan, references, calibration):
        try:
            with self.lock:
                if self.jobs[ident]["status"] == "cancelled":
                    return
                self.jobs[ident]["status"] = "running"
            for item in self.service.items:
                with self.lock:
                    if self.jobs[ident]["status"] == "cancelled":
                        return

                # Never show an object's reference image when judging that same
                # object, including duplicate views linked by group or hash.
                def same_object(ref):
                    other = self.service.by_id[ref["item_id"]]
                    return ref["item_id"] == item["item_id"] or any(
                        item.get(k) and item[k] == other.get(k)
                        for k in ["object_group", "image_sha256"]
                    )

                refs = [ref for ref in references if not same_object(ref)]
                score = self.scorer.score(item["image_path"], plan, references=refs)
                yes, probability, reason = binary_decision(score, calibration=calibration)
                row = {
                    "item_id": item["item_id"],
                    "yes": yes,
                    "score": probability,
                    "reason": score.get("decision_reason", reason),
                    "raw_vqa_score": score["score"],
                    "cached": score.get("cached", False),
                }
                if "condition_evidence" in score:
                    row["condition_evidence"] = score["condition_evidence"]
                with self.lock:
                    job = self.jobs[ident]
                    job["decisions"].append(row)
                    job["processed"] += 1
                    if yes:
                        job["results"].append(
                            {
                                **row,
                                "title": item.get("title", ""),
                                "image_url": f"/api/images/{item['item_id']}",
                                "thumb_url": f"/api/images/{item['item_id']}?thumb=true",
                            }
                        )
                        job["matched_count"] += 1
            with self.lock:
                job = self.jobs[ident]
                if job["status"] == "cancelled":
                    return
                job.update(
                    status="complete",
                    complete=True,
                    finished=time.time(),
                    latency_ms=round((time.time() - job["started"]) * 1000, 2),
                )
                job["results"].sort(key=lambda r: (-r["score"], r["item_id"]))
                for i, row in enumerate(job["results"]):
                    row["rank"] = i + 1
                target = self.directory / (ident + ".json")
                temp = target.with_suffix(".tmp")
                temp.write_text(json.dumps(job, ensure_ascii=False))
                temp.replace(target)
        except Exception as error:
            with self.lock:
                self.jobs[ident].update(
                    status="failed", complete=False, error=str(error), finished=time.time()
                )

    def get(self, ident, *, include_decisions=False):
        with self.lock:
            if ident not in self.jobs:
                raise KeyError(ident)
            job = self.jobs[ident]
            result = {
                k: v for k, v in job.items() if k not in {"results", "decisions", "references"}
            }
            result["reference_item_ids"] = [r["item_id"] for r in job["references"]]
            result["elapsed_ms"] = round(
                (job.get("finished", time.time()) - job["started"]) * 1000, 2
            )
            if job["complete"]:
                result["elapsed_ms"] = job["latency_ms"]
            if job["complete"]:
                result.update(
                    method=self.method,
                    gallery_count=job["total"],
                    returned=job["matched_count"],
                    rejected_count=job["total"] - job["matched_count"],
                    uncertain_count=0,
                    uncertain_results=[],
                    results=list(job["results"]),
                    matched_item_ids=[r["item_id"] for r in job["results"]],
                    notices=[
                        "馆藏类别与图片共同判定：器类/材质可引用馆藏记录，颜色和口部等外观须图片支持；No 包含证据不足。模型分数不等于正确率。"
                    ]
                    if self.method == "catalogue_visual"
                    else ["图文联合判别实验版：逐图扫描完成；模型分数不等于正确率。"],
                    boundary={
                        "version": "catalogue-visual-v15"
                        if self.method == "catalogue_visual"
                        else "joint-set-v1",
                        "plan": job["plan"],
                        "calibration_applied": job["calibration_applied"],
                    },
                )
                if include_decisions:
                    result["decisions"] = list(job["decisions"])
            return result

    def cancel(self, ident):
        with self.lock:
            job = self.jobs[ident]
            if not job["complete"] and job["status"] in {"queued", "running"}:
                job["status"] = "cancelled"
                job["finished"] = time.time()
            return self.get(ident)

    def close(self):
        with self.lock:
            for job in self.jobs.values():
                if job["status"] in {"queued", "running"}:
                    job["status"] = "cancelled"
        if self._owns_pool:
            self.pool.shutdown(wait=True, cancel_futures=True)
        self.scorer.close()
