"""Summarize frozen visual experiments from JSON, without running or tuning models.

Missing reports remain explicitly pending. Existing mismatched provenance raises
before any output is written. Image pixels are read only with --failure-cards;
cards carry blind aliases and never an automatically inferred error cause.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parents[1]
MODELS = ("clip", "siglip", "siglip_adapted", "siglip_visual_adapted")
RUNS = tuple(f"{name}_baseline_translated_test" for name in MODELS) + (
    "selected_test", "siglip_siglip_test", "siglip_vision_test",
)
COMPARISONS = ("clip_vs_siglip", "metadata_lora", "visual_lora", "translated_vs_selected", "base_vs_detail_heads")
PROBES = ("siglip", "vision", "siglip_adapted", "vision_metadata_adapted", "siglip_visual_adapted", "vision_visual_adapted")
METRICS = ("ndcg@10", "precision@5", "ap@10", "recall@20", "mrr@10")
LABEL_WARNING = "AI 图像标注与 AI 视觉相关性审计，不是真人 gold；不报告真人一致性或独立人工收益。"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def number(value, digits=4, signed=False):
    if value is None:
        return "—"
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Invalid numeric evidence: {value!r}")
    return format(value, f"{'+' if signed else ''}.{digits}f")


def interval(metric, signed=False):
    if not isinstance(metric, dict) or metric.get("mean") is None:
        return "—"
    return f"{number(metric['mean'], signed=signed)} [{number(metric.get('lower'), signed=signed)}, {number(metric.get('upper'), signed=signed)}]"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


class Evidence:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.files = {}
        self.missing = []

    def name(self, path):
        path = Path(path).resolve()
        return path.relative_to(self.root).as_posix() if path.is_relative_to(self.root) else str(path)

    def load(self, path, *, required=False, jsonl=False):
        path = Path(path)
        if not path.is_file():
            if required:
                self.missing.append(self.name(path))
            return None
        self.files[self.name(path)] = {"sha256": sha(path), "bytes": path.stat().st_size}
        return read_rows(path) if jsonl else json.loads(path.read_text(encoding="utf-8"))

    def verify_hash(self, path, expected, label):
        path = Path(path)
        if expected and not path.is_file():
            self.missing.append(self.name(path))
        elif expected:
            actual = sha(path)
            self.files[self.name(path)] = {"sha256": actual, "bytes": path.stat().st_size}
            if actual != expected:
                raise ValueError(f"Stale {label}: SHA256 mismatch for {path}")


def verify_retrieval(report, *, data_sha, expected_gallery, expected_queries, common=None):
    if report.get("label_source") != "ai_visual_audit":
        raise ValueError("Visual main results require ai_visual_audit labels, never metadata silver or human relabeling")
    if not report.get("gallery_restricted") or report.get("qrels_scope") != "exhaustive_explicit_gallery":
        raise ValueError("Visual audit must use the explicitly restricted exhaustive gallery")
    if report.get("gallery_count") != expected_gallery:
        raise ValueError(f"Visual gallery count is not the declared {expected_gallery}")
    ids = report.get("gallery_ids")
    if not isinstance(ids, list) or len(ids) != expected_gallery or len(set(ids)) != expected_gallery:
        raise ValueError("Missing or repeated explicit gallery IDs")
    digest = hashlib.sha256(json.dumps(sorted(ids), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    if digest != report.get("gallery_sha256"):
        raise ValueError("Gallery IDs and fingerprint disagree")
    provenance = {key: report.get(key) for key in ("data_sha256", "query_sha256", "qrels_sha256", "gallery_sha256")}
    if any(not isinstance(value, str) or not value for value in provenance.values()):
        raise ValueError("Missing visual report provenance")
    if data_sha and provenance["data_sha256"] != data_sha:
        raise ValueError("Visual retrieval result belongs to a different frozen data snapshot")
    if common and provenance != common:
        raise ValueError("Visual test reports use different data, queries, qrels or galleries")
    if report.get("index", {}).get("data_sha256") != provenance["data_sha256"]:
        raise ValueError("Index and evaluation data hashes disagree")
    seen = set()
    for summary in report.get("summary", []):
        key = (summary["variant"], summary["language"])
        if key in seen:
            raise ValueError("Duplicate variant/language summary")
        seen.add(key)
        for metric in METRICS:
            values = summary.get("metrics", {}).get(metric, {})
            if values.get("n") != expected_queries:
                raise ValueError(f"Visual semantic-query count is not the declared {expected_queries}")
            for field in ("mean", "lower", "upper"):
                value = values.get(field)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Missing/out-of-range {metric} {field}")
        if summary["language"] not in {"en", "zh"}:
            raise ValueError("Expected bilingual en/zh visual summaries")
    if not seen or any({language for variant, language in seen if variant == v} != {"en", "zh"} for v, _ in seen):
        raise ValueError("Every reported variant must include English and Chinese")
    return provenance


def summarize_coverage(records):
    output = {}
    for variant, language in sorted({(row["variant"], row["language"]) for row in records}):
        selected = [row for row in records if row["variant"] == variant and row["language"] == language]
        counts = Counter()
        by_task = defaultdict(Counter)
        for row in selected:
            for polarity, entries in (row.get("probe_coverage") or {}).items():
                for entry in entries:
                    status = entry.get("status", "unknown")
                    counts[f"{polarity}_{status}"] += 1
                    by_task[entry["slot"]][f"{polarity}_{status}"] += 1
        output[f"{variant}/{language}"] = {
            "query_language_observations": len(selected),
            "probe_applied_queries": sum(bool(row.get("probe_applied")) for row in selected),
            "effective_variants": dict(Counter(row.get("effective_variant", row["variant"]) for row in selected)),
            "counts": dict(counts), "by_task": {task: dict(values) for task, values in sorted(by_task.items())},
        }
    return output


def failure_shortlist(details, rankings, qrels, mapping, *, selected_variant, expected_queries, limit=6):
    """One failure per semantic qid, ranked by its worst language's nDCG."""
    alias_by_id = {row["item_id"]: row["alias"] for row in mapping}
    if len(alias_by_id) != len(mapping) or len(set(alias_by_id.values())) != len(mapping):
        raise ValueError("Repeated ID/alias in blind mapping")
    grades = {row["qid"]: row["grades"] for row in qrels}
    if len(grades) != len(qrels):
        raise ValueError("Repeated qid in qrels")
    grouped = defaultdict(dict)
    for row in details:
        if row["variant"] != selected_variant:
            continue
        if row["language"] in grouped[row["qid"]]:
            raise ValueError("Repeated selected query/language")
        value = row.get("ndcg@10")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Invalid selected per-query nDCG")
        grouped[row["qid"]][row["language"]] = row
    if len(grouped) != expected_queries or any(set(values) != {"en", "zh"} for values in grouped.values()):
        raise ValueError("Selected failure inputs do not contain the full bilingual query set")
    if grouped.keys() != grades.keys():
        raise ValueError("Selected queries and visual qrels differ")
    indexed = {(row["qid"], row["language"], row["variant"]): row for row in rankings}
    if len(indexed) != len(rankings):
        raise ValueError("Repeated ranking query/language/variant")
    candidates = []
    for qid, observations in grouped.items():
        language, worst = min(observations.items(), key=lambda item: (item[1]["ndcg@10"], item[0]))
        ranked = indexed[(qid, language, selected_variant)]
        judged = grades[qid]
        if set(judged) != set(alias_by_id):
            raise ValueError("Failure qrels must grade every blind-gallery ID explicitly")
        if any(type(grade) is not int or grade not in range(4) for grade in judged.values()) or max(judged.values()) < 2:
            raise ValueError("Invalid or no-positive visual failure qrels")
        highest = max(judged.values())
        targets = sorted(alias_by_id[key] for key, grade in judged.items() if grade == highest)
        if len(ranked["ranking"]) < min(5, len(mapping)) or len(set(ranked["ranking"])) != len(ranked["ranking"]):
            raise ValueError("Selected ranking is incomplete or repeats image IDs")
        top5 = []
        for position, item_id in enumerate(ranked["ranking"][:5], 1):
            if item_id not in judged:
                raise ValueError("Retrieved item has no explicit visual judgment")
            top5.append({"rank": position, "item_id": item_id, "alias": alias_by_id[item_id], "grade": judged[item_id]})
        candidates.append({
            "qid": qid, "worst_language": language, "variant": selected_variant,
            "minimum_bilingual_ndcg@10": worst["ndcg@10"],
            "ndcg@10_by_language": {lang: value["ndcg@10"] for lang, value in sorted(observations.items())},
            "query": ranked["query"],
            "queries_by_language": {lang: indexed[(qid, lang, selected_variant)]["query"] for lang in sorted(observations)},
            "target_aliases": targets, "target_grade": highest,
            "relevant_aliases": [{"alias": alias_by_id[key], "grade": grade} for key, grade in sorted(judged.items(), key=lambda pair: alias_by_id[pair[0]]) if grade >= 2],
            "top5": top5, "label_source": "ai_visual_audit",
            "manual_analysis_status": "pending_image_review", "error_cause": None,
        })
    return sorted(candidates, key=lambda row: (row["minimum_bilingual_ndcg@10"], row["qid"]))[:limit]


def render_failure_cards(cases, sheets, output):
    """Copy existing blind image cards; do not classify or explain their content."""
    from PIL import Image, ImageDraw

    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for index, case in enumerate(cases, 1):
        cards = [(case["target_aliases"][0], f"TARGET grade {case['target_grade']}")]
        cards += [(row["alias"], f"RANK {row['rank']} grade {row['grade']}") for row in case["top5"]]
        canvas = Image.new("RGB", (960, 740), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((10, 6), f"{case['qid']} / {case['worst_language']} / nDCG@10={case['minimum_bilingual_ndcg@10']:.4f}", fill="black")
        english = case["queries_by_language"]["en"]
        for line, text in enumerate(textwrap.wrap(english, 125)[:3]):
            draw.text((10, 22 + line * 15), text, fill="black")
        for position, (alias, label) in enumerate(cards):
            ordinal = int(alias[1:]) - 1
            sheet = sheets / f"sheet_{ordinal // 12 + 1:03d}.png"
            with Image.open(sheet) as source:
                sx, sy = ordinal % 12 % 3 * 320, ordinal % 12 // 3 * 300
                crop = source.convert("RGB").crop((sx, sy, sx + 320, sy + 300))
            x, y = position % 3 * 320, 80 + position // 3 * 330
            draw.text((x + 10, y), f"{label}: {alias}", fill="black")
            canvas.paste(crop, (x, y + 20))
        path = output / f"failure_{index:02d}_{case['qid']}.png"
        canvas.save(path)
        paths.append(str(path))
    return paths


def merge_probe_test_report(report, test_report, training_report_sha):
    """Join held-out metrics only when they belong to these frozen heads."""
    if test_report.get("training_report_sha256") != training_report_sha:
        raise ValueError("Probe test metrics do not match frozen training report")
    for key in ("data_sha256", "index_sha256", "index_manifest_sha256", "detail_annotations_sha256"):
        if test_report.get(key) != report.get(key):
            raise ValueError(f"Probe test provenance differs from training report: {key}")
    trained = {task for task, row in report["tasks"].items() if row.get("status") == "trained"}
    if set(test_report.get("tasks", {})) != trained:
        raise ValueError("Probe test metrics omit or add trained tasks")
    for task in trained:
        name = f"{task}.npz"
        digest = report.get("artifacts_sha256", {}).get(name)
        if not digest or test_report.get("artifacts_sha256", {}).get(name) != digest:
            raise ValueError(f"Probe test weights differ from frozen training weights: {name}")
    return {**report, "test_evaluation": test_report,
            "tasks": {task: {**row, **test_report.get("tasks", {}).get(task, {})}
                      for task, row in report["tasks"].items()}}


def verify_frozen_retrieval(report, *, model, probes, weight, variants, model_files):
    """Connect each reported run to the exact model and head choice frozen on dev."""
    manifest = f"artifacts/{model}/manifest.json"
    if not model_files.get(manifest) or report.get("index_manifest_sha256") != model_files[manifest]:
        raise ValueError(f"Retrieval index differs from frozen model: {model}")
    if {row["variant"] for row in report["summary"]} != set(variants):
        raise ValueError("Retrieval variants differ from the declared frozen comparison")
    configured = report.get("probes", {})
    if bool(configured.get("configured")) != bool(probes):
        raise ValueError("Retrieval probe configuration differs from frozen selection")
    if probes:
        if configured.get("path") != probes or configured.get("weight") != weight:
            raise ValueError("Retrieval probe path/weight differs from frozen selection")
        prefix = f"{Path(probes).as_posix().rstrip('/')}/"
        expected = {name[len(prefix):]: digest for name, digest in model_files.items()
                    if name.startswith(prefix)}
        if not expected or configured.get("files_sha256") != expected:
            raise ValueError("Retrieval probe files differ from frozen weights")


def classify_system_acceptance(report, selection):
    """A filename or healthy server alone cannot establish production provenance."""
    config = report.get("runtime_config")
    if not isinstance(config, dict) or not selection:
        return {"scope": "unverified_configuration", "configuration_matches": False,
                "reasons": ["Missing runtime fingerprints or frozen production selection; cannot prove the current configuration."]}
    choice, reasons = selection["selected"], []
    expected = {"data_sha256": selection.get("data_sha256"), "default_variant": choice["variant"],
                "encoder_name": "clip" if choice["model"] == "clip" else "siglip",
                "index_manifest_sha256": selection.get("model_files_sha256", {}).get(f"artifacts/{choice['model']}/manifest.json")}
    for key, value in expected.items():
        if value is None or config.get(key) != value:
            reasons.append(f"Runtime {key} differs from frozen production")
    probe_path = choice.get("probes")
    probes = config.get("probes") or {}
    if bool(probes.get("configured")) != bool(probe_path):
        reasons.append("Runtime probe configuration differs from frozen production")
    if probe_path:
        prefix = f"{Path(probe_path).as_posix().rstrip('/')}/"
        probe_files = {name[len(prefix):]: digest for name, digest in selection.get("model_files_sha256", {}).items() if name.startswith(prefix)}
        if not probes.get("available") or probes.get("weight") != choice["weight"] or not probe_files or probes.get("files_sha256") != probe_files:
            reasons.append("Runtime probe files/weight differ from frozen production")
    matches = not reasons
    if matches:
        records, requested = report.get("request_records"), report.get("requested") or {}
        if report.get("schema") != "curio-system-acceptance-v2" or not isinstance(records, list):
            reasons.append("Missing recorded acceptance requests")
        else:
            for phase in ("warm_sequential", "concurrent_image_only"):
                if sum(row.get("phase") == phase for row in records) != requested.get("count_per_phase"):
                    reasons.append(f"Recorded {phase} request count differs from declared count")
            request_ids = [row.get("request_id") for row in records]
            if not request_ids or any(not request_id for request_id in request_ids) or len(set(request_ids)) != len(request_ids):
                reasons.append("Missing or repeated recorded request IDs")
            for row in records:
                payload, wall = row.get("request") or {}, row.get("wall_ms")
                if (not payload.get("query") or not payload.get("variant") or row.get("degraded") is not False
                        or isinstance(wall, bool) or not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall < 0):
                    reasons.append("Incomplete or degraded recorded request")
                    break
                if row.get("phase") == "concurrent_image_only" and payload["variant"] != choice["variant"]:
                    reasons.append("Concurrent requests did not exercise the production variant")
                    break
        if (report.get("health_after") or {}).get("status") != "ok":
            reasons.append("System health did not pass")
    return {"scope": "current_frozen_production" if not reasons else "not_current_production_evidence",
            "configuration_matches": matches, "reasons": reasons}


def build(root=ROOT, *, evaluation=None, selection_path=None, output=None, evidence_output=None,
          failure_output=None, expected_gallery=60, expected_queries=24, require_complete=False, failure_cards=False):
    root = Path(root).resolve()
    evaluation = Path(evaluation) if evaluation else root / "artifacts/evaluation"
    audit = evaluation / "image_audit"
    output = Path(output) if output else root / "docs/VISUAL_RESULTS.md"
    evidence_output = Path(evidence_output) if evidence_output else root / "docs/visual_experiment_evidence.json"
    failure_output = Path(failure_output) if failure_output else audit / "failure_shortlist.json"
    selection_path = Path(selection_path) if selection_path else root / "artifacts/visual_selection.json"
    files = Evidence(root)
    selection = files.load(selection_path, required=True)
    status = files.load(root / "artifacts/visual_experiment_status.json")
    test_freeze = files.load(root / "artifacts/visual_test_freeze.json", required=selection is not None)
    data_sha = (selection or {}).get("data_sha256")
    if selection:
        files.verify_hash(root / "data/snapshots/collection_v2.jsonl", data_sha, "data snapshot")
        files.verify_hash(root / "eval/visual_labels/annotations.jsonl", selection.get("annotations_sha256"), "visual annotations")
        for name, digest in selection.get("visual_dev_files_sha256", {}).items():
            files.verify_hash(root / "eval/visual_dev" / name, digest, "frozen visual dev")
        for name, digest in selection.get("model_files_sha256", {}).items():
            path = Path(name) if Path(name).is_absolute() else root / name
            files.verify_hash(path, digest, "frozen visual model")
        if test_freeze is not None:
            if test_freeze.get("selection_sha256") != files.files[files.name(selection_path)]["sha256"]:
                raise ValueError("Visual test freeze belongs to a different dev selection")
            frozen_config = test_freeze.get("config", {})
            if not frozen_config or any(selection.get(key) != value for key, value in frozen_config.items()):
                raise ValueError("Visual test freeze and dev configuration disagree")
            for name, digest in test_freeze.get("test_files_sha256", {}).items():
                files.verify_hash(root / "eval/visual_test" / name, digest, "frozen visual test")
    paired, retrieval, details, comparisons, probes, training = {}, {}, {}, {}, {}, {}
    common = None
    for model in MODELS:
        report = files.load(evaluation / f"visual_paired_{model}_test.json", required=True)
        if report is None:
            continue
        if data_sha and report.get("data_sha256") != data_sha:
            raise ValueError(f"Stale paired data result: {model}")
        if report.get("split") != "test" or report.get("label_source") != "held-out AI image-only caption pairs":
            raise ValueError("Visual paired results need held-out AI image-only captions")
        if report.get("metrics", {}).get("gallery") != expected_gallery:
            raise ValueError("Paired visual gallery differs from the declared image count")
        if selection and report.get("visual_annotations_sha256") != selection.get("annotations_sha256"):
            raise ValueError("Paired captions differ from frozen visual annotations")
        paired[model] = report
    for name in RUNS:
        report = files.load(audit / name / "metrics.json", required=True)
        if report is None:
            continue
        provenance = verify_retrieval(report, data_sha=data_sha, expected_gallery=expected_gallery,
                                      expected_queries=expected_queries, common=common)
        if selection and selection.get("model_files_sha256"):
            if name == "selected_test":
                choice = selection["selected"]
                model, probe_path, weight = choice["model"], choice.get("probes"), choice["weight"]
                variants = [choice["variant"]]
            elif name in {"siglip_siglip_test", "siglip_vision_test"}:
                model, probe_name = "siglip", name.removeprefix("siglip_").removesuffix("_test")
                probe_path, weight = f"artifacts/probes/{probe_name}", selection["head_ablation_weight"]
                variants = ["image_only"]
            else:
                model = name.removesuffix("_baseline_translated_test")
                probe_path, weight, variants = None, None, ["baseline", "translated"]
            verify_frozen_retrieval(report, model=model, probes=probe_path, weight=weight,
                                    variants=variants, model_files=selection["model_files_sha256"])
        common = common or provenance
        retrieval[name] = report
        records = files.load(audit / name / "per_query.jsonl", required=True, jsonl=True)
        if records is not None:
            details[name] = records
    if common:
        for name, key in (("queries.jsonl", "query_sha256"), ("qrels.jsonl", "qrels_sha256")):
            files.verify_hash(root / "eval/visual_test" / name, common[key], "visual test inputs")
        gallery = files.load(root / "eval/visual_test/gallery_ids.json", required=True)
        if gallery is not None:
            digest = hashlib.sha256(json.dumps(sorted(gallery), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            if digest != common["gallery_sha256"]:
                raise ValueError("Visual test gallery file differs from reported gallery")
    for name, digest in (status or {}).get("test_files_sha256", {}).items():
        files.verify_hash(root / "eval/visual_test" / name, digest, "frozen visual test")
    for name in COMPARISONS:
        report = files.load(audit / "comparisons" / name / "comparison.json", required=True)
        if report is not None:
            if common and any(report.get("provenance", {}).get(key) != value for key, value in common.items()):
                raise ValueError(f"Stale/incompatible comparison: {name}")
            comparisons[name] = report
    for name in PROBES:
        directory = root / "artifacts/probes" / name
        report_path = directory / "report.json"
        report = files.load(report_path, required=True)
        if report is not None:
            if data_sha and report.get("data_sha256") != data_sha:
                raise ValueError(f"Stale probe data: {name}")
            if name.startswith("vision") and selection and report.get("detail_annotations_sha256") != selection.get("annotations_sha256"):
                raise ValueError("Detail probe annotations differ from frozen selection")
            test_report = files.load(directory / "test_report.json", required=report.get("test_deferred", False))
            if test_report is not None:
                report = merge_probe_test_report(report, test_report, files.files[files.name(report_path)]["sha256"])
                for filename, digest in test_report.get("artifacts_sha256", {}).items():
                    files.verify_hash(directory / filename, digest, "frozen probe test weights")
            probes[name] = report
    for sampling in ("random", "hard"):
        report = files.load(root / "artifacts/training" / f"visual_lora_{sampling}" / "run.json", required=True)
        if report is not None:
            if data_sha and report.get("data_sha256") != data_sha:
                raise ValueError(f"Stale visual training data: {sampling}")
            if selection and report.get("visual_annotations_sha256") != selection.get("annotations_sha256"):
                raise ValueError("Visual training captions differ from frozen annotations")
            training[sampling] = report
    acceptance = {}
    acceptance_paths = set((root / "artifacts").glob("*system*accept*.json")) | set(evaluation.glob("*accept*.json"))
    for path in sorted(acceptance_paths):
        report = files.load(path)
        classification = ({"scope": "pilot_only", "configuration_matches": False, "reasons": ["Historical pilot only"]}
                          if "pilot" in path.name else classify_system_acceptance(report, selection))
        acceptance[files.name(path)] = {**classification, "report": report}
    failures = []
    if "selected_test" in retrieval:
        rankings = files.load(audit / "selected_test/rankings.jsonl", required=True, jsonl=True)
        qrels = files.load(root / "eval/visual_test/qrels.jsonl", required=True, jsonl=True)
        mapping = files.load(root / "eval/visual_test/blind_map.json", required=True)
        if all(value is not None for value in (rankings, qrels, mapping, details.get("selected_test"))):
            if {row["item_id"] for row in mapping} != set(retrieval["selected_test"]["gallery_ids"]):
                raise ValueError("Blind mapping differs from the selected test gallery")
            selected_variant = (selection or {}).get("selected", {}).get("variant")
            variants = {row["variant"] for row in details["selected_test"]}
            if not selected_variant and len(variants) == 1:
                selected_variant = next(iter(variants))
            if selected_variant not in variants or len(variants) != 1:
                raise ValueError("Selected test variant differs from frozen dev selection")
            failures = failure_shortlist(details["selected_test"], rankings, qrels, mapping,
                                         selected_variant=selected_variant, expected_queries=expected_queries)
    if require_complete and files.missing:
        raise FileNotFoundError("Required visual evidence is pending:\n" + "\n".join(sorted(set(files.missing))))
    evidence = {
        "created_at": datetime.now(timezone.utc).isoformat(), "schema": "curio-visual-results-v1",
        "status": "complete" if not files.missing else "partial_pending_reports",
        "protocol": {"gallery_count": expected_gallery, "semantic_queries": expected_queries,
                     "languages": ["en", "zh"], "label_source": "ai_visual_audit", "warning": LABEL_WARNING,
                     "counts_source": "declared_protocol; checked against every available test report",
                     "bilingual_queries_independent": False, "test_images_analyzed_by_summarizer": False},
        "selection": selection, "test_freeze": test_freeze, "runner_status": status, "paired": paired, "retrieval": retrieval,
        "training": training, "probes": probes, "comparisons": comparisons,
        "probe_coverage": {name: summarize_coverage(records) for name, records in details.items()},
        "system_acceptance": acceptance, "source_files": files.files,
        "missing_reports": sorted(set(files.missing)),
        "failure_selection_rule": "Distinct semantic qids ranked by minimum en/zh selected-test nDCG@10 ascending, ties by qid; worst-language ties prefer en; exactly first six when at least six qids exist.",
        "failure_target_definition": "All highest-grade judged relevant aliases, not an assumed query-author source object.",
        "failure_shortlist": failures,
    }
    lines = markdown(evidence)
    # Refuse a mixed snapshot if a concurrent runner replaced any input.
    for path, fingerprint in files.files.items():
        source = Path(path) if Path(path).is_absolute() else root / path
        if sha(source) != fingerprint["sha256"]:
            raise ValueError(f"Evidence changed while summarizing: {source}")
    cards = render_failure_cards(failures, root / "docs/qa/visual_test", audit / "failure_cards") if failure_cards and failures else []
    evidence["failure_contact_sheets"] = cards
    evidence_json = json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    failure_json = json.dumps(failures, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    for path in (output, evidence_output, failure_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(lines, encoding="utf-8")
    evidence_output.write_text(evidence_json, encoding="utf-8")
    failure_output.write_text(failure_json, encoding="utf-8")
    return evidence


def markdown(e):
    protocol, selection = e["protocol"], e["selection"] or {}
    lines = ["# 视觉主实验结果", "", "由 `scripts/summarize_visual_results.py` 读取真实 JSON 生成；缺失结果保留待完成，不填示例数值。", "",
             f"**{LABEL_WARNING}**", "",
             f"状态：`{e['status']}`。协议声明显式受限图库 **{protocol['gallery_count']} 图**、**{protocol['semantic_queries']} 个语义查询**；每份已到报告均核对数量。中英分别报告，两种语言不是两倍独立样本。", "",
             "nDCG 使用 0–3 级和 `2^grade−1`；P/AP/Recall/MRR 以 grade≥2 为相关。AP@10 分母为全部相关项。CI 在各语言内 bootstrap 语义查询，不能推断对完整馆藏或真人查询的泛化。", "",
             "## 来源与冻结选择", "",
             "- CLIP / SigLIP 为零样本图文基线；`siglip_adapted` 是馆藏弱元数据 caption LoRA 对照；`siglip_visual_adapted` 使用披露的 AI 盲图 caption。",
             "- 基础 material/type heads 来自馆藏元数据弱标签；vision 在对应特征空间追加 AI 盲图 colors/shape/motifs sigmoid heads。SigLIP 零样本与视觉 LoRA 各有匹配的独立 probe artifacts，不能跨空间混用。推理输入仍为图像向量，不是查询时的元数据分数。",
             "- `image_only` 不使用 BM25、metadata slot 或 metadata conflict 评分；显式 facet 可以按元数据筛选候选。",
             "- 视觉主路径和正权重 head ablation 都在 dev 冻结；test 结果不反过来选模型、采样、epoch、C、查询或重排权重。", "",
             f"冻结时间：`{selection.get('frozen_at', '待生成')}`；数据 SHA256：`{selection.get('data_sha256', '待生成')}`；视觉标注 SHA256：`{selection.get('annotations_sha256', '待生成')}`。", "",
             f"选中 production：`{json.dumps(selection.get('selected'), ensure_ascii=False)}`。",
             f"全比较系统 dev 最佳（与 production 区分）：`{json.dumps(selection.get('overall_best_comparison'), ensure_ascii=False)}`。",
             f"独立 head ablation 正权重：`{selection.get('head_ablation_weight', '待生成')}`；选择规则：{selection.get('selection_rule', '待生成')}。", "",
             "| dev 候选 | 变体 | probes | 权重 | 双语 dev nDCG@10 |", "|---|---|---|---:|---:|"]
    for row in selection.get("options", []):
        lines.append(f"| {cell(row['model'])} | {cell(row['variant'])} | {cell(row.get('probes'))} | {number(row.get('weight'), 2)} | {number(row.get('dev_ndcg@10'))} |")
    lines += ["", "## AI caption 配对测试", "", "配对标签是 AI 盲图 caption；完全相同 caption 多正例。R@K 为查询命中率。此任务与下方自由查询相关性审计不同。", "",
              "| 模型 | 图库 | 独立 caption | R@1 | R@5 | R@10 | MRR |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, report in e["paired"].items():
        m = report["metrics"]
        lines.append(f"| {name} | {m.get('gallery', '—')} | {m.get('unique_captions', '—')} | {number(m.get('r@1'))} | {number(m.get('r@5'))} | {number(m.get('r@10'))} | {number(m.get('mrr'))} |")
    lines += ["", "## LoRA 训练与负结果", "", "保留 random/hard 两条真实训练记录；`improved_over_baseline=false` 不被删除。下表只描述 paired dev，不能将 dev 改善冒充自由查询 test 改善。", "",
              "| 采样 | caption 来源 | 训练/dev 图数 | 最佳 epoch | baseline dev R@5 | best dev R@5 | ΔR@5 | best MRR | 优于 baseline |", "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for name, row in e["training"].items():
        base, best = row.get("baseline_dev", {}), row.get("best_dev", {})
        delta = best.get("r@5") - base.get("r@5") if best.get("r@5") is not None and base.get("r@5") is not None else None
        lines.append(f"| {name} | {cell(row.get('label_source', '未报告'))} | {row.get('train_items', '—')}/{row.get('dev_items', '—')} | {row.get('best_epoch', '—')} | {number(base.get('r@5'))} | {number(best.get('r@5'))} | {number(delta, signed=True)} | {number(best.get('mrr'))} | {row.get('improved_over_baseline', '未报告')} |")
        lines.append(f"\n`{name}` caption policy：{row.get('caption_policy', '未报告')}。\n")
    lines += ["", "## 全变体自由查询测试", "", "每个指标均显示 mean [95% CI]；展示全部模型、变体与语言，不只展示获胜结果。", "",
              "| run | 变体 | 语言 | nDCG@10 [CI] | P@5 [CI] | AP@10 [CI] | R@20 [CI] | MRR@10 [CI] |", "|---|---|---|---|---|---|---|---|"]
    for name, report in e["retrieval"].items():
        for row in report["summary"]:
            values = " | ".join(interval(row["metrics"][metric]) for metric in METRICS)
            lines.append(f"| {name} | {row['variant']} | {row['language']} | {values} |")
    lines += ["", "## 实际 probe 使用与降级", "", "计数按 query-language 观测分别列出；covered 不是图像属性真值。weight=0 不算实际 probe 应用；未知模型类不当作属性不存在。", "",
              "| run | 变体/语言 | 实际 probe 查询 | 降级查询 | effective variants | P50/P95 ms |", "|---|---|---:|---:|---|---:|"]
    for name, report in e["retrieval"].items():
        for row in report["summary"]:
            lines.append(f"| {name} | {row['variant']}/{row['language']} | {row.get('probe_applied_queries', '未报告')} | {row.get('degraded_queries', '未报告')} | {cell(row.get('effective_variants', {}))} | {number(row.get('p50_ms'), 2)}/{number(row.get('p95_ms'), 2)} |")
    lines += ["", "| run | 变体/语言 | positive covered/unknown | negation covered/unknown |", "|---|---|---:|---:|"]
    for name, languages in e["probe_coverage"].items():
        for language, row in languages.items():
            c = row["counts"]
            lines.append(f"| {name} | {language} | {c.get('positive_covered', 0)}/{c.get('positive_unknown', 0)} | {c.get('negations_covered', 0)}/{c.get('negations_unknown', 0)} |")
    lines += ["", "## Head 标签来源与支持度", "", "材料/器型与 AI 颜色/器形/纹饰的标签来源分开。AI detail test 的 support 是该属性已知标签图数，unknown 被排除；无双类支持时 AUC 为空，不补成零。", "",
              "| probe artifacts | 特征维数 | index version | AI 标注 train/dev/test 图数 | 标注 SHA256 |", "|---|---:|---|---|---|"]
    for name, report in e["probes"].items():
        lines.append(f"| {name} | {report.get('feature_dimension', '未报告')} | {cell(report.get('index_version', '未报告'))} | {cell(report.get('detail_annotation_counts', '不适用/未报告'))} | {cell(report.get('detail_annotations_sha256', '不适用/未报告'))} |")
    lines += ["",
              "| probes | 任务 | 状态/标签来源 | 类别 | selected C | dev macro F1 |", "|---|---|---|---|---:|---:|"]
    for name, report in e["probes"].items():
        for task, row in report.get("tasks", {}).items():
            source = row.get("label_source", "museum_metadata_weak" if task in {"material", "object_type"} else "未报告")
            lines.append(f"| {name} | {task} | {cell(row.get('status'))}/{cell(source)} | {cell(', '.join(row.get('classes', [])) or row.get('reason', '无'))} | {number(row.get('selected_C'), 2)} | {number(row.get('dev_macro_f1'))} |")
    lines += ["", "| probes | 元数据弱标签任务 | train/dev support | test eligible/excluded | test support | test accuracy | test balanced accuracy | test macro F1 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, report in e["probes"].items():
        for task, row in report.get("tasks", {}).items():
            if task not in {"material", "object_type"} or row.get("status") != "trained":
                continue
            m = row.get("test") or {}
            lines.append(f"| {name} | {task} | {row.get('train_support', '—')}/{row.get('dev_support', '—')} | {row.get('test_eligible_support', '—')}/{row.get('test_excluded_support', '—')} | {m.get('support', '—')} | {number(m.get('accuracy'))} | {number(m.get('balanced_accuracy'))} | {number(m.get('macro_f1'))} |")
    lines += ["", "| probes | AI 任务/类别 | train 0/1 | test known | test positives | test F1 | test ROC AUC |", "|---|---|---:|---:|---:|---:|---:|"]
    for name, report in e["probes"].items():
        for task, row in report.get("tasks", {}).items():
            if row.get("activation") != "sigmoid":
                continue
            for label, m in (row.get("test") or {}).items():
                counts = row.get("train_counts", {}).get(label, {})
                lines.append(f"| {name} | {task}/{label} | {counts.get('0', '—')}/{counts.get('1', '—')} | {m.get('support', '—')} | {m.get('positives', '—')} | {number(m.get('f1'))} | {number(m.get('roc_auc'))} |")
    lines += ["", "## 成对比较：保留正负差值", "", "候选减基线；每组 JSON 的预声明指标×语言共同组成 Holm 家族。CI、p 值与所有指标均保留；负值或不显著结果不会隐藏。", "",
              "| 对比 | 语言 | 指标 | 查询数 | baseline | candidate | Δ [95% CI] | p | p Holm |", "|---|---|---|---:|---:|---:|---|---:|---:|"]
    for name, report in e["comparisons"].items():
        for row in report.get("comparisons", []):
            lines.append(f"| {name} | {row['language']} | {row['metric']} | {row.get('n', '—')} | {number(row.get('baseline_mean'))} | {number(row.get('candidate_mean'))} | {interval(row.get('delta'), signed=True)} | {number(row.get('p_value'), 5)} | {number(row.get('p_value_holm'), 5)} |")
    lines += ["", "## 固定失败短名单", "", "按 selected test 每个语义 qid 在中英两种表述中的最小 nDCG@10 升序，qid 稳定打破并列，固定保留六个不同 qid。先后规则不使用人工挑选。",
              "目标 alias 是 qrels 最高等级的全部匹配图，不是假定查询源图。下表展示最差语言的前五；JSON 保留 ID、alias 和显式等级。错误原因留空，等待真正看盲图分析。", "",
              "| qid | 最差语言 | min nDCG@10 | 目标 aliases | top5 alias:grade |", "|---|---|---:|---|---|"]
    for row in e["failure_shortlist"]:
        top = ", ".join(f"{item['alias']}:{item['grade']}" for item in row["top5"])
        lines.append(f"| {row['qid']} | {row['worst_language']} | {number(row['minimum_bilingual_ndcg@10'])} | {', '.join(row['target_aliases'])} | {top} |")
    lines += ["", "联系表须显式加 `--failure-cards` 才生成；本汇总不识图、不推断错误原因、不改变参数。", "", "## 系统验收证据", ""]
    if not e["system_acceptance"]:
        lines.append("尚无系统验收 JSON；不推断端到端功能、延迟或吞吐已通过。")
    for path, entry in e["system_acceptance"].items():
        report = entry["report"]
        lines += [f"- `{path}`：scope=`{entry['scope']}`；health=`{(report.get('health_after') or {}).get('status', report.get('status', '未报告'))}`。",
                  f"  warm sequential：`{json.dumps(report.get('warm_sequential'), ensure_ascii=False)}`；并发：`{json.dumps(report.get('concurrent_image_only', report.get('concurrent_hybrid', report.get('concurrent'))), ensure_ascii=False)}`。"]
        if entry.get("reasons"):
            lines.append(f"  不能证明当前冻结生产配置通过验收：{cell('; '.join(entry['reasons']))}。")
    lines += ["", "pilot 验收仅证明其当时图库和配置；不能替代当前冻结图库或 production image_only 的系统验收。原始 checks、健康状态与时延均保留在 evidence JSON。", "", "## 待完成与解释边界", ""]
    lines += [f"- 待生成：`{path}`。" for path in e["missing_reports"]] or ["- 所有预期 JSON 已齐备；数值仍受小规模 AI 视觉审计的边界约束。"]
    lines += ["- 60 图受限图库的结果不能当作全馆藏检索精度。AI caption paired 与 AI 自由查询共享 AI 判断偏差。",
              "- 黑白历史图无法恢复真实本体颜色；细粒度纹饰和难例需另行看图，未知不能当确定负例。",
              "- 详情 heads、LoRA 和查询改写可能无收益或退化；完整训练和成对比较保留这些负结果。",
              "- 元数据银标仅为附录：`scripts/summarize_results.py` / `docs/METADATA_RESULTS.md`，不替代视觉主结果。",
              "- 所有读取 JSON 的 SHA256、完整原始 reports、选项与失败记录见 `docs/visual_experiment_evidence.json`。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--evaluation", type=Path)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--evidence-output", type=Path)
    parser.add_argument("--failure-output", type=Path)
    parser.add_argument("--expected-gallery", type=int, default=60)
    parser.add_argument("--expected-queries", type=int, default=24)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--failure-cards", action="store_true", help="Render blind cards for later manual image review; never infer causes")
    args = parser.parse_args()
    if args.expected_gallery < 1 or args.expected_queries < 1:
        parser.error("Expected gallery/query counts must be positive")
    result = build(args.root, evaluation=args.evaluation, selection_path=args.selection, output=args.output,
                   evidence_output=args.evidence_output, failure_output=args.failure_output,
                   expected_gallery=args.expected_gallery, expected_queries=args.expected_queries,
                   require_complete=args.require_complete, failure_cards=args.failure_cards)
    print(json.dumps({"status": result["status"], "missing_reports": result["missing_reports"],
                      "failure_cases": len(result["failure_shortlist"]), "test_images_analyzed": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
