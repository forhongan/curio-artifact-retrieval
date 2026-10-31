"""Paired query comparisons with explicit provenance and Holm correction."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .stats import holm_adjust, paired_permutation


METRICS = ("ndcg@10", "ap@10", "precision@5", "recall@20", "mrr@10")


def _read_run(path, variant=None, report_path=None):
    path = Path(path)
    report_path = Path(report_path) if report_path else path.with_name("metrics.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"Empty per-query file: {path}")
    variants = sorted({row.get("variant", "") for row in rows})
    if variant is None:
        if len(variants) != 1:
            raise ValueError(f"Select a variant explicitly for {path}; found {variants}")
        variant = variants[0]
    if variant not in variants or not variant:
        raise ValueError(f"Unknown or missing variant {variant!r} in {path}; found {variants}")
    selected = [row for row in rows if row.get("variant") == variant]
    keyed = {}
    for row in selected:
        if not isinstance(row.get("qid"), str) or not isinstance(row.get("language"), str):
            raise ValueError(f"Every row needs string qid and language: {path}")
        key = (row["qid"], row["language"])
        if key in keyed:
            raise ValueError(f"Duplicate qid+language {key!r} for variant {variant}: {path}")
        keyed[key] = row
    fingerprint = {"path": str(path), "report": str(report_path), "variant": variant,
                   "per_query_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                   "index_version": report.get("index", {}).get("index_version"),
                   "encoder": report.get("model", {}).get("encoder"),
                   "probes": report.get("probes"),
                   "effective_variants": dict(Counter(row.get("effective_variant", variant) for row in selected)),
                   "probe_applied_queries": sum(bool(row.get("probe_applied", False)) for row in selected),
                   "degraded_queries": sum(bool(row.get("degraded", False)) for row in selected)}
    return report, keyed, fingerprint


def _verify_provenance(first, second):
    common = {}
    for key in ("query_sha256", "qrels_sha256", "label_source", "gallery_sha256"):
        values = [report.get(key) for report in (first, second)]
        if not all(isinstance(value, str) and value for value in values):
            raise ValueError(f"Missing required provenance field {key}")
        if values[0] != values[1]:
            raise ValueError(f"Incompatible {key}: comparisons require identical queries and qrels provenance")
        common[key] = values[0]
    counts = [report.get("gallery_count") for report in (first, second)]
    if any(not isinstance(count, int) or isinstance(count, bool) or count <= 0 for count in counts):
        raise ValueError("Missing or invalid gallery_count; regenerate reports with gallery provenance")
    if counts[0] != counts[1]:
        raise ValueError("Incompatible gallery_count")
    common["gallery_count"] = counts[0]
    for report in (first, second):
        if "gallery_ids" in report:
            ids = report["gallery_ids"]
            if (not isinstance(ids, list) or any(not isinstance(key, str) or not key for key in ids)
                    or len(ids) != len(set(ids)) or len(ids) != report["gallery_count"]):
                raise ValueError("Invalid gallery_ids in evaluation provenance")
            actual_hash = hashlib.sha256(json.dumps(sorted(ids), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
            if actual_hash != report["gallery_sha256"]:
                raise ValueError("gallery_ids do not match declared gallery_sha256")
    source_sets = [sorted(report.get("label_sources", [report["label_source"]])) for report in (first, second)]
    if source_sets[0] != source_sets[1]:
        raise ValueError("Incompatible label_sources")
    common["label_sources"] = source_sets[0]
    indices = [report.get("index", {}) for report in (first, second)]
    # The encoders/index hashes may differ: that is the treatment under test.
    # Their data snapshot and gallery identity must still be the same.
    for key in ("data_sha256", "catalog_sha256", "items"):
        values = [index.get(key) for index in indices]
        if any(value is None or value == "" for value in values):
            raise ValueError(f"Missing index provenance {key}; regenerate reports with a complete index manifest")
        if values[0] != values[1]:
            raise ValueError(f"Incompatible index {key}: gallery/data snapshots differ")
        common[key] = values[0]
    for report, index in zip((first, second), indices):
        if not index.get("index_version"):
            raise ValueError("Missing index_version in evaluation report")
        if report.get("data_sha256") and report["data_sha256"] != index["data_sha256"]:
            raise ValueError("Evaluation data hash and its index data hash disagree")
        if not isinstance(index["items"], int) or index["items"] <= 0:
            raise ValueError("Index items must be a positive gallery size")
    return common


def compare_runs(baseline, candidate, *, baseline_variant=None, candidate_variant=None,
                 baseline_report=None, candidate_report=None, metrics=("ndcg@10",),
                 repetitions=10000, seed=5523, output=None):
    """Compare candidate minus baseline, preserving every language and sign.

    Holm's family is all requested metric-by-language comparisons in this call.
    Both files must contain identical qid/language keys after variant selection.
    """
    if repetitions < 1:
        raise ValueError("Permutation repetitions must be positive")
    metrics = tuple(metrics)
    if not metrics or len(set(metrics)) != len(metrics) or set(metrics) - set(METRICS):
        raise ValueError(f"Choose distinct metrics from {METRICS}")
    first, a, baseline_info = _read_run(baseline, baseline_variant, baseline_report)
    second, b, candidate_info = _read_run(candidate, candidate_variant, candidate_report)
    provenance = _verify_provenance(first, second)
    if a.keys() != b.keys():
        missing_candidate, missing_baseline = sorted(a.keys() - b.keys()), sorted(b.keys() - a.keys())
        raise ValueError(
            f"Unpaired queries: candidate missing {len(missing_candidate)} (sample {missing_candidate[:5]}); "
            f"baseline missing {len(missing_baseline)} (sample {missing_baseline[:5]})"
        )
    comparisons = []
    for language in sorted({language for _, language in a}):
        keys = sorted(key for key in a if key[1] == language)
        for metric in metrics:
            first_scores, second_scores = [], []
            for key in keys:
                for record, values in ((a[key], first_scores), (b[key], second_scores)):
                    score = record.get(metric)
                    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                        raise ValueError(f"Missing/nonfinite {metric} for {key}")
                    if not 0 <= score <= 1:
                        raise ValueError(f"Out-of-range {metric}={score} for {key}")
                    values.append(score)
            test = paired_permutation(first_scores, second_scores, repetitions=repetitions, seed=seed)
            comparisons.append({"language": language, "metric": metric, "n": len(keys),
                                "baseline_mean": float(np.mean(first_scores)),
                                "candidate_mean": float(np.mean(second_scores)),
                                "delta": test["difference"], "p_value": test["p_value"],
                                "qids": [key[0] for key in keys]})
    adjusted = holm_adjust([row["p_value"] for row in comparisons])
    for row, p_adjusted in zip(comparisons, adjusted):
        row["p_value_holm"] = p_adjusted
        row["significant_0_05_holm"] = p_adjusted < .05
    warnings = ["Paired deltas are candidate minus baseline; all signs/languages are reported. "
                "A confidence interval or p-value does not establish independent visual relevance for silver labels."]
    if baseline_info["degraded_queries"] or candidate_info["degraded_queries"]:
        warnings.append("At least one run contains degraded searches; inspect fallback counts before interpreting model effects.")
    if any(row["n"] < 2 for row in comparisons):
        warnings.append("At least one language has a single query; its bootstrap interval is degenerate and cannot estimate query-population uncertainty.")
    report = {"created_at": datetime.now(timezone.utc).isoformat(), "baseline": baseline_info,
              "candidate": candidate_info, "provenance": provenance,
              "method": {"paired_by": ["qid", "language"], "delta": "candidate - baseline",
                         "confidence_interval": "95% percentile paired bootstrap; 2000 resamples",
                         "test": "two-sided paired sign permutation with plus-one correction",
                         "permutation_repetitions": repetitions, "seed": seed,
                         "multiplicity": "Holm across all requested metric-by-language comparisons",
                         "family_size": len(comparisons)},
              "comparisons": comparisons, "warnings": warnings}
    if output is not None:
        destination = Path(output)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        lines = ["# Paired retrieval comparison", "", f"Baseline: `{baseline_info['path']}` / `{baseline_info['variant']}`.",
                 f"Candidate: `{candidate_info['path']}` / `{candidate_info['variant']}`.", "",
                 f"Label source: **{provenance['label_source']}**. Delta = candidate minus baseline.",
                 f"Holm family: {len(comparisons)} requested metric-by-language comparisons; two-sided permutation test.", "",
                 "| Language | Metric | Queries | Baseline | Candidate | Delta (95% CI) | p | p (Holm) |",
                 "|---|---|---:|---:|---:|---:|---:|---:|"]
        for row in comparisons:
            delta = row["delta"]
            lines.append(f"| {row['language']} | {row['metric']} | {row['n']} | {row['baseline_mean']:.4f} | {row['candidate_mean']:.4f} | {delta['mean']:+.4f} [{delta['lower']:+.4f}, {delta['upper']:+.4f}] | {row['p_value']:.5f} | {row['p_value_holm']:.5f} |")
        lines.extend(["", *warnings, ""])
        (destination / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Baseline per_query.jsonl")
    parser.add_argument("candidate", help="Candidate per_query.jsonl")
    parser.add_argument("--baseline-variant")
    parser.add_argument("--candidate-variant")
    parser.add_argument("--baseline-report", help="metrics.json; defaults to sibling of baseline")
    parser.add_argument("--candidate-report", help="metrics.json; defaults to sibling of candidate")
    parser.add_argument("--metrics", default="ndcg@10", help="Comma-separated metrics; all selected tests share one Holm family")
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=5523)
    parser.add_argument("--output", default="artifacts/evaluation/comparison")
    args = parser.parse_args()
    report = compare_runs(args.baseline, args.candidate, baseline_variant=args.baseline_variant,
                          candidate_variant=args.candidate_variant, baseline_report=args.baseline_report,
                          candidate_report=args.candidate_report, metrics=args.metrics.split(","),
                          repetitions=args.repetitions, seed=args.seed, output=args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
