"""Reproducible SigLIP2 prompt calibration with a frozen dev selection.

The experiment separates prompt selection from supplementary test evaluation:

    python scripts/run_prompt_calibration.py --phase select-dev
    python scripts/run_prompt_calibration.py --phase evaluate-test

The second command refuses to read the test split unless the exact selection
manifest written by the first command is still present.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re

import numpy as np

from curio.evaluate.metrics import (
    average_precision,
    ndcg,
    precision,
    recall,
    reciprocal_rank,
)
from curio.evaluate.stats import bootstrap_ci, holm_adjust
from curio.represent.encoders import Encoder

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "prompt-calibration-v1"
SUMMARY_METRICS = ("P@5", "nDCG@5", "nDCG@10", "AP@10", "Recall@10", "MRR@10")
SELECTION_METRIC = "nDCG@10"
SELECTION_METRIC_TOLERANCE = 1e-12


@dataclass(frozen=True)
class PromptSpec:
    name: str
    family: str
    template: str
    description: str


PROMPT_SPECS = (
    PromptSpec("raw", "baseline", "{text}", "Reference query without a transformation."),
    PromptSpec("lowercase", "case_punctuation", "{lower}", "Lowercase the complete query."),
    PromptSpec(
        "capitalize_first",
        "case_punctuation",
        "{capitalized}",
        "Capitalize the first character and lowercase the remaining text.",
    ),
    PromptSpec(
        "terminal_period",
        "case_punctuation",
        "{without_terminal_punctuation}.",
        "Replace terminal punctuation with one period.",
    ),
    PromptSpec(
        "remove_terminal_punctuation",
        "case_punctuation",
        "{without_terminal_punctuation}",
        "Remove terminal punctuation without changing case or words.",
    ),
    PromptSpec(
        "strip_leading_article",
        "article",
        "{without_leading_article}",
        "Remove a leading English a, an, or the; no-op for Chinese.",
    ),
    PromptSpec("prepend_the", "article", "the {text}", "Prepend the definite article."),
    PromptSpec(
        "prefix_a_photo_of", "prefix", "a photo of {text}", "Common image-caption prefix."
    ),
    PromptSpec(
        "prefix_photo_of", "prefix", "photo of {text}", "Article-free photo prefix."
    ),
    PromptSpec(
        "prefix_a_photograph_of",
        "prefix",
        "a photograph of {text}",
        "Photograph wording rather than photo.",
    ),
    PromptSpec(
        "prefix_a_picture_of", "prefix", "a picture of {text}", "Picture-based prefix."
    ),
    PromptSpec(
        "prefix_an_image_of", "prefix", "an image of {text}", "Image-based prefix."
    ),
    PromptSpec("prefix_image_of", "prefix", "image of {text}", "Article-free image prefix."),
    PromptSpec(
        "prefix_museum_photograph_of",
        "prefix",
        "a museum photograph of {text}",
        "Museum-specific photograph prefix.",
    ),
    PromptSpec(
        "prefix_close_up_photograph_of",
        "prefix",
        "a close-up photograph of {text}",
        "Close-up photograph prefix.",
    ),
    PromptSpec(
        "suffix_plain_background",
        "suffix",
        "{without_terminal_punctuation}, photographed against a plain background",
        "Describe the background convention as a suffix.",
    ),
    PromptSpec(
        "suffix_museum_object",
        "suffix",
        "{without_terminal_punctuation}, museum object photograph",
        "Short museum-photograph suffix.",
    ),
    PromptSpec(
        "caption_image_showing",
        "caption_like",
        "an image showing {text}",
        "Caption-like present-participle phrasing.",
    ),
    PromptSpec(
        "caption_illustrating",
        "caption_like",
        "an image illustrating {text}",
        "Caption-like illustrating phrasing.",
    ),
)


def _without_terminal_punctuation(text: str) -> str:
    return text.rstrip(" \t\r\n.。!！?？;；,，:：")


def _without_leading_article(text: str) -> str:
    return re.sub(r"^(?:a|an|the)\s+", "", text, count=1, flags=re.IGNORECASE)


def render_prompt(text: str, name: str) -> str:
    """Render one registered prompt without external state."""
    try:
        spec = next(item for item in PROMPT_SPECS if item.name == name)
    except StopIteration as exc:
        raise ValueError(f"Unknown prompt template: {name}") from exc
    if not text.strip():
        raise ValueError("Prompt source text must not be empty")
    stripped = text.strip()
    values = {
        "text": stripped,
        "lower": stripped.lower(),
        "capitalized": stripped[:1].upper() + stripped[1:].lower(),
        "without_terminal_punctuation": _without_terminal_punctuation(stripped),
        "without_leading_article": _without_leading_article(stripped),
    }
    return spec.template.format(**values)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ranking_metrics(ranking: list[str], qrels: dict[str, int]) -> dict[str, float]:
    return {
        "P@5": precision(ranking, qrels, 5),
        "nDCG@5": ndcg(ranking, qrels, 5),
        "nDCG@10": ndcg(ranking, qrels, 10),
        "AP@10": average_precision(ranking, qrels, 10),
        "Recall@10": recall(ranking, qrels, 10),
        "MRR@10": reciprocal_rank(ranking, qrels, 10),
    }


def summarize_rows(rows: list[dict]) -> dict:
    return {
        "queries": len(rows),
        **{
            metric: bootstrap_ci([row[metric] for row in rows])
            for metric in SUMMARY_METRICS
        },
    }


def paired_difference(raw_values, candidate_values, *, repetitions=10_000, seed=5523) -> dict:
    """Bootstrap a paired mean difference and sign-flip test its null.

    The centered null distribution uses exactly zero paired differences as
    fixed points. This keeps the conventional two-sided test for integer or
    bounded metrics instead of obtaining an artificially small tail from a
    cosmetic, non-null-changing template.
    """
    raw = np.asarray(raw_values, dtype=float)
    candidate = np.asarray(candidate_values, dtype=float)
    if raw.shape != candidate.shape or raw.ndim != 1 or not len(raw):
        raise ValueError("Paired one-dimensional observations are required")
    delta = candidate - raw
    rng = np.random.default_rng(seed)
    resampled = delta[rng.integers(len(delta), size=(2000, len(delta)))]
    means = resampled.mean(axis=1)
    nonzero = delta != 0
    signs = rng.choice([-1.0, 1.0], size=(repetitions, int(nonzero.sum())))
    null = (delta[nonzero] * signs).sum(axis=1) / len(delta)
    observed = delta.mean()
    p_value = (np.sum(np.abs(null) >= abs(observed)) + 1) / (repetitions + 1)
    return {
        "difference": float(observed),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "p_value": float(p_value),
        "n": len(delta),
        "zero_differences": int((~nonzero).sum()),
    }


def select_variant(results: dict[str, dict]) -> tuple[str, str]:
    """Select dev nDCG@10, preferring raw for a numerical tie."""
    names = [spec.name for spec in PROMPT_SPECS]
    best_ndcg = max(results[name]["summary"][SELECTION_METRIC]["mean"] for name in names)
    tied = [
        name
        for name in names
        if best_ndcg - results[name]["summary"][SELECTION_METRIC]["mean"]
        <= SELECTION_METRIC_TOLERANCE
    ]
    if "raw" in tied:
        return "raw", "highest dev nDCG@10 within 1e-12; raw wins ties"
    selected = max(tied, key=lambda name: results[name]["summary"][SELECTION_METRIC]["mean"])
    return selected, "highest dev nDCG@10 (tolerance 1e-12); raw wins ties"


def paired_comparisons(results: dict[str, dict]) -> dict[str, dict[str, dict]]:
    """Compare every non-raw prompt with raw by query, including Holm correction."""
    raw_rows = results["raw"]["rows"]
    comparisons = {}
    for spec in PROMPT_SPECS:
        if spec.name == "raw":
            continue
        comparisons[spec.name] = {}
        candidate_rows = results[spec.name]["rows"]
        for metric in SUMMARY_METRICS:
            test = paired_difference(
                [row[metric] for row in raw_rows],
                [row[metric] for row in candidate_rows],
                repetitions=10000,
                seed=5523,
            )
            comparisons[spec.name][metric] = {
                "difference": test["difference"],
                "lower": test["lower"],
                "upper": test["upper"],
                "p_value": test["p_value"],
                "n": test["n"],
                "zero_differences": test["zero_differences"],
                "direction": "template - raw",
                "bootstrap_repetitions": 2000,
                "permutation_repetitions": 10000,
                "seed": 5523,
            }

    for metric in SUMMARY_METRICS:
        names = [spec.name for spec in PROMPT_SPECS if spec.name != "raw"]
        adjusted = holm_adjust([comparisons[name][metric]["p_value"] for name in names])
        for name, p_value in zip(names, adjusted, strict=True):
            comparisons[name][metric]["p_value_holm"] = p_value
            comparisons[name][metric]["holm_family"] = f"{len(names)} templates x {metric}"
    return comparisons


def _gallery_paths(
    split_dir: Path,
    image_vectors: np.ndarray,
    index_ids: list[str],
) -> tuple[list[str], np.ndarray, dict[str, dict]]:
    gallery_ids = json.loads((split_dir / "gallery_ids.json").read_text(encoding="utf-8"))
    if len(gallery_ids) != len(set(gallery_ids)):
        raise ValueError(f"Duplicate gallery item IDs in {split_dir}")
    source_rows = {item_id: index for index, item_id in enumerate(index_ids)}
    missing = [item_id for item_id in gallery_ids if item_id not in source_rows]
    if missing:
        raise ValueError(f"Gallery items missing from index: {missing[:5]}")
    gallery = image_vectors[[source_rows[item_id] for item_id in gallery_ids]]
    qrels_rows = read_jsonl(split_dir / "qrels.jsonl")
    qrels = {row["qid"]: row["grades"] for row in qrels_rows}
    if len(qrels) != len(qrels_rows):
        raise ValueError(f"Duplicate qrels query IDs in {split_dir}")
    expected = set(gallery_ids)
    malformed = [qid for qid, grades in qrels.items() if set(grades) != expected]
    if malformed:
        raise ValueError(f"Qrels do not cover exactly the gallery: {malformed[:5]}")
    return gallery_ids, gallery, qrels


def evaluate_split(
    encoder: Encoder,
    image_vectors: np.ndarray,
    index_ids: list[str],
    split_dir: Path,
    languages: tuple[str, ...] = ("en", "zh"),
) -> dict:
    gallery_ids, gallery, qrels = _gallery_paths(split_dir, image_vectors, index_ids)
    queries = read_jsonl(split_dir / "queries.jsonl")
    if len({query["qid"] for query in queries}) != len(queries):
        raise ValueError(f"Duplicate query IDs in {split_dir}")
    expected_qids = set(qrels)
    if {query["qid"] for query in queries} != expected_qids:
        raise ValueError(f"Query and qrels IDs differ in {split_dir}")

    names = [spec.name for spec in PROMPT_SPECS]
    output = {}
    for intent in languages:
        field = f"text_{intent}"
        prompt_rows = [
            render_prompt(query[field], name) for query in queries for name in names
        ]
        embeddings = encoder.encode_texts(prompt_rows).astype(np.float32, copy=False)
        if embeddings.shape != (len(prompt_rows), image_vectors.shape[1]):
            raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
        results = {}
        for variant_index, name in enumerate(names):
            text_vectors = embeddings[variant_index::len(names)]
            scores = text_vectors @ gallery.T
            rows = []
            for query_index, query in enumerate(queries):
                order = np.lexsort((np.asarray(gallery_ids), -scores[query_index]))
                ranking = [gallery_ids[index] for index in order]
                qrels_for_query = qrels[query["qid"]]
                rows.append(
                    {
                        "qid": query["qid"],
                        **ranking_metrics(ranking, qrels_for_query),
                        "top10": ranking[:10],
                    }
                )
            results[name] = {"summary": summarize_rows(rows), "rows": rows}
        output[intent] = {
            "split": split_dir.name,
            "gallery_size": len(gallery_ids),
            "queries": len(queries),
            "no_op_transformations": {
                name: sum(render_prompt(query[field], name) == query[field] for query in queries)
                for name in names
            },
            "templates": results,
            "comparisons_to_raw": paired_comparisons(results),
        }
    return output


def _load_experiment_inputs(index_dir: Path, device: str):
    index_manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    index_ids = json.loads((index_dir / "item_ids.json").read_text(encoding="utf-8"))
    image_vectors = np.load(index_dir / "image_embeddings.npy", allow_pickle=False)
    if image_vectors.shape[0] != len(index_ids):
        raise ValueError("Image embedding count and item ID count differ")
    norms = np.linalg.norm(image_vectors, axis=1, keepdims=True)
    if not np.allclose(norms, 1.0, atol=1e-4):
        image_vectors = image_vectors / norms
    encoder = Encoder("siglip", device=device)
    expected = index_manifest.get("encoder_metadata", {})
    for field in ("model_id", "revision", "adapter_hash", "normalization"):
        if encoder.metadata.get(field) != expected.get(field):
            raise ValueError(
                f"Encoder/index mismatch for {field}: {encoder.metadata.get(field)!r} != {expected.get(field)!r}"
            )
    return encoder, index_manifest, index_ids, image_vectors


def _package_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "torch", "transformers"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def _dump_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))


def _write_summary_csv(path: Path, split_results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("split", "language", "template", "family", "metric", "n", "mean", "lower", "upper"),
        )
        writer.writeheader()
        for language, language_results in split_results.items():
            for spec in PROMPT_SPECS:
                summary = language_results["templates"][spec.name]["summary"]
                for metric in SUMMARY_METRICS:
                    interval = summary[metric]
                    writer.writerow(
                        {
                            "split": language_results["split"],
                            "language": language,
                            "template": spec.name,
                            "family": spec.family,
                            "metric": metric,
                            "n": interval["n"],
                            "mean": interval["mean"],
                            "lower": interval["lower"],
                            "upper": interval["upper"],
                        }
                    )


def _write_comparison_csv(path: Path, split_results: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "split", "language", "template", "family", "metric", "n", "difference",
                "lower", "upper", "p_value", "p_value_holm",
            ),
        )
        writer.writeheader()
        for language, language_results in split_results.items():
            for spec in PROMPT_SPECS:
                if spec.name == "raw":
                    continue
                for metric in SUMMARY_METRICS:
                    comparison = language_results["comparisons_to_raw"][spec.name][metric]
                    writer.writerow(
                        {
                            "split": language_results["split"],
                            "language": language,
                            "template": spec.name,
                            "family": spec.family,
                            "metric": metric,
                            "n": comparison["n"],
                            "difference": comparison["difference"],
                            "lower": comparison["lower"],
                            "upper": comparison["upper"],
                            "p_value": comparison["p_value"],
                            "p_value_holm": comparison["p_value_holm"],
                        }
                    )


def _write_query_metrics_csv(path: Path, split_results: dict) -> None:
    """Retain every query/template score as a flat audit table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("split", "language", "template", "qid", *SUMMARY_METRICS),
        )
        writer.writeheader()
        for language, language_results in split_results.items():
            for template, template_results in language_results["templates"].items():
                for row in template_results["rows"]:
                    writer.writerow(
                        {
                            "split": language_results["split"],
                            "language": language,
                            "template": template,
                            "qid": row["qid"],
                            **{metric: row[metric] for metric in SUMMARY_METRICS},
                        }
                    )


def _write_report_tables(path: Path, split: str, split_results: dict, selected: dict) -> None:
    """Write a compact report table for every template and the fixed selection."""
    lines = [
        f"# {split.capitalize()} prompt metrics",
        "",
        "All intervals are paired-query bootstrap 95% percentile intervals (2000 repetitions).",
        "Δ and p compare each template with raw; Holm adjusts the nDCG@10 family of all non-raw templates.",
        "",
    ]
    for language, result in split_results.items():
        lines.extend(
            [
                f"## {language}",
                "",
                "| Template | Family | P@5 | nDCG@5 | nDCG@10 | AP@10 | Recall@10 | MRR@10 | ΔnDCG@10 [95% CI] | p | p Holm |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for spec in PROMPT_SPECS:
            summary = result["templates"][spec.name]["summary"]
            comparison = result["comparisons_to_raw"].get(spec.name, {}).get("nDCG@10")
            if comparison:
                delta = f"{comparison['difference']:+.4f} [{comparison['lower']:+.4f}, {comparison['upper']:+.4f}]"
                p_value = f"{comparison['p_value']:.4f}"
                p_holm = f"{comparison['p_value_holm']:.4f}"
            else:
                delta, p_value, p_holm = "—", "—", "—"
            values = " | ".join(f"{summary[metric]['mean']:.4f}" for metric in SUMMARY_METRICS)
            lines.append(
                f"| {spec.name} | {spec.family} | {values} | {delta} | {p_value} | {p_holm} |"
            )
        selected_name = selected[language]
        lines.extend(
            [
                "",
                f"Frozen selection: `{selected_name}`.",
                "",
                "| Metric | Raw | Selected | Δ [95% CI] | p | p Holm |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for metric in SUMMARY_METRICS:
            raw_summary = result["templates"]["raw"]["summary"][metric]
            selected_summary = result["templates"][selected_name]["summary"][metric]
            comparison = result["comparisons_to_raw"][selected_name][metric]
            lines.append(
                f"| {metric} | {raw_summary['mean']:.4f} | {selected_summary['mean']:.4f} | "
                f"{comparison['difference']:+.4f} [{comparison['lower']:+.4f}, {comparison['upper']:+.4f}] | "
                f"{comparison['p_value']:.4f} | {comparison['p_value_holm']:.4f} |"
            )
        lines.append("")
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def _write_focused_mechanism_table(path: Path, split_results: dict, prompts_for_split: dict) -> None:
    """Cross-case, punctuation, and sentence-ending variants under one comparison."""
    focus = ("lowercase", "capitalize_first", "terminal_period", "remove_terminal_punctuation")
    lines = [
        "# Focused case and punctuation mechanics",
        "",
        "Δ is template minus raw. `Input changed` counts literal prompt changes; `Embedding changed` counts",
        "L2 difference greater than 1e-7; `Top-10 changed` counts reordered candidate IDs.",
        "",
        "| Language | Template | Input changed | Embedding changed | Top-10 changed | ΔnDCG@10 [95% CI] | p | p Holm |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for language, result in split_results.items():
        raw_rows = result["templates"]["raw"]["rows"]
        for name in focus:
            rows = result["templates"][name]["rows"]
            mechanism = prompts_for_split[language][name]
            comparison = result["comparisons_to_raw"][name]["nDCG@10"]
            lines.append(
                f"| {language} | {name} | {mechanism['input_changed']}/{mechanism['n']} | "
                f"{mechanism['embedding_changed']}/{mechanism['n']} | "
                f"{sum(a['top10'] != b['top10'] for a, b in zip(raw_rows, rows, strict=True))}/{len(rows)} | "
                f"{comparison['difference']:+.4f} [{comparison['lower']:+.4f}, {comparison['upper']:+.4f}] | "
                f"{comparison['p_value']:.4f} | {comparison['p_value_holm']:.4f} |"
            )
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def _prompt_mechanism(encoder: Encoder, queries: list[dict], names: list[str]) -> dict:
    result = {}
    for language in ("en", "zh"):
        source_field = f"text_{language}"
        raw = [query[source_field] for query in queries]
        rendered = {
            name: [render_prompt(query[source_field], name) for query in queries] for name in names
        }
        raw_vectors = encoder.encode_texts(raw)
        language_result = {}
        for name in names:
            vectors = encoder.encode_texts(rendered[name])
            language_result[name] = {
                "n": len(queries),
                "input_changed": sum(a != b for a, b in zip(raw, rendered[name], strict=True)),
                "embedding_changed": int(
                    (np.linalg.norm(vectors - raw_vectors, axis=1) > 1e-7).sum()
                ),
                "mean_embedding_l2_delta": float(
                    np.linalg.norm(vectors - raw_vectors, axis=1).mean()
                ),
            }
        result[language] = language_result
    return result


def _write_output_manifest(output_dir: Path, output_files: list[str]) -> None:
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "files": {
            filename: sha256_file(output_dir / filename) for filename in sorted(output_files)
        },
    }
    _dump_json(output_dir / "artifact_manifest.json", manifest)


def verify_output_manifest(output_dir: Path) -> dict:
    path = output_dir / "artifact_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mismatches = []
    for filename, expected in manifest["files"].items():
        actual = sha256_file(output_dir / filename)
        if actual != expected:
            mismatches.append({"file": filename, "expected": expected, "actual": actual})
    if mismatches:
        raise ValueError(f"Artifact manifest mismatch: {mismatches}")
    return manifest


def _write_embedding_artifact(
    output_dir: Path,
    split_dir: Path,
    encoder: Encoder,
    image_vectors: np.ndarray,
    index_ids: list[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    queries = read_jsonl(split_dir / "queries.jsonl")
    names = [spec.name for spec in PROMPT_SPECS]
    languages = {}
    for intent in ("en", "zh"):
        vectors = encoder.encode_texts(
            [render_prompt(query[f"text_{intent}"], name) for query in queries for name in names]
        )
        np.save(output_dir / f"text_embeddings_{intent}.npy", vectors.astype(np.float32, copy=False))
        languages[intent] = {
            "query_ids": [query["qid"] for query in queries],
            "templates": names,
            "order": "query-major, then template registry order",
            "shape": list(vectors.shape),
            "dtype": "float32",
            "sha256": sha256_file(output_dir / f"text_embeddings_{intent}.npy"),
        }
    _dump_json(
        output_dir / "text_embeddings.json",
        {
            "encoder": encoder.metadata,
            "source_queries": str(split_dir.relative_to(ROOT)) + "/queries.jsonl",
            "source_queries_sha256": sha256_file(split_dir / "queries.jsonl"),
            "index_item_count": len(index_ids),
            "image_embedding_shape": list(image_vectors.shape),
            "languages": languages,
        },
    )


def write_selection_manifest(
    output_dir: Path,
    encoder: Encoder,
    index_dir: Path,
    image_vectors: np.ndarray,
    dev_results: dict,
) -> dict:
    selections, rules, dev_scores = {}, {}, {}
    for language, result in dev_results.items():
        selected, rule = select_variant(result["templates"])
        selections[language] = selected
        rules[language] = rule
        dev_scores[language] = {
            spec.name: {
                metric: result["templates"][spec.name]["summary"][metric]
                for metric in ("P@5", "nDCG@10")
            }
            for spec in PROMPT_SPECS
        }
    specs = [asdict(spec) for spec in PROMPT_SPECS]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "dev-selection-frozen-before-test",
        "encoder": encoder.metadata,
        "package_versions": _package_versions(),
        "index": {
            "path": str(index_dir.relative_to(ROOT)),
            "manifest_sha256": sha256_file(index_dir / "manifest.json"),
            "image_embeddings_sha256": sha256_file(index_dir / "image_embeddings.npy"),
            "image_embeddings_shape": list(image_vectors.shape),
        },
        "dev": {
            "path": "eval/visual_dev",
            "queries_sha256": sha256_file(ROOT / "eval/visual_dev/queries.jsonl"),
            "qrels_sha256": sha256_file(ROOT / "eval/visual_dev/qrels.jsonl"),
            "gallery_ids_sha256": sha256_file(ROOT / "eval/visual_dev/gallery_ids.json"),
            "query_count": dev_results["en"]["queries"],
            "gallery_size": dev_results["en"]["gallery_size"],
        },
        "prompt_registry_sha256": canonical_sha256(specs),
        "prompt_templates": specs,
        "selection_rule": {
            "primary_metric": SELECTION_METRIC,
            "tolerance": SELECTION_METRIC_TOLERANCE,
            "tie_break": "raw wins; otherwise deterministic registry order",
            "secondary_metrics_used_for_selection": False,
        },
        "selected": selections,
        "selection_notes": rules,
        "dev_scores": dev_scores,
        "inference": {
            "unit": "query",
            "paired_against": "raw",
            "difference": "template - raw",
            "bootstrap": "paired query resampling, 95% percentile interval, 2000 repetitions, seed 5523",
        "test": "two-sided sign-flip paired permutation with zero differences fixed, 10000 repetitions, seed 5523",
            "multiplicity": "Holm across all non-raw templates separately for each split/language/metric",
        "primary_metric_multiplicity": "Holm primary metric, all non-raw templates within each split and language",
        "secondary_metric_multiplicity": "Holm each metric within each split and language; metrics are separate outcome groups",
        },
        "test_policy": "Test is not read by select-dev; evaluate-test is a fixed supplementary evaluation because the split has prior exposure.",
    }
    payload = json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    manifest_path = output_dir / "selection_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(payload.encode("utf-8"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    (output_dir / "selection_manifest.sha256").write_bytes((digest + "\n").encode("ascii"))
    _dump_json(
        output_dir / "dev/results.json",
        {
            "schema_version": SCHEMA_VERSION,
            "selection_manifest_sha256": digest,
            "results": dev_results,
        },
    )
    _write_summary_csv(output_dir / "dev/metrics.csv", dev_results)
    _write_comparison_csv(output_dir / "dev/paired_comparisons.csv", dev_results)
    _write_query_metrics_csv(output_dir / "dev/query_metrics.csv", dev_results)
    _write_report_tables(
        output_dir / "dev/metrics.md", "development", dev_results, manifest["selected"]
    )
    return {**manifest, "manifest_sha256": digest}


def load_frozen_selection(output_dir: Path) -> dict:
    manifest_path = output_dir / "selection_manifest.json"
    digest_path = output_dir / "selection_manifest.sha256"
    payload = manifest_path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    expected = digest_path.read_text(encoding="ascii").strip()
    if actual != expected:
        raise ValueError("Selection manifest SHA-256 does not match its frozen digest")
    manifest = json.loads(payload)
    expected_registry = canonical_sha256([asdict(spec) for spec in PROMPT_SPECS])
    if manifest.get("prompt_registry_sha256") != expected_registry:
        raise ValueError("Prompt registry changed after the dev selection was frozen")
    return {**manifest, "manifest_sha256": actual}


def run_select_dev(args, encoder=None, index=None):
    index_dir = ROOT / args.index
    if encoder is None or index is None:
        encoder, _manifest, index_ids, image_vectors = _load_experiment_inputs(index_dir, args.device)
    else:
        index_ids, image_vectors = index
    output_dir = ROOT / args.output
    dev_results = evaluate_split(
        encoder, image_vectors, index_ids, ROOT / "eval/visual_dev"
    )
    manifest = write_selection_manifest(
        output_dir, encoder, index_dir, image_vectors, dev_results
    )
    _write_embedding_artifact(
        output_dir / "dev", ROOT / "eval/visual_dev", encoder, image_vectors, index_ids
    )
    dev_queries = read_jsonl(ROOT / "eval/visual_dev/queries.jsonl")
    mechanism = _prompt_mechanism(encoder, dev_queries, [spec.name for spec in PROMPT_SPECS])
    _dump_json(output_dir / "dev/mechanism.json", mechanism)
    _write_focused_mechanism_table(output_dir / "dev/mechanism.md", dev_results, mechanism)
    _write_output_manifest(
        output_dir,
        [
            "selection_manifest.json",
            "selection_manifest.sha256",
            "dev/results.json",
            "dev/metrics.csv",
            "dev/metrics.md",
            "dev/paired_comparisons.csv",
            "dev/query_metrics.csv",
            "dev/mechanism.json",
            "dev/mechanism.md",
            "dev/text_embeddings_en.npy",
            "dev/text_embeddings_zh.npy",
            "dev/text_embeddings.json",
        ],
    )
    print(json.dumps({"selected": manifest["selected"], "manifest_sha256": manifest["manifest_sha256"]}, indent=2))
    return encoder, (index_ids, image_vectors), manifest


def run_evaluate_test(args, encoder=None, index=None):
    output_dir = ROOT / args.output
    # Verify the frozen selection before this function reads any test query.
    manifest = load_frozen_selection(output_dir)
    index_dir = ROOT / args.index
    if encoder is None or index is None:
        encoder, _index_manifest, index_ids, image_vectors = _load_experiment_inputs(
            index_dir, args.device
        )
    else:
        index_ids, image_vectors = index
    if encoder.metadata.get("model_id") != manifest["encoder"].get("model_id"):
        raise ValueError("Encoder model differs from the frozen selection manifest")
    if encoder.metadata.get("revision") != manifest["encoder"].get("revision"):
        raise ValueError("Encoder revision differs from the frozen selection manifest")
    if sha256_file(index_dir / "image_embeddings.npy") != manifest["index"]["image_embeddings_sha256"]:
        raise ValueError("Image index differs from the frozen selection manifest")
    if canonical_sha256([asdict(spec) for spec in PROMPT_SPECS]) != manifest["prompt_registry_sha256"]:
        raise ValueError("Prompt registry differs from the frozen selection manifest")
    test_results = evaluate_split(
        encoder, image_vectors, index_ids, ROOT / "eval/visual_test"
    )
    _write_embedding_artifact(
        output_dir / "test", ROOT / "eval/visual_test", encoder, image_vectors, index_ids
    )
    test_queries = read_jsonl(ROOT / "eval/visual_test/queries.jsonl")
    mechanism = _prompt_mechanism(encoder, test_queries, [spec.name for spec in PROMPT_SPECS])
    _dump_json(output_dir / "test/mechanism.json", mechanism)
    _write_focused_mechanism_table(output_dir / "test/mechanism.md", test_results, mechanism)
    _dump_json(
        output_dir / "test/results.json",
        {
            "schema_version": SCHEMA_VERSION,
            "selection_manifest_sha256": manifest["manifest_sha256"],
            "selected": manifest["selected"],
            "evaluation_status": "fixed supplementary test after prior split exposure; not a newly blind holdout",
            "results": test_results,
        },
    )
    _write_summary_csv(output_dir / "test/metrics.csv", test_results)
    _write_comparison_csv(output_dir / "test/paired_comparisons.csv", test_results)
    _write_query_metrics_csv(output_dir / "test/query_metrics.csv", test_results)
    _write_report_tables(
        output_dir / "test/metrics.md", "supplementary test", test_results, manifest["selected"]
    )
    _write_output_manifest(
        output_dir,
        [
            "selection_manifest.json",
            "selection_manifest.sha256",
            "dev/results.json",
            "dev/metrics.csv",
            "dev/metrics.md",
            "dev/paired_comparisons.csv",
            "dev/query_metrics.csv",
            "dev/mechanism.json",
            "dev/mechanism.md",
            "dev/text_embeddings_en.npy",
            "dev/text_embeddings_zh.npy",
            "dev/text_embeddings.json",
            "test/results.json",
            "test/metrics.csv",
            "test/metrics.md",
            "test/paired_comparisons.csv",
            "test/query_metrics.csv",
            "test/mechanism.json",
            "test/mechanism.md",
            "test/text_embeddings_en.npy",
            "test/text_embeddings_zh.npy",
            "test/text_embeddings.json",
        ],
    )
    selected_test = {
        language: {
            "template": manifest["selected"][language],
            "P@5": test_results[language]["templates"][manifest["selected"][language]]["summary"]["P@5"],
            "nDCG@10": test_results[language]["templates"][manifest["selected"][language]]["summary"]["nDCG@10"],
        }
        for language in ("en", "zh")
    }
    print(json.dumps(selected_test, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("select-dev", "evaluate-test", "all", "verify"),
        default="select-dev",
    )
    parser.add_argument("--index", default="artifacts/siglip")
    parser.add_argument("--output", default="artifacts/prompt_calibration_20261108")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.phase == "verify":
        manifest = verify_output_manifest(ROOT / args.output)
        print(json.dumps({"verified_files": len(manifest["files"])}, indent=2))
        return
    encoder = index = None
    if args.phase in ("select-dev", "all"):
        encoder, index, _manifest = run_select_dev(args)
    if args.phase in ("evaluate-test", "all"):
        run_evaluate_test(args, encoder=encoder, index=index)


if __name__ == "__main__":
    main()
