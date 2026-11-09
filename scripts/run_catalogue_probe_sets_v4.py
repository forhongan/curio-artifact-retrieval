"""Evaluate per-class visual probes as explicit yes/no catalogue sets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curio.evaluate.catalogue_boundary import build_query_specs, query_dataset  # noqa: E402
from curio.evaluate.probe_sets import (  # noqa: E402
    explicit_probe_evaluation,
    supported_atoms,
    two_stage_summary,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_probe_probabilities(root: Path, vectors: np.ndarray, report: dict) -> dict:
    probabilities = {}
    for task, task_report in report["tasks"].items():
        classes = list(task_report["models"])
        task_report["classes"] = classes
        columns = []
        for value in classes:
            artifact = task_report["models"][value]["artifact"]
            with np.load(root / artifact, allow_pickle=False) as model:
                weights, bias = model["weights"], model["bias"]
                # Binary one-vs-rest probes use sklearn's positive column.
                column = expit(vectors @ weights[-1] + bias[-1])
            columns.append(column)
        probabilities[task] = np.stack(columns, axis=1).astype(np.float32)
    return probabilities


def prepare(rows: list[dict], vectors: np.ndarray, probabilities: dict,
            probe_report: dict, specs: list[dict], split: str) -> list[dict]:
    prepared = []
    for spec in specs:
        data = query_dataset(rows, spec, split)
        # Probabilities are prepared for the tri-state eligible candidate set;
        # no candidate's catalogue labels guide the threshold decision.
        subset = {task: values[data["indices"]] for task, values in probabilities.items()}
        prepared.append({"spec": spec, "data": {**data, "probabilities": subset}})
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", default=(
        "artifacts/cv_v4_20261108/catalogue_v3_primary_plus_candidates/"
        "labels_catalogue_v2.jsonl"
    ))
    parser.add_argument("--probes", default="artifacts/cv_v4_20261108/probes_siglip")
    parser.add_argument("--output", default="artifacts/cv_v4_20261108/probe_sets")
    args = parser.parse_args()
    labels_path, probes_path = ROOT / args.labels, ROOT / args.probes
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(labels_path)
    ids = [str(value) for value in json.loads(
        (ROOT / "artifacts/siglip/item_ids.json").read_text(encoding="utf-8")
    )]
    by_id = {row["item_id"]: row for row in rows}
    aligned = [by_id[item_id] for item_id in ids]
    vectors = np.load(ROOT / "artifacts/siglip/image_embeddings.npy", allow_pickle=False)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    probe_report = json.loads((probes_path / "report.json").read_text(encoding="utf-8"))
    probabilities = load_probe_probabilities(probes_path, vectors, probe_report)
    specs = build_query_specs(aligned)
    supported = [spec for spec in specs
                 if supported_atoms(spec.atoms, probe_report)[0]]
    unsupported = [{"qid": spec.qid, "family": spec.family,
                    "missing_atoms": supported_atoms(spec.atoms, probe_report)[1]}
                   for spec in specs if not supported_atoms(spec.atoms, probe_report)[0]]
    dev = prepare(aligned, vectors, probabilities, probe_report, supported, "dev")
    test = prepare(aligned, vectors, probabilities, probe_report, supported, "test")
    thresholds = tuple(float(value) for value in np.linspace(0, .95, 20))
    selected_rule = False
    dev_by_gate = {}
    for require_class_winner in (False, True):
        for gate in thresholds:
            rows = explicit_probe_evaluation(
                dev, probe_report, gate_threshold=gate,
                require_class_winner=require_class_winner,
            )
            dev_by_gate[(require_class_winner, gate)] = rows
    # Select the rule and gate jointly on dev, with the conservative class
    # winner rule winning ties.
    choices = []
    for (require_class_winner, gate), rows in dev_by_gate.items():
        summary = two_stage_summary(rows)
        choices.append((summary["gate_f1"], summary["macro_set_score"],
                        summary["macro_f1"], int(require_class_winner),
                        -gate, require_class_winner, gate, summary))
    *_, require_class_winner, selected_gate, dev_summary = max(choices)
    selected_rule = bool(require_class_winner)
    dev_rows = dev_by_gate[(selected_rule, selected_gate)]
    test_rows = explicit_probe_evaluation(
        test, probe_report, gate_threshold=selected_gate,
        require_class_winner=selected_rule,
    )
    report = {
        "version": "catalogue-probe-sets-v4",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "policy": (
            "Per-class probe threshold and query gate frozen on dev only; "
            "test is a fixed supplementary evaluation after earlier exposure."
        ),
        "labels_path": labels_path.relative_to(ROOT).as_posix(),
        "labels_sha256": digest(labels_path),
        "probe_report": (probes_path / "report.json").relative_to(ROOT).as_posix(),
        "probe_report_sha256": digest(probes_path / "report.json"),
        "image_index_sha256": digest(ROOT / "artifacts/siglip/image_embeddings.npy"),
        "query_count": len(specs), "supported_query_count": len(supported),
        "unsupported_queries": unsupported,
        "threshold_source": "dev-selected per-class probe thresholds in probe report",
        "gate": {"threshold": selected_gate, "require_class_winner": selected_rule,
                 "selection_metric": "gate F1 then macro set score; dev only"},
        "dev": {"summary": two_stage_summary(dev_rows), "rows": dev_rows},
        "test": {"summary": two_stage_summary(test_rows), "rows": test_rows},
        "threshold_sweep_dev": {
            f"class_winner={winner},gate={gate}": two_stage_summary(rows)
            for (winner, gate), rows in dev_by_gate.items()
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"gate": report["gate"], "dev": report["dev"]["summary"],
                      "test": report["test"]["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
