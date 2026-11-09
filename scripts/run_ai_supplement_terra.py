"""Run catalogue-first Terra observations for an explicit review-queue batch.

Each task receives its complete raw museum record and original image.  The
runner permits model tools and read-only access to the complete workspace, so
the annotator can inspect crops, pixels, and documented registry resources.
Official catalogue labels and frozen scores are read-only inputs; all outputs
are written to a new append-only run directory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = ROOT / "scripts/build_cv_v4_catalogue_v2.py"
DEFAULT_CONTRACT = ROOT / "artifacts/cv_v4_20261108/catalogue_v2_ai_supplement"
DEFAULT_QUEUE = DEFAULT_CONTRACT / "ai_review_queue.jsonl"
DEFAULT_OUTPUT = ROOT / "artifacts/ai_supplement_terra_20261108_batch01"


def load_builder():
    spec = importlib.util.spec_from_file_location("ai_supplement_builder", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BUILDER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_inputs(item: dict) -> tuple[Path, Path, dict]:
    image = ROOT / item["image_path"]
    record_path = ROOT / item["catalog_record_path"]
    if not image.is_file() or sha256(image) != item["image_sha256"]:
        raise ValueError(f"Image unavailable or changed: {item['item_id']}")
    if not record_path.is_file() or sha256(record_path) != item["catalog_record_sha256"]:
        raise ValueError(f"Catalogue record unavailable or changed: {item['item_id']}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    return image, record_path, record


def prompt_for(item: dict, record: dict, queue_path: Path, schema_path: Path) -> str:
    base_prompt = (schema_path.parent / "ai_supplement_prompt.md").read_text(encoding="utf-8")
    input_payload = {
        "item_id": item["item_id"],
        "image_path": item["image_path"],
        "image_sha256": item["image_sha256"],
        "catalog_record_path": item["catalog_record_path"],
        "catalog_record_sha256": item["catalog_record_sha256"],
        "priority_reasons": item["priority_reasons"],
        "queue_path": str(queue_path.relative_to(ROOT)),
    }
    return (
        base_prompt
        + "\n\n## Concrete task\n\n"
        + "Use the attached original image and the complete raw record below. The image and record are also "
        + "available at the listed read-only workspace paths. You may run read-only tools, inspect pixels or "
        + "crops, and consult documented open-access registry resources. Return only the JSON object. Use this "
        + "exact item identity and provenance; set `reviewer` to `gpt-6.1-terra` and `created_at` to the current "
        + "UTC ISO-8601 time.\n\n"
        + "```json\n"
        + json.dumps(input_payload, ensure_ascii=False, indent=2)
        + "\n```\n\n## Complete raw catalogue record\n\n```json\n"
        + json.dumps(record, ensure_ascii=False, indent=2)
        + "\n```\n"
    )


def run_item(item: dict, queue_path: Path, output: Path, model: str, view: bool,
             schema_path: Path) -> dict:
    directory = output / "tasks" / item["item_id"]
    directory.mkdir(parents=True, exist_ok=False)
    image, record_path, record = verify_inputs(item)
    prompt = prompt_for(item, record, queue_path, schema_path)
    (directory / "prompt.md").write_text(prompt, encoding="utf-8")
    (directory / "input_manifest.json").write_text(
        json.dumps({
            **item,
            "image_sha256_verified": sha256(image),
            "catalog_record_sha256_verified": sha256(record_path),
            "complete_record_supplied": True,
            "original_image_supplied": True,
            "tools_allowed": True,
            "read_only_workspace": True,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    schema = schema_path
    response_path = directory / "response.json"
    command = [
        shutil.which("codex") or "",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--model",
        model,
        "-c",
        'model_reasoning_effort="high"',
        "-c",
        "features.view_image=true" if view else "features.view_image=false",
        "--sandbox",
        "read-only",
        "--json",
        "--output-schema",
        str(schema),
        "--output-last-message",
        str(response_path),
        "-C",
        str(ROOT),
        "--image",
        str(image),
        "-",
    ]
    (directory / "command.json").write_text(
        json.dumps({"argv": command, "cwd": str(ROOT), "model": model, "sandbox": "read-only",
                    "timed_out": False}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    started = time.perf_counter()
    timed_out = False
    with (
        (directory / "cli_events.jsonl").open("wb") as events,
        (directory / "cli_stderr.log").open("wb") as errors,
    ):
        try:
            result = subprocess.run(command, input=prompt.encode("utf-8"), stdout=events,
                                    stderr=errors, timeout=1200)
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            returncode = -1
            timed_out = True
    elapsed = round(time.perf_counter() - started, 3)
    command_manifest = json.loads((directory / "command.json").read_text(encoding="utf-8"))
    command_manifest.update({"returncode": returncode, "timed_out": timed_out, "elapsed_seconds": elapsed})
    (directory / "command.json").write_text(
        json.dumps(command_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {"item_id": item["item_id"], "returncode": returncode, "timed_out": timed_out,
            "elapsed_seconds": elapsed, "response_exists": response_path.is_file()}


def write_training_queue(output: Path, queue: Path, builder) -> dict:
    """Join accepted observations to their original catalogue without editing it."""
    validation = json.loads((output / "validation.json").read_text(encoding="utf-8"))
    accepted = {row["item_id"]: row for row in validation["results"] if row["accepted"]}
    selected_ids = set(accepted)
    queue_rows = {row["item_id"]: row for row in read_jsonl(queue)
                  if row["item_id"] in selected_ids}
    labels = read_jsonl(queue.parent / "labels_catalogue_v2.jsonl")
    labels_by_id = {row["item_id"]: row for row in labels if row["item_id"] in selected_ids}
    paths = []
    for item_id in sorted(selected_ids):
        queue_row = queue_rows[item_id]
        label = labels_by_id[item_id]
        result = accepted[item_id]
        response = json.loads((output / "tasks" / item_id / "response.json").read_text(encoding="utf-8"))
        paths.append({
            "task": "catalogue_first_visible_observations",
            "item_id": item_id,
            "split": label["split"],
            "object_group": label["object_group"],
            "image_path": label["image_path"],
            "image_sha256": label["image_sha256"],
            "catalog_record_path": label["official_catalog"]["record_path"],
            "catalog_record_sha256": queue_row["catalog_record_sha256"],
            "official_catalogue": label["official_catalog"],
            "visible_observations": result["claims"],
            "no_supplement_reason": response.get("no_supplement_reason"),
            "reviewer": response["reviewer"],
            "created_at": response["created_at"],
            "authority": {
                "official_catalogue": "immutable_primary_supervision",
                "visible_observations": "append_only_ai_image_observations_review_required",
                "official_fields_overwritten": False,
            },
        })
    path = output / "training_queue.jsonl"
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in paths),
                    encoding="utf-8")
    return {"path": str(path), "records": len(paths), "claims": sum(
        len(row["visible_observations"]) for row in paths), "sha256": sha256(path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, default=DEFAULT_QUEUE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--model", default="gpt-6.1-terra")
    parser.add_argument("--schema", type=Path,
                        default=DEFAULT_CONTRACT / "ai_supplement.schema.json")
    parser.add_argument("--view", action="store_true",
                        help="Enable the Codex image viewer for model-side crop/pixel inspection.")
    parser.add_argument("--assemble-only", action="store_true",
                        help="Build training_queue.jsonl from an existing validated run.")
    args = parser.parse_args(argv)
    queue = args.queue.resolve()
    output = args.output.resolve()
    schema = args.schema.resolve()
    if args.assemble_only:
        assembled = write_training_queue(output, queue, load_builder())
        print(json.dumps(assembled, ensure_ascii=False, indent=2), flush=True)
        return 0
    if output.exists():
        raise FileExistsError(f"Append-only output already exists: {output}")
    if shutil.which("codex") is None:
        raise RuntimeError("Codex CLI is unavailable")
    if not schema.is_file():
        raise FileNotFoundError(f"Output schema is unavailable: {schema}")
    output.mkdir(parents=True)
    selected = read_jsonl(queue)[:max(0, args.limit)]
    run = {
        "created_at": datetime.now(timezone.utc).isoformat(), "queue": str(queue),
        "queue_sha256": sha256(queue), "model": args.model, "sandbox": "read-only",
        "schema": str(schema), "schema_sha256": sha256(schema),
        "prompt": str(schema.parent / "ai_supplement_prompt.md"),
        "prompt_sha256": sha256(schema.parent / "ai_supplement_prompt.md"),
        "codex_cli": shutil.which("codex"),
        "codex_cli_version": subprocess.run([shutil.which("codex") or "codex", "--version"],
                                           capture_output=True, text=True, check=False).stdout.strip(),
        "tools_allowed": True, "view_image": args.view, "items_requested": len(selected),
        "original_images_supplied": True, "complete_records_supplied": True,
    }
    (output / "run_manifest.json").write_text(json.dumps(run, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    results = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(run_item, item, queue, output, args.model, args.view, schema)
                   for item in selected]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)
    results.sort(key=lambda row: row["item_id"])
    (output / "task_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n",
                                              encoding="utf-8")
    responses = []
    for item in selected:
        path = output / "tasks" / item["item_id"] / "response.json"
        if path.is_file():
            responses.append(json.loads(path.read_text(encoding="utf-8")))
    raw_path = output / "responses.jsonl"
    raw_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in responses),
                        encoding="utf-8")
    builder = load_builder()
    labels = builder.read_jsonl(queue.parent / "labels_catalogue_v2.jsonl")
    by_id = {row["item_id"]: row for row in labels}
    validation = builder.validate_supplement_file(raw_path, by_id, ROOT, phase="model")
    (output / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")
    training = (write_training_queue(output, queue, builder)
                if validation["accepted"] else None)
    summary = {
        **run, "responses": len(responses), "validated_items": len(validation["results"]),
        "accepted_items": sum(result["accepted"] for result in validation["results"]),
        "claims": validation["accepted_claims"],
        "all_accepted": bool(selected) and len(responses) == len(selected) and validation["accepted"],
        "task_results": results, "responses_sha256": sha256(raw_path) if raw_path.exists() else None,
        "validation_sha256": sha256(output / "validation.json"),
        "training_queue": training,
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["all_accepted"] and len(responses) == len(selected) else 1


if __name__ == "__main__":
    raise SystemExit(main())
