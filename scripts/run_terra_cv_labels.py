"""Independent, image-attached Terra annotation; no catalog in the prompt."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "eval/cv_research_20261104"


def run(person):
    directory = BASE / f"annotator_{person}"
    if (directory / "labels.jsonl").exists():
        raise FileExistsError("Frozen labels already exist; use a new annotation directory")
    aliases = json.loads((directory / "aliases.json").read_text())
    value = {"type": ["integer", "null"], "enum": [0, 1, None]}

    def attrs(keys):
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": dict.fromkeys(keys, value),
            "required": keys,
        }

    properties = {
        "alias": {"type": "string", "enum": aliases},
        "object_type": {
            "type": "string",
            "enum": [
                "bowl",
                "plate",
                "vase",
                "jar",
                "cup",
                "teapot",
                "ewer",
                "pot",
                "figure",
                "mirror",
                "box",
                "textile",
                "ritual vessel",
                "pendant",
                "stele",
                "ornament",
                "cong",
                "bi",
                "lid",
                "other",
            ],
        },
        "colors": attrs(["blue", "white", "green", "red", "black", "yellow", "brown", "pink"]),
        "shape": attrs(["narrow_mouth", "wide_mouth", "handles", "footed", "round_body"]),
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "uncertainty": {"type": "string"},
    }
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "labels": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": properties,
                    "required": list(properties),
                },
            }
        },
        "required": ["labels"],
    }
    (directory / "schema.json").write_text(json.dumps(schema), encoding="utf8")
    prompt = (directory / "TASK.md").read_text(encoding="utf8")
    sheets = sorted(directory.glob("sheet_*.jpg"))
    prompt += f"\nAll {len(sheets)} sheets are attached in numbered order, with aliases above the images. Work only from these images, without using tools or reading any files. Return all {len(aliases)} labels as the schema JSON. Do not write files: the calling process saves your response.\n"
    command = [
        shutil.which("codex"),
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--model",
        "gpt-6.1-terra",
        "--sandbox",
        "read-only",
        "--json",
        "--output-schema",
        str(directory / "schema.json"),
        "--output-last-message",
        str(directory / "response.json"),
        "-C",
        str(directory),
    ]
    for path in sheets:
        command += ["--image", str(path)]
    command.append("-")
    logs = ROOT / "artifacts/cv_research_20261104/annotation_sessions" / person
    logs.mkdir(parents=True, exist_ok=True)
    with (
        (logs / "cli_events.jsonl").open("wb") as log,
        (logs / "cli_stderr.log").open("wb") as err,
    ):
        result = subprocess.run(
            command, input=prompt.encode("utf8"), stdout=log, stderr=err, timeout=1500
        )
    if result.returncode:
        raise RuntimeError(f"Terra {person} exited {result.returncode}; see CLI log")
    rows = json.loads((directory / "response.json").read_text(encoding="utf8"))["labels"]
    if sorted(r["alias"] for r in rows) != sorted(aliases):
        raise ValueError(f"Incomplete or duplicate aliases for {person}")
    for row in rows:
        row["annotator"] = f"gpt-6.1-terra-cli-{person}"
    (directory / "labels.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf8"
    )
    print(f"Terra {person}: {len(rows)} labels saved", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", choices=["a", "b", "qa", "both"], default="both")
    arg = parser.parse_args()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, ["a", "b"] if arg.person == "both" else [arg.person]))
