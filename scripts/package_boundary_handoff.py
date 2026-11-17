"""Package Windows replay and live inference resources from verified v15 outputs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
TAG = "curio-boundary-v15-20261117"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, required=True, help="Original verified research workspace"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "_release" / TAG)
    parser.add_argument(
        "--repository",
        default="forhongan/curio-artifact-retrieval",
        help="GitHub owner/repo hosting the release assets",
    )
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    stage = out / "windows"
    data = stage / "data/boundary_v15"
    (data / "images").mkdir(parents=True, exist_ok=True)
    gallery = list(
        map(json.loads, (source / "data/snapshots/collection_v2.jsonl").read_text().splitlines())
    )
    items = []
    fields = [
        "item_id",
        "title",
        "source",
        "source_url",
        "medium",
        "period",
        "culture",
        "description",
        "object_name",
        "image_sha256",
    ]
    for row in gallery:
        item = {k: row.get(k) for k in fields}
        item.update(
            image_url=f"/api/images/{row['item_id']}",
            thumb_url=f"/api/images/{row['item_id']}?thumb=true",
        )
        items.append(item)

    def preview(row):
        with Image.open(source / row["image_path"]) as loaded:
            pic = ImageOps.exif_transpose(loaded).convert("RGB")
            pic.thumbnail((384, 384), Image.Resampling.LANCZOS)
            pic.save(data / "images" / f"{row['item_id']}.jpg", quality=85, optimize=True)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(preview, gallery))
    queries = []
    for path in sorted((source / "artifacts/catalogue_visual_v15/verification").glob("*.json")):
        report = json.loads(path.read_text())
        if (
            not report["complete"]
            or report["total"] != len(items)
            or not report["bilingual_same_job"]
        ):
            raise ValueError(f"Unverified results: {path.name}")
        report["original_job_id"] = report["job_id"]
        report["job_id"] = "snapshot_v15_" + report["plan"]["key"][:32]
        queries.append({"aliases": report["bilingual_aliases"], "report": report})
    bundle = {
        "schema": "curio-boundary-snapshot-v1",
        "version": TAG,
        "items": items,
        "queries": queries,
        "provenance": "Original-image v15 decisions; 384px JPEGs here are display-only previews.",
    }
    encoded = json.dumps(bundle, ensure_ascii=False, separators=(",", ":")).encode()
    (data / "snapshot.json.gz").write_bytes(gzip.compress(encoded, mtime=0))
    write_json(
        data / "manifest.json",
        {
            "schema": bundle["schema"],
            "snapshot_sha256": digest(data / "snapshot.json.gz"),
            "items": len(items),
            "queries": len(queries),
        },
    )
    code = [
        "src/curio/__init__.py",
        "src/curio/serve/__init__.py",
        "src/curio/serve/boundary_snapshot.py",
        "src/curio/retrieve/__init__.py",
        "src/curio/retrieve/catalogue_query.py",
        "src/curio/retrieve/joint_query.py",
        "src/curio/retrieve/condition_parser.py",
        "src/curio/retrieve/condition_boundary.py",
        "requirements-boundary-demo.txt",
        "scripts/run_boundary_windows.ps1",
        "scripts/verify_boundary_handoff.py",
        "run-boundary-windows.cmd",
        "integration/boundary-client.ts",
        "docs/BOUNDARY_FRONTEND_HANDOFF.md",
    ]
    for name in code:
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    (stage / "START_HERE.txt").write_text(
        "Install Python 3.12 for Windows. Double-click run-boundary-windows.cmd.\nAPI: http://127.0.0.1:8015/docs\nRead docs/BOUNDARY_FRONTEND_HANDOFF.md for integration and the snapshot limitation.\n",
        encoding="utf-8",
    )
    runtime = out / "runtime"
    resources = [
        "data/snapshots/collection_v2.jsonl",
        "artifacts/factor_boundary_v11/heads.json",
        "artifacts/surface_boundary_v13/red_surface_head.json",
        "artifacts/targeted_boundary_v10/support_followup/meiping_gallery.jsonl",
        "artifacts/targeted_boundary_v10/support_followup/gourd_gallery.jsonl",
    ]
    for name in resources:
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    receipt = json.loads((source / "artifacts/joint_vlm_v9/model_receipt.json").read_text())
    receipt["local_dir"] = "models/vlm_qwen3_8b_4bit"
    write_json(runtime / "artifacts/joint_vlm_v9/model_receipt.json", receipt)
    assets = []
    for folder, filename, mode in [
        (stage, "curio-boundary-v15-windows.zip", "snapshot_replay"),
        (runtime, "curio-boundary-v15-runtime.zip", "live_resources"),
    ]:
        target = out / filename
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(folder.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(folder).as_posix())
        assets.append(
            {
                "name": filename,
                "bytes": target.stat().st_size,
                "sha256": digest(target),
                "mode": mode,
            }
        )
    manifest = {
        "repository": args.repository,
        "source_repository": "forhongan/curio-artifact-retrieval",
        "tag": TAG,
        "assets": assets,
        "gallery_count": len(items),
        "query_count": len(queries),
        "frontend_modified": False,
        "live_backend": "Apple Silicon MLX; Windows package replays completed decisions only",
    }
    write_json(out / "resources_manifest.json", manifest)
    write_json(ROOT / "docs/BOUNDARY_RELEASE_MANIFEST.json", manifest)
    (out / "SHA256SUMS.txt").write_text(
        "".join(f"{a['sha256']}  {a['name']}\n" for a in assets), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
