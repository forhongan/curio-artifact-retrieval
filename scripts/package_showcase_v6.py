"""Package the supplemental runtime, research records, and offline presentation."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "_release/showcase-v6"
TAG = "curio-showcase-v6-20261109"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runtime = [ROOT / "artifacts/relevance_v5" / name for name in [
        "boundary_production_lora_en.json", "boundary_production_lora_zh.json",
        "thresholds.json", "color_binding.json",
    ]]
    runtime += [ROOT / "artifacts/cv_research_20261104/fresh" / name for name in [
        "raw_color_heads.npz", "raw_color_predictions.npy",
    ]]
    runtime.append(ROOT / "artifacts/classical_cv/item_ids.json")
    runtime += list((ROOT / "artifacts/catalogue_set_runtime_v5").rglob("*.json"))
    runtime += list((ROOT / "artifacts/catalogue_set_runtime_v5").rglob("*.npz"))
    research = []
    for folder in ["relevance_v5", "classical_cv", "cv_backbones_20261104", "cv_research_20261104/fresh",
                   "backbone_catalogue_v4_semantic_20261108", "background_downstream_v4_20261108",
                   "background_downstream_338_v4_20261108", "bg_v4"]:
        base = ROOT / "artifacts" / folder
        research += [p for p in base.rglob("*") if p.is_file() and p.suffix in {".json", ".jsonl", ".npz", ".npy", ".csv", ".md"}]
    research += [p for p in (ROOT / "artifacts/reference_scenarios_v6").rglob("*.json") if "cache" not in p.parts]
    research += list((ROOT / "artifacts/reference_scenarios_v6").rglob("*.jsonl"))
    assets = []
    for name, paths, prefix in [
        ("curio-v6-runtime.zip", runtime, ROOT),
        ("curio-v6-research.zip", research, ROOT),
        ("curio-v6-presentation.zip", list((ROOT / "web/public/assets").rglob("*")), ROOT / "web/public/assets"),
    ]:
        files = sorted({p for p in paths if p.is_file()})
        path = OUT / name
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for p in files:
                z.write(p, p.relative_to(prefix).as_posix())
        assets.append({"name": name, "bytes": path.stat().st_size, "sha256": digest(path), "files": len(files)})
    report = ROOT / "output/pdf/Curio_Technical_Report_Draft.pdf"
    if report.exists():
        target = OUT / report.name
        target.write_bytes(report.read_bytes())
        assets.append({"name": target.name, "bytes": target.stat().st_size, "sha256": digest(target)})
    manifest = {"schema": "curio-showcase-supplement-v1", "repository": "forhongan/curio-artifact-retrieval",
                "tag": TAG, "created_at": datetime.now(timezone.utc).isoformat(),
                "base_resources": "curio-resources-v2-20261104", "assets": assets,
                "runtime_files": [{"path": p.relative_to(ROOT).as_posix(), "sha256": digest(p), "bytes": p.stat().st_size} for p in runtime]}
    (OUT / "resources_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "SHA256SUMS.txt").write_text("".join(f"{x['sha256']}  {x['name']}\n" for x in assets), encoding="utf-8")
    (ROOT / "docs/showcase_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(assets, indent=2))


if __name__ == "__main__":
    main()
