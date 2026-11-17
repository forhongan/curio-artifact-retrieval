"""Warm independent image-only color features using four CPU processes.

Run before color scans, not concurrently with another surface-cache writer.
No model decisions or annotation labels are read or written by this utility.
"""

from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace

from curio.retrieve.catalogue_query import catalogue_fact, compile_catalogue
from curio.retrieve.catalogue_visual import CatalogueVisualScorer
from curio.retrieve.joint_pixels import color_observability

ROOT = Path(__file__).resolve().parents[1]


def warm(paths):
    scorer = CatalogueVisualScorer(SimpleNamespace(root=ROOT, items=[]), SimpleNamespace(root=ROOT))
    for i, path in enumerate(paths):
        if color_observability(ROOT / path)["observable"]:
            scorer._surface(path)
        if i % 100 == 0:
            print(json.dumps({"part": paths[0], "processed": i, "total": len(paths)}), flush=True)
    return len(paths)


def main():
    items = map(json.loads, (ROOT / "data/snapshots/collection_v2.jsonl").read_text().splitlines())
    plans = [compile_catalogue(q) for q in ["红色碗", "绿色窄口瓷瓶"]]
    paths = [
        r["image_path"]
        for r in items
        if any(
            all(catalogue_fact(r, c)["status"] != "contradicted" for c in p.clauses) for p in plans
        )
    ]
    assert len(paths) == len(set(paths))
    with ProcessPoolExecutor(max_workers=4) as pool:
        counts = list(pool.map(warm, [paths[i::4] for i in range(4)]))
    print(json.dumps({"completed": sum(counts)}), flush=True)


if __name__ == "__main__":
    main()
