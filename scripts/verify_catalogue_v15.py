"""Run the actual online scorer over the entire gallery, with bilingual replay.

These are engineering/regression checks, not independent human ground truth.
The same completed jobs can subsequently be served by the application.
"""

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

from curio.retrieve.catalogue_visual import CatalogueVisualScorer
from curio.retrieve.joint_sets import JointSetJobs
from curio.retrieve.joint_vlm import LocalVQA

ROOT = Path(__file__).resolve().parents[1]
QUERIES = {
    "meiping": ("梅瓶", "meiping vase"),
    "gourd": ("葫芦瓶", "gourd-shaped vase"),
    "gourd_double": ("双腹葫芦瓶", "a double-gourd vase"),
    "gourd_triple": ("三节葫芦瓶", "a triple-gourd vase"),
    "red_bowl": ("红色碗", "a red bowl"),
    "blue_white": ("青花瓷", "blue and white porcelain"),
    "green_narrow": ("绿色窄口瓷瓶", "green narrow-mouth porcelain vase"),
    "green_vase": ("绿色瓷瓶", "green porcelain vase"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", nargs="+", choices=QUERIES, default=list(QUERIES))
    args = parser.parse_args()
    items = list(
        map(json.loads, (ROOT / "data/snapshots/collection_v2.jsonl").read_text().splitlines())
    )
    service = SimpleNamespace(root=ROOT, items=items, by_id={r["item_id"]: r for r in items})
    visual = LocalVQA(ROOT)
    scorer = CatalogueVisualScorer(service, visual)
    jobs = JointSetJobs(service, scorer, method="catalogue_visual")
    output = ROOT / "artifacts/catalogue_visual_v15/verification"
    output.mkdir(parents=True, exist_ok=True)
    try:
        for name in args.queries:
            zh, en = QUERIES[name]
            assert scorer.plan(zh).key == scorer.plan(en).key, (
                name,
                "different bilingual meanings",
            )
            result = jobs.create(zh)
            ident = result["job_id"]
            while not result["complete"]:
                print(
                    json.dumps(
                        {
                            "query": name,
                            **{
                                k: result.get(k)
                                for k in (
                                    "status",
                                    "processed",
                                    "total",
                                    "matched_count",
                                    "elapsed_ms",
                                    "error",
                                )
                            },
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                if result["status"] in {"failed", "cancelled"}:
                    raise RuntimeError(result)
                time.sleep(10)
                result = jobs.get(ident)
            result = jobs.get(ident, include_decisions=True)
            assert jobs.create(en)["job_id"] == ident
            assert len(result["decisions"]) == len(items)
            assert {r["item_id"] for r in result["decisions"]} == set(service.by_id)
            assert all(type(r["yes"]) is bool for r in result["decisions"])
            assert result["matched_count"] + result["rejected_count"] == len(items)
            result.update(bilingual_aliases=[zh, en], bilingual_same_job=True)
            (output / f"{name}.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {
                        "query": name,
                        "status": "verified",
                        "yes": result["matched_count"],
                        "no": result["rejected_count"],
                        "job_id": ident,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    finally:
        jobs.close()
        visual.close()


if __name__ == "__main__":
    main()
