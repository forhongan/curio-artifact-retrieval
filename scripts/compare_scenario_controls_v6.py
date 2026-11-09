"""Add unchanged production-query rewriting controls to all real reference cases."""

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from curio.query import parse_query
from curio.represent.encoders import Encoder
from curio.retrieve.dynamic_reference_runtime import make_sheet

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "artifacts/reference_scenarios_v6"
    encoder = Encoder("siglip")
    ids = json.loads((ROOT / "artifacts/siglip_adapted/item_ids.json").read_text())
    vectors = np.load(ROOT / "artifacts/siglip_adapted/image_embeddings.npy")
    catalog = {
        r["item_id"]: r
        for r in map(
            json.loads,
            (ROOT / "data/snapshots/collection_v2.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    }
    for folder in out.iterdir():
        path = folder / "comparison.json"
        if not path.exists():
            continue
        r = json.loads(path.read_text(encoding="utf-8"))
        if not r.get("references"):
            continue
        r["control_texts"] = {}
        for name, text in [
            ("production_rewritten", parse_query(r["query"]).rewritten),
            ("production_zh_rewritten", parse_query(r["label"]).rewritten),
        ]:
            q = encoder.encode_texts([text])[0]
            ranking = [ids[i] for i in np.argsort(-(vectors @ q))[:10]]
            r["rankings"][name] = ranking
            r["control_texts"][name] = text
            make_sheet(
                folder / (name + "_top5.jpg"),
                text + " | " + name,
                [(v, ROOT / catalog[v]["image_path"]) for v in ranking[:5]],
                columns=5,
            )
        pool = sorted({v for ranks in r["rankings"].values() for v in ranks[:5]})
        r["review_pool"] = [
            {
                k: catalog[v].get(k)
                for k in [
                    "item_id",
                    "title",
                    "medium",
                    "period",
                    "description",
                    "source_url",
                    "image_path",
                ]
            }
            for v in pool
        ]
        make_sheet(
            folder / "all_controls_review.jpg",
            r["query"] + " | all methods pooled",
            [(v, ROOT / catalog[v]["image_path"]) for v in pool],
            columns=5,
        )
        path.write_text(json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
        print(r["key"], r["rankings"]["production_zh_rewritten"][:5], flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
