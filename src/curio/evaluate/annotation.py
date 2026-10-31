"""Blind relevance annotation with append-only judgments and source image whitelist."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import threading

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from curio.represent.train import read_items


def build_pool(ranking_paths, queries_path, output, top_k=15):
    queries = {q["qid"]: q for q in read_items(queries_path)}
    seen = set()
    tasks = []
    for path in ranking_paths:
        for ranking in read_items(path):
            qid = ranking["qid"]
            for item_id in ranking["ranking"][:top_k]:
                if (qid, item_id) in seen:
                    continue
                seen.add((qid, item_id))
                query = queries[qid]
                tasks.append(
                    {
                        "task_id": hashlib.sha256(f"{qid}:{item_id}".encode()).hexdigest()[:20],
                        "qid": qid,
                        "item_id": item_id,
                        "text": query["text_zh"],
                        "text_en": query["text_en"],
                    }
                )
    random.Random(5523).shuffle(tasks)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in tasks), encoding="utf-8"
    )
    return {"tasks": len(tasks), "queries": len({x["qid"] for x in tasks}), "path": str(output)}


class Judgment(BaseModel):
    task_id: str = Field(min_length=1, max_length=100)
    annotator: str = Field(min_length=1, max_length=100)
    grade: int = Field(ge=0, le=3)
    reason: str = Field(default="", max_length=2000)


def create_annotation_app(
    data="data/snapshots/collection_v2.jsonl",
    pool="eval/pool.jsonl",
    judgments="eval/human_judgments.jsonl",
    root=".",
):
    root = Path(root).resolve()
    items = {x["item_id"]: x for x in read_items(root / data)}
    tasks = read_items(root / pool)
    by_id = {x["task_id"]: x for x in tasks}
    target = root / judgments
    records = read_items(target) if target.exists() else []
    lock = threading.Lock()
    app = FastAPI(title="Curio blind human relevance annotation")

    def judged_by(annotator):
        return {r["task_id"] for r in records if r["annotator"] == annotator}

    @app.get("/")
    def home():
        return FileResponse(root / "web" / "public" / "annotate.html")

    @app.get("/api/task")
    def task(annotator: str = Query(default="", max_length=100), skip_task_id: str | None = None):
        done = judged_by(annotator)
        remaining = [x for x in tasks if x["task_id"] not in done]
        if skip_task_id:
            remaining = [x for x in remaining if x["task_id"] != skip_task_id] + [
                x for x in remaining if x["task_id"] == skip_task_id
            ]
        if not remaining:
            return {"task": None, "remaining": 0, "judged": len(done), "total": len(tasks)}
        current = remaining[0]
        return {
            **current,
            "image_url": f"/api/image/{current['item_id']}",
            "remaining": len(remaining),
            "judged": len(done),
            "total": len(tasks),
        }

    @app.post("/api/judgment", status_code=201)
    def judgment(record: Judgment):
        if record.task_id not in by_id:
            raise HTTPException(404, "Unknown task")
        if not record.annotator.strip():
            raise HTTPException(422, "Annotator must identify themself")
        with lock:
            history = [
                r
                for r in records
                if r["task_id"] == record.task_id and r["annotator"] == record.annotator
            ]
            saved = {
                **record.model_dump(),
                "qid": by_id[record.task_id]["qid"],
                "item_id": by_id[record.task_id]["item_id"],
                "created_at": datetime.now(timezone.utc).isoformat(),
                "judgment_id": len(records) + 1,
                "supersedes": history[-1]["judgment_id"] if history else None,
            }
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(saved, ensure_ascii=False) + "\n")
            records.append(saved)
        return {"saved": True, "judgment_id": saved["judgment_id"]}

    @app.get("/api/progress")
    def progress():
        annotators = sorted({r["annotator"] for r in records})
        return {
            "total_tasks": len(tasks),
            "total_judgments": len(records),
            "annotators": {a: len(judged_by(a)) for a in annotators},
            "label_source": "human",
            "blind": True,
        }

    @app.get("/api/image/{item_id}")
    def image(item_id: str):
        if item_id not in items:
            raise HTTPException(404, "Unknown image")
        path = (root / items[item_id]["image_path"]).resolve()
        if not path.is_relative_to(root / "data" / "processed") or not path.is_file():
            raise HTTPException(404, "Image unavailable")
        return FileResponse(path)

    return app


def export_human(path, output):
    import itertools
    import math

    from sklearn.metrics import cohen_kappa_score

    latest = {}
    for r in read_items(path):
        latest[(r["task_id"], r["annotator"])] = r
    by_task = {}
    for record in latest.values():
        by_task.setdefault(record["task_id"], []).append(record)
    qrels, conflicts = {}, []
    paired_grades = {}
    double_judged = 0
    for task_id, records in by_task.items():
        grades = [r["grade"] for r in records]
        if len(records) >= 2:
            double_judged += 1
        # Cohen's kappa belongs to a fixed pair of annotators. Pooling the
        # first two arrivals mixes raters and can make agreement order-dependent.
        ordered = sorted(records, key=lambda record: record["annotator"])
        for first, second in itertools.combinations(ordered, 2):
            pair = (first["annotator"], second["annotator"])
            paired_grades.setdefault(pair, []).append((first["grade"], second["grade"]))
        if len(set(grades)) > 1:
            conflicts.append({"task_id": task_id, "judgments": records})
            continue
        record = records[0]
        qrels.setdefault(record["qid"], {})[record["item_id"]] = record["grade"]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "qrels_human.jsonl").write_text(
        "".join(
            json.dumps(
                {"qid": qid, "grades": grades, "label_source": "human_unanimous_or_single"},
                ensure_ascii=False,
            )
            + "\n"
            for qid, grades in sorted(qrels.items())
        ),
        encoding="utf-8",
    )
    (output / "conflicts.json").write_text(
        json.dumps(conflicts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pairwise_kappa = []
    for (first, second), grades in sorted(paired_grades.items()):
        kappa, reason = None, None
        if len(grades) < 2:
            reason = "fewer_than_two_shared_tasks"
        elif len({grade for pair in grades for grade in pair}) == 1:
            reason = "zero_expected_disagreement"
        else:
            value = float(
                cohen_kappa_score(
                    [pair[0] for pair in grades],
                    [pair[1] for pair in grades],
                    labels=[0, 1, 2, 3],
                    weights="quadratic",
                )
            )
            if math.isfinite(value):
                kappa = value
            else:
                reason = "undefined_kappa"
        pairwise_kappa.append(
            {
                "annotator_a": first,
                "annotator_b": second,
                "shared_tasks": len(grades),
                "weighted_kappa": kappa,
                "reason": reason,
            }
        )
    # Keep the legacy scalar only when exactly one fixed pair was evaluated.
    agreement = {
        "unique_pairs": len(by_task),
        "double_judged_pairs": double_judged,
        "unresolved_conflicts": len(conflicts),
        "pairwise_kappa": pairwise_kappa,
        "weighted_kappa": pairwise_kappa[0]["weighted_kappa"] if len(pairwise_kappa) == 1 else None,
        "kappa_reason": "multiple_annotator_pairs_reported_separately"
        if len(pairwise_kappa) > 1
        else (pairwise_kappa[0]["reason"] if pairwise_kappa else "no_double_judged_tasks"),
        "warning": "Disagreements excluded until explicit adjudication; no automatic consensus invented.",
    }
    (output / "agreement.json").write_text(json.dumps(agreement, indent=2), encoding="utf-8")
    return agreement


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    pool_parser = sub.add_parser("pool")
    pool_parser.add_argument("rankings", nargs="+")
    pool_parser.add_argument("--queries", default="eval/queries_dev.jsonl")
    pool_parser.add_argument("--output", default="eval/pool.jsonl")
    serve = sub.add_parser("serve")
    serve.add_argument("--port", type=int, default=8001)
    serve.add_argument("--pool", default="eval/pool.jsonl")
    serve.add_argument("--data", default="data/snapshots/collection_v2.jsonl")
    serve.add_argument("--judgments", default="eval/human_judgments.jsonl")
    export = sub.add_parser("export")
    export.add_argument("--judgments", default="eval/human_judgments.jsonl")
    export.add_argument("--output", default="eval/human")
    args = p.parse_args()
    if args.command == "pool":
        print(build_pool(args.rankings, args.queries, args.output))
    elif args.command == "export":
        print(export_human(args.judgments, args.output))
    else:
        import uvicorn

        uvicorn.run(
            create_annotation_app(data=args.data, pool=args.pool, judgments=args.judgments),
            host="127.0.0.1",
            port=args.port,
        )
