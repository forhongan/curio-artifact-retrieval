"""Portable replay of verified whole-gallery decisions; never runs new inference."""

import argparse
import copy
import gzip
import hashlib
import json
from pathlib import Path
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from curio.retrieve.catalogue_query import compile_catalogue


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=1000)
    method: str = "catalogue_visual"
    use_references: bool = False


def create_snapshot_app(directory):
    root = Path(directory).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    path = root / "snapshot.json.gz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["snapshot_sha256"]:
        raise ValueError("Snapshot checksum mismatch")
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        bundle = json.load(stream)
    if bundle["schema"] != "curio-boundary-snapshot-v1":
        raise ValueError("Unsupported snapshot schema")
    items = {r["item_id"]: r for r in bundle["items"]}
    plans, jobs, aliases = {}, {}, []
    for record in bundle["queries"]:
        report = record["report"]
        decisions = report["decisions"]
        if (
            not report["complete"]
            or len(decisions) != len(items)
            or {r["item_id"] for r in decisions} != set(items)
            or any(type(r["yes"]) is not bool for r in decisions)
        ):
            raise ValueError("Incomplete binary snapshot")
        actual = {r["item_id"] for r in decisions if r["yes"]}
        if (
            actual != {r["item_id"] for r in report["results"]}
            or len(actual) != report["matched_count"]
        ):
            raise ValueError("Snapshot result/decision mismatch")
        for alias in record["aliases"]:
            plan = compile_catalogue(alias)
            if plan is None or plan.key != report["plan"]["key"]:
                raise ValueError("Snapshot parser version differs")
            plans[plan.key] = report["job_id"]
        jobs[report["job_id"]] = report
        aliases.append(
            {"aliases": record["aliases"], "yes": len(actual), "no": len(items) - len(actual)}
        )
    app = FastAPI(title="Curio boundary v15 portable snapshot API", version="15.0")
    origins = os.environ.get(
        "CURIO_ALLOWED_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173,http://localhost:3000,http://127.0.0.1:3000",
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[s.strip() for s in origins if s.strip()],
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    def reply(ident, include_decisions=False, query=None):
        if ident not in jobs:
            raise HTTPException(404, "Unknown snapshot job")
        value = copy.deepcopy(jobs[ident])
        if not include_decisions:
            value.pop("decisions", None)
        if query is not None:
            value["query"] = query
        value.update(
            execution_mode="snapshot_replay",
            live_inference=False,
            snapshot_version=bundle["version"],
        )
        value["notices"] = [
            "当前读取固定图库的已验证结果，不执行新图像推理；只支持列出的查询意图。",
            *value.get("notices", []),
        ]
        return value

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "items": len(items),
            "execution_mode": "snapshot_replay",
            "live_inference": False,
        }

    @app.get("/api/boundary/capabilities")
    def capabilities():
        return {
            "method": "catalogue_visual",
            "execution_mode": "snapshot_replay",
            "live_inference": False,
            "gallery_count": len(items),
            "supported_queries": aliases,
            "images": "384px display previews; decisions were computed from original photos",
        }

    @app.post("/api/search/set/jobs", status_code=202)
    def create(request: Query):
        if request.method != "catalogue_visual" or request.use_references:
            raise HTTPException(
                422, "Snapshot supports catalogue_visual without reference adaptation"
            )
        try:
            plan = compile_catalogue(request.query)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        if plan is None or plan.key not in plans:
            raise HTTPException(
                422,
                {
                    "message": "查询不在此结果包中；请连接实时推理后端。",
                    "supported_queries": aliases,
                },
            )
        return reply(plans[plan.key], query=request.query)

    @app.get("/api/search/set/jobs/{job_id}")
    def get(job_id: str, decisions: bool = False):
        return reply(job_id, decisions)

    @app.post("/api/search/set/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        # A completed precomputed job requires no cancellation.
        return reply(job_id)

    @app.get("/api/items/{item_id}")
    def detail(item_id: str):
        if item_id not in items:
            raise HTTPException(404, "Unknown item")
        return items[item_id]

    @app.get("/api/images/{item_id}")
    def image(item_id: str, thumb: bool = False):
        if item_id not in items:
            raise HTTPException(404, "Unknown item")
        path = root / "images" / f"{item_id}.jpg"
        if not path.is_file():
            raise HTTPException(404, "Preview missing; restore the complete Windows bundle")
        return FileResponse(path, media_type="image/jpeg")

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/boundary_v15")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8015)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_snapshot_app(args.data), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
