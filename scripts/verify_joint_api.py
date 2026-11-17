"""Wait for a real whole-gallery job, then verify binary and bilingual contracts."""

import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--query", default="青花瓷")
    parser.add_argument("--paired-query", default="blue and white porcelain")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/joint_vlm_v9/full_gallery_blue_white.json")
    )
    args = parser.parse_args()

    def call(path, data=None):
        req = urllib.request.Request(
            args.url + path,
            data=json.dumps(data).encode() if data is not None else None,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.load(r)

    r = call("/api/search/set/jobs", {"query": args.query})
    ident = r["job_id"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix(".active.json").write_text(json.dumps(r, ensure_ascii=False, indent=2))
    print("job", ident, flush=True)
    while True:
        r = call("/api/search/set/jobs/" + ident)
        print(r["status"], r["processed"], r["total"], r["matched_count"], flush=True)
        if r["status"] in ["complete", "failed", "cancelled"]:
            break
        time.sleep(30)
    if r["status"] != "complete":
        raise RuntimeError(r)
    r = call("/api/search/set/jobs/" + ident + "?decisions=true")
    assert r["processed"] == r["total"] == 4692 and len(r["decisions"]) == 4692
    assert all(type(v["yes"]) is bool for v in r["decisions"])
    assert not any(v["yes"] and v["reason"] == "color_not_observable" for v in r["decisions"])
    paired = call("/api/search/set/jobs", {"query": args.paired_query})
    assert paired["job_id"] == ident and paired["complete"]
    args.output.write_text(json.dumps(r, ensure_ascii=False, indent=2) + "\n")
    print("FULL GALLERY AND BILINGUAL CACHE VERIFIED", flush=True)


if __name__ == "__main__":
    main()
