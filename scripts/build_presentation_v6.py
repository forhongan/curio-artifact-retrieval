"""Build an offline HTML presentation from measured results and reviewed cases."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "web/public/assets/presentation"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = json.loads((ROOT / "web/public/assets/research-v6.json").read_text(encoding="utf-8"))
    case = json.loads(
        (ROOT / "artifacts/reference_scenarios_v6/meiping/comparison.json").read_text(
            encoding="utf-8"
        )
    )
    data = {
        "stats": stats,
        "case": case,
        "review": json.loads(
            (ROOT / "artifacts/reference_scenarios_v6/root_review.json").read_text(encoding="utf-8")
        ),
    }
    (OUT / "data.js").write_text(
        "window.CURIO_DATA=" + json.dumps(data, ensure_ascii=False) + ";", encoding="utf-8"
    )
    print(OUT / "index.html")


if __name__ == "__main__":
    main()
