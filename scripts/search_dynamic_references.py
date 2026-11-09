"""Run specialist-description retrieval without benchmark labels or answer banks."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from curio.retrieve.dynamic_reference_runtime import DynamicReferenceRuntime  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("description", help="User specialist description (Chinese or English)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--mode", choices=("direct", "qualified", "curated"), default="direct")
    parser.add_argument("--aggregation", choices=("single_best", "uniform_mean", "max", "diverse", "query_aligned"), default="uniform_mean")
    parser.add_argument("--search-text", help="Explicit translation/override; original description is preserved")
    parser.add_argument("--acquisition", type=Path, help="Replay a same-description acquisition JSON")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--reference-limit", type=int, default=8)
    parser.add_argument("--per-query", type=int, default=20)
    parser.add_argument("--page-depth", type=int, default=2)
    parser.add_argument("--max-metadata-candidates", type=int, default=150)
    parser.add_argument("--top-k", type=int, default=24)
    args = parser.parse_args()
    runtime = DynamicReferenceRuntime(ROOT, device=args.device)
    if args.mode == 'qualified':
        result=runtime.run(args.description,args.output,mode=args.mode,search_text=args.search_text,
                           cache_dir=args.cache_dir,offline=args.offline,reference_limit=args.reference_limit,
                           top_k=args.top_k,aggregation=args.aggregation)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return
    result = runtime.run(args.description, args.output, mode=args.mode, search_text=args.search_text, cache_dir=args.cache_dir,
        offline=args.offline, acquisition=args.acquisition, reference_limit=args.reference_limit,
        per_query=args.per_query, page_depth=args.page_depth,
        max_metadata_candidates=args.max_metadata_candidates, top_k=args.top_k)
    print(json.dumps({key: result.get(key) for key in
        ("status", "description", "evidence_level", "reference_count", "reference_ids", "gallery_count")},
        ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
