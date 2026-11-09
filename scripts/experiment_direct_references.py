"""Acquire varied specialist queries directly from APIs; compare image retrieval.

No target catalogue text is read by any scorer. Museum records are retained for
later human-readable provenance and review, never for rank construction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from curio.retrieve.dynamic_reference_runtime import (  # noqa: E402
    DynamicReferenceRuntime, make_sheet, write_json,
)

QUERIES = (
    ('kangxi_meiping', '康熙梅瓶', 'Kangxi meiping'),
    ('wanli_bluewhite', '明代万历青花瓷', 'Wanli blue and white porcelain'),
    ('ming_bluewhite', '明代青花瓷', 'Ming dynasty blue and white porcelain'),
    ('song_celadon', '宋代青瓷碗', 'Song dynasty celadon bowl'),
    ('qianlong_vase', '清乾隆粉彩瓷瓶', 'Qianlong famille rose vase'),
    ('liao_flask', '辽代皮囊壶', 'Liao dynasty flask'),
    ('qianlong_famillerose', '乾隆粉彩瓷', 'Qianlong famille rose porcelain'),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('artifacts/direct_reference_experiments_20261108'))
    parser.add_argument('--only', nargs='+', choices=[q[0] for q in QUERIES])
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--skip-existing', action='store_true')
    parser.add_argument('--acquire-only', action='store_true')
    args = parser.parse_args()
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    runtime = DynamicReferenceRuntime(ROOT)
    summaries = []
    queries = [q for q in QUERIES if not args.only or q[0] in args.only]
    for qid, original, search_text in queries:
        folder = out / qid
        start = time.perf_counter()
        if args.skip_existing and (folder / 'result.json').exists():
            result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        else:
            result = runtime.run(original, folder, mode='direct', search_text=search_text,
                                 offline=args.offline, reference_limit=8, top_k=12,
                                 per_query=30, page_depth=2, max_metadata_candidates=120)
        summaries.append({
            'qid': qid, 'original_description': original, 'search_text': search_text,
            'status': result['status'], 'reference_count': result['reference_count'],
            'reference_ids': result['reference_ids'],
            'request_summary': result['request_summary'],
            'gallery_count': result['gallery_count'],
            'elapsed_this_invocation_s': round(time.perf_counter() - start, 3),
            'result_file': (folder / 'result.json').relative_to(ROOT).as_posix(),
        })
        write_json(out / 'acquisition_summary.json', summaries)
        print(json.dumps(summaries[-1], ensure_ascii=False), flush=True)
    if args.acquire_only:
        return

    # Same base image space as the reference method; production LoRA is an
    # additional operational baseline, not a change to the reference index.
    from curio.represent.encoders import Encoder
    from curio.retrieve.engine import SearchService, load_items

    runtime._load()
    siglip = runtime._indices[0]
    items = runtime._gallery[0]
    by_id = {row['item_id']: row for row in items}
    encoder = runtime._siglip or Encoder('siglip')
    english = [q[2] for q in queries]
    text = encoder.encode_texts(english)
    production = SearchService(load_items(ROOT / 'data/snapshots/collection_v2.jsonl'),
                               index_dir=ROOT / 'artifacts/siglip_adapted',
                               encoder_name='siglip', root=ROOT, probe_weight=0)
    comparisons = []
    for (qid, original, search_text), query_vector in zip(queries, text):
        folder = out / qid
        base_scores = siglip[0] @ query_vector
        base_order = np.array(sorted(range(len(base_scores)),
                                    key=lambda i: (-float(base_scores[i]), siglip[1][i])))
        prod_scores = production._dense(search_text)
        prod_order = np.array(sorted(range(len(prod_scores)),
                                    key=lambda i: (-float(prod_scores[i]), production.items[i]['item_id'])))
        rankings = {
            'text_base': [siglip[1][i] for i in base_order],
            'text_production': [production.items[i]['item_id'] for i in prod_order],
        }
        np.savez_compressed(folder / 'text_baselines.npz',
                            item_ids=np.array(siglip[1]), text_base=base_scores,
                            query_vector=query_vector)
        for method, ids in rankings.items():
            make_sheet(folder / f'results_{method}.jpg', f'{search_text} / {method}',
                       [(f'{j + 1} {item_id}', ROOT / by_id[item_id]['image_path'])
                        for j, item_id in enumerate(ids[:12])])
        write_json(folder / 'text_baselines.json', {
            'query': original, 'encoded_query': search_text, 'rankings': rankings,
            'base_index_sha256': siglip[2]['index_sha256'],
            'base_model': encoder.metadata, 'candidate_catalogue_used_for_ranking': False,
            'selection_note': 'English translation is explicit and shared across methods.'})
        result = json.loads((folder / 'result.json').read_text(encoding='utf-8'))
        fusion = result.get('rankings', {}).get('fusion', [])
        comparisons.append({'qid': qid, 'reference_count': result['reference_count'],
                            'text_base_vs_reference_top12_overlap': len(set(rankings['text_base'][:12]) & set(fusion[:12])),
                            'interpretation': 'Ranking overlap is not retrieval accuracy.'})
    write_json(out / 'comparison_summary.json', comparisons)
    write_json(out / 'source_manifest.json', {
        'source_sha256': {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
                          for path in ('scripts/experiment_direct_references.py',
                                       'src/curio/retrieve/dynamic_reference_runtime.py',
                                       'src/curio/retrieve/dynamic_references.py')},
        'gallery_sha256': hashlib.sha256((ROOT / 'data/snapshots/collection_v2.jsonl').read_bytes()).hexdigest(),
        'candidate_catalogue_used_for_ranking': False})


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
