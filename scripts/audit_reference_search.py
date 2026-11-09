"""Verify API image availability and source relevance before image retrieval.

Fetch keeps every hit and its original position. Ranking requires an explicit
review file grounded in the downloaded source descriptions and actual pictures.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
from urllib.parse import urlencode

import numpy as np
from PIL import Image
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from curio.retrieve.dynamic_reference_runtime import (  # noqa: E402
    DynamicReferenceRuntime, make_sheet, score_gallery, write_json,
)
from curio.retrieve.dynamic_references import DynamicMuseumAcquirer  # noqa: E402
from curio.retrieve.references import image_fingerprints  # noqa: E402

ATTRIBUTION = '© Victoria and Albert Museum, London'
TERMS = 'https://www.vam.ac.uk/info/va-websites-terms-conditions'


def fetch(query: str, folder: Path):
    runtime = DynamicReferenceRuntime(ROOT)
    runtime._load()
    acquirer = DynamicMuseumAcquirer(ROOT, folder, runtime._gallery[1])
    url = 'https://api.vam.ac.uk/v2/objects/search?' + urlencode(
        {'q': query, 'images_exist': 1, 'page_size': 100})
    response = acquirer.get_json(url, 'search', 'vam-search')
    rows = []
    for position, hit in enumerate(response.get('records', []), 1):
        sid = hit['systemNumber']
        row = {'reference_id': f'vam_{sid}', 'source': 'vam', 'source_id': sid,
               'source_api_rank': position, 'source_url': f'https://collections.vam.ac.uk/item/{sid}',
               'api_primary_image_id': hit.get('_primaryImageId'),
               'api_reported_has_image': bool(hit.get('_primaryImageId')),
               'download_verified': False, 'semantic_relevance': 'pending_actual_review',
               'license': 'V&A educational/noncommercial terms; not CC0',
               'license_url': TERMS, 'attribution': ATTRIBUTION}
        rows.append(row)
        try:
            detail = acquirer.get_json(f'https://api.vam.ac.uk/v2/museumobject/{sid}',
                                       'object', f'vam-object-{sid}')
            write_json(folder / 'metadata' / f'{sid}.json', detail)
            record = detail['record']
            row.update(title=record.get('objectType'), date=record.get('productionDates'),
                       brief_description=record.get('briefDescription'),
                       physical_description=record.get('physicalDescription'),
                       summary_description=record.get('summaryDescription'),
                       materials=record.get('materials'), styles=record.get('styles'),
                       medium=record.get('materialsAndTechniques'),
                       accession=record.get('accessionNumber'),
                       api_detail_image_count=len(record.get('images', [])),
                       metadata_sha256=hashlib.sha256(json.dumps(detail, sort_keys=True).encode()).hexdigest())
            image_id = hit.get('_primaryImageId')
            if not image_id:
                row['failure'] = 'API returned no primary image identifier'
                continue
            manifest_url = f'https://iiif.vam.ac.uk/collections/{sid}/manifest.json'
            manifest = acquirer.get_json(manifest_url, 'object', f'vam-manifest-{sid}')
            row.update(manifest_attribution=manifest.get('attribution'),
                       manifest_license=manifest.get('license'))
            # V&A educational digital-use limit: longest side <=768 pixels.
            row['image_url'] = f'https://framemark.vam.ac.uk/collections/{image_id}/full/!768,768/0/default.jpg'
            binary, request = acquirer.get_image(row, f'vam-image-{sid}')
            with Image.open(io.BytesIO(binary)) as image:
                image.load()
                row.update(width=image.width, height=image.height,
                           **image_fingerprints(image))
            path = folder / 'images' / f'{sid}.jpg'
            path.write_bytes(binary)
            row.update(image_path=path.relative_to(ROOT).as_posix(),
                       image_sha256=hashlib.sha256(binary).hexdigest(),
                       image_request=request, download_verified=True,
                       gallery_overlap=runtime._gallery[1].reason(row))
        except (OSError, ValueError, KeyError, RuntimeError) as exc:
            row['failure'] = str(exc)
        print(json.dumps({k: row.get(k) for k in ('reference_id', 'title', 'date',
                        'brief_description', 'download_verified', 'failure')}, ensure_ascii=False), flush=True)
    acquisition = {'query': query, 'source': 'vam', 'request_url': url,
                   'api_info': response.get('info'),
                   'raw_returned_ids_in_order': [h['systemNumber'] for h in response.get('records', [])],
                   'image_availability_contract': {
                       'search': 'images_exist=1; _primaryImageId and _images per record; info.image_count',
                       'detail': 'images list',
                       'verified': 'HTTP image download followed by Pillow decode',
                       'semantic_relevance': 'not implied by image availability; separately reviewed'},
                   'candidates': rows}
    write_json(folder / 'acquisition.json', acquisition)
    acquirer.save_manifest(folder / 'http_requests.jsonl')
    cells = [(f'{row["source_api_rank"]} {row["reference_id"]}', ROOT / row['image_path'])
             for row in rows if row['download_verified']]
    for start in range(0, len(cells), 12):
        make_sheet(folder / f'api_images_{start // 12 + 1}.jpg',
                   f'API hits: {query} / V&A; {ATTRIBUTION}', cells[start:start + 12], columns=4)


def rank(folder: Path, review_file: Path):
    acquired = json.loads((folder / 'acquisition.json').read_text(encoding='utf-8'))
    review = json.loads(review_file.read_text(encoding='utf-8'))
    if review['query'] != acquired['query']:
        raise ValueError('Review query does not match acquired query')
    decisions = {r['reference_id']: r for r in review['decisions']}
    accepted = []
    for row in acquired['candidates']:
        decision = decisions.get(row['reference_id'], {})
        if decision.get('accepted') is not True:
            continue
        if not row['download_verified'] or row.get('gallery_overlap'):
            raise ValueError('Review accepted an unavailable/overlapping image')
        if not decision.get('source_checked') or not decision.get('picture_checked') or not decision.get('reason'):
            raise ValueError('Accepted references need source and picture inspection with reasons')
        path = ROOT / row['image_path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['image_sha256']:
            raise ValueError('Reference bytes changed since acquisition')
        accepted.append(row)
    if not accepted:
        raise ValueError('No confirmed relevant references; do not run a meaningless image retrieval')
    runtime = DynamicReferenceRuntime(ROOT)
    runtime._load()
    sr, dr, encoders = runtime._encode(accepted, folder)
    sg, dg = runtime._indices
    scored = score_gallery(sg[1], sg[0], dg[0], sr, dr)
    by_id = {r['item_id']: r for r in runtime._gallery[0]}
    rankings = {key: [sg[1][i] for i in order] for key, order in scored['orders'].items()}
    for method in ('siglip_mean', 'dino_mean', 'fusion'):
        cells = [(f'{i + 1} {item_id}', ROOT / by_id[item_id]['image_path'])
                 for i, item_id in enumerate(rankings[method][:12])]
        make_sheet(folder / f'results_{method}.jpg',
                   f'Checked API reference photographs -> {method}', cells, columns=4)
    np.savez_compressed(folder / 'gallery_scores.npz', item_ids=np.array(sg[1]),
                        **scored['channels'], siglip_pairs=scored['siglip_pairs'],
                        dino_pairs=scored['dino_pairs'])
    write_json(folder / 'result.json', {
        'query': acquired['query'], 'reference_ids': [r['reference_id'] for r in accepted],
        'reference_count': len(accepted), 'gallery_count': len(sg[1]),
        'reference_review_file': str(review_file.relative_to(ROOT)),
        'reference_review_sha256': hashlib.sha256(review_file.read_bytes()).hexdigest(),
        'ranking_inputs': ['confirmed external reference image pixels', 'gallery image pixels'],
        'candidate_catalogue_used_for_ranking': False,
        'encoders': encoders, 'consensus': scored['consensus'], 'rankings': rankings})
    print(f'Ranked {len(sg[1])} images from {len(accepted)} inspected API references', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('fetch', 'rank'))
    parser.add_argument('--query', default='Qianlong famille rose vase')
    parser.add_argument('--output', type=Path,
                        default=Path('artifacts/checked_api_references_20261108/qianlong_vase'))
    parser.add_argument('--review', type=Path)
    args = parser.parse_args()
    folder = ROOT / args.output
    folder.mkdir(parents=True, exist_ok=True)
    if args.phase == 'fetch':
        fetch(args.query, folder)
    elif args.review:
        rank(folder, ROOT / args.review)
    else:
        parser.error('rank requires --review from completed image/source inspection')


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
