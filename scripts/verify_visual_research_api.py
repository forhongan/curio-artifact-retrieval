"""Exercise real full-gallery set and direct-reference APIs on loopback HTTP."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

import httpx
from threadpoolctl import threadpool_limits
import uvicorn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')

from curio.retrieve.engine import SearchService, load_items  # noqa: E402
from curio.serve.app import create_app  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-cache', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8011)
    parser.add_argument('--output', type=Path, default=Path('artifacts/research_api_acceptance_20261108.json'))
    args = parser.parse_args()
    shutil.copytree(args.reference_cache / 'http', ROOT / 'artifacts/dynamic_reference_api/cache/http',
                    dirs_exist_ok=True)
    service = SearchService(load_items(ROOT / 'data/snapshots/collection_v2.jsonl'),
                            index_dir=ROOT / 'artifacts/siglip_adapted',
                            encoder_name='siglip', root=ROOT, probe_weight=0)
    app = create_app(service=service, root=ROOT)
    server = uvicorn.Server(uvicorn.Config(app, host='127.0.0.1', port=args.port, log_level='warning'))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(100):
            if server.started:
                break
            if not thread.is_alive():
                raise RuntimeError('Loopback server could not start')
            time.sleep(.1)
        with httpx.Client(base_url=f'http://127.0.0.1:{args.port}', timeout=180) as client:
            response = client.post('/api/search/set', json={'query': 'porcelain', 'top_k': 3})
            response.raise_for_status()
            result = response.json()
            assert result['gallery_count'] == 4692
            assert result['matched_count'] > 3
            assert len(result['results']) == len(result['matched_item_ids']) == result['matched_count']
            assert len(result['preview']) == 3
            empty = client.post('/api/search/set', json={'query': 'a red sports car'}).json()
            assert empty['matched_count'] == 0 and empty['no_match']
            response = client.post('/api/search/reference/dynamic', json={
                'query': 'Qianlong famille rose porcelain', 'offline': True,
                'reference_limit': 8, 'top_k': 6})
            if response.is_error:
                raise RuntimeError(f'Direct-reference HTTP {response.status_code}: {response.text}')
            response.raise_for_status()
            reference = response.json()
            assert reference['mode'] == 'direct_api_text'
            assert reference['search_text'] == 'Qianlong famille rose porcelain'
            assert reference['gallery_count'] == 4692 and reference['reference_count'] == 7
            assert reference['candidate_catalogue_used_for_ranking'] is False
            assert reference['acquisition_requests']['live'] == 0
            assert reference['acquisition_requests']['cached'] > 0
            assert len(reference['results']) == 6 and len(reference['references']) == 7
            image = client.get(reference['references'][0]['image_url'])
            image.raise_for_status()
            assert image.headers['content-type'].startswith('image/')
            assert client.post('/api/search/reference/dynamic', json={'query': '  '}).status_code == 422
            report = {
                'transport': 'actual loopback HTTP; temporary server stopped after verification',
                'set': {key: result[key] for key in ('gallery_count', 'matched_count', 'method', 'latency_ms')},
                'set_full_collection_returned': True, 'preview_count': 3,
                'negative_control_matched_count': empty['matched_count'],
                'direct_reference': {key: reference[key] for key in (
                    'audit_id', 'mode', 'search_text', 'reference_count', 'reference_ids',
                    'gallery_count', 'acquisition_requests', 'candidate_catalogue_used_for_ranking')},
                'reference_image_sha256': hashlib.sha256(image.content).hexdigest(),
                'status': 'passed',
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps(report, ensure_ascii=False, indent=2))
    finally:
        server.should_exit = True
        thread.join(timeout=5)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
