"""Extract fixed classical features from a JSONL pixel manifest."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import skimage

from curio.represent.classical_cv import PARAMETERS, VERSION, extract_classical


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process(task):
    root, row = task
    image = Path(root) / row['image_path']
    digest = sha(image)
    if digest != row['image_sha256']:
        raise ValueError(f"Image hash mismatch: {row['item_id']}")
    cv2.setNumThreads(1)
    result = extract_classical(image)
    return result['features'], {'item_id': row['item_id'], 'image_sha256': digest,
                                **result['diagnostics']}


def build(manifest, output, root='.', workers=1):
    start = time.perf_counter()
    root = Path(root).resolve()
    source = Path(manifest)
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    ids = [str(row['item_id']) for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise ValueError('Manifest must contain unique image IDs')
    # Reject catalog descriptions; only these input fields may influence extraction.
    allowed = {'item_id', 'image_path', 'image_sha256'}
    if any(set(row) != allowed for row in rows):
        raise ValueError('Manifest rows require only item_id, image_path, image_sha256')
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(process, ((root, row) for row in rows), chunksize=8))
    keys = list(results[0][0])
    arrays = {key: np.stack([result[0][key] for result in results]) for key in keys}
    arrays['gc_valid'] = np.array([result[1]['grabcut_accepted'] for result in results], dtype=bool)
    np.savez_compressed(output / 'features.npz', **arrays)
    (output / 'item_ids.json').write_text(json.dumps(ids, ensure_ascii=False), encoding='utf-8')
    (output / 'diagnostics.jsonl').write_text(
        ''.join(json.dumps(result[1], ensure_ascii=False) + '\n' for result in results), encoding='utf-8')
    files = {name: sha(output / name) for name in ('features.npz', 'item_ids.json', 'diagnostics.jsonl')}
    metadata = {'version': VERSION, 'source_manifest_sha256': sha(source), 'rows': len(ids),
                'parameters': PARAMETERS, 'dimensions': {k: int(v.shape[1]) for k, v in arrays.items() if k != 'gc_valid'},
                'grabcut_accepted': int(arrays['gc_valid'].sum()),
                'raw_fallback_count': int((~arrays['gc_valid']).sum()), 'files_sha256': files,
                'implementation_sha256': sha(Path(__file__).resolve().parents[1] / 'src/curio/represent/classical_cv.py'),
                'builder_sha256': sha(__file__), 'opencv_version': cv2.__version__,
                'skimage_version': skimage.__version__, 'numpy_version': np.__version__,
                'seconds': round(time.perf_counter()-start, 3),
                'created_at': datetime.now(timezone.utc).isoformat()}
    (output / 'manifest.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--root', default='.')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(build(args.manifest, args.output, args.root, args.workers), indent=2))


if __name__ == '__main__':
    main()
