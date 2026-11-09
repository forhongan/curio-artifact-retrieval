"""Build fixed visual predicates for full-gallery qualitative set inspection.

No test scoring or threshold search is performed. Catalogue facets train the
heads; the .55 engineering floor is explicit and not called calibrated accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from curio.evaluate.catalogue_sets import fit_catalogue_heads  # noqa: E402
from curio.represent.validation import validate_object_splits  # noqa: E402


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/catalogue_set_runtime_v5')
    parser.add_argument('--labels', default='artifacts/cv_v4_20261108/catalogue_v4_semantic_review/labels_catalogue_v2.jsonl')
    args = parser.parse_args()
    output, labels = ROOT / args.output, ROOT / args.labels
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing overwrite: {output}')
    output.mkdir(parents=True, exist_ok=True)
    index = ROOT / 'artifacts/siglip'
    ids = json.loads((index / 'item_ids.json').read_text(encoding='utf-8'))
    lookup = {row['item_id']: row for row in map(json.loads, labels.read_text(encoding='utf-8').splitlines())}
    rows = [lookup[item] for item in ids]
    validate_object_splits(rows)
    vectors = np.load(index / 'image_embeddings.npy', allow_pickle=False)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    heads, tasks = fit_catalogue_heads(rows, vectors, fixed_c=1., fixed_threshold=.55)
    for domain, head in heads.items():
        path = output / f'{domain}.npz'
        np.savez(path, weights=head['weights'], bias=head['bias'], classes=np.asarray(head['classes']))
        tasks[domain].update(artifact=path.name, artifact_sha256=sha(path))
    head_report = {
        'version': 'catalogue-multiclass-heads-v5', 'tasks': tasks,
        'labels_sha256': sha(labels), 'index_sha256': sha(index / 'image_embeddings.npy'),
        'item_ids_sha256': sha(index / 'item_ids.json'), 'dimension': int(vectors.shape[1]),
        'training': 'train-only known, mutually-exclusive official catalogue facets; fixed C=1',
    }
    dump(output / 'report.json', head_report)
    dump(output / 'set_policy.json', {
        'version': 'catalogue-sets-v5', 'head_report_sha256': sha(output / 'report.json'),
        'labels_sha256': sha(labels), 'deployment_method': 'calibrated_min',
        'deployment_threshold': 1., 'confidence_floor': .55,
        'policy': 'All requested predicates must exceed fixed .55. This floor is an engineering choice, not calibrated correctness. No test scoring or threshold tuning.',
    })
    print(json.dumps({'items': len(rows), 'heads': list(heads), 'output': str(output)}, indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
