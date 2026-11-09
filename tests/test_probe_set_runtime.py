import hashlib
import json

from fastapi.testclient import TestClient
import numpy as np

from curio.retrieve.engine import SearchService
from curio.retrieve.probe_sets import ProbeSetService
from curio.serve.app import create_app


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime(tmp_path):
    heads, index = tmp_path / 'heads', tmp_path / 'base'
    heads.mkdir()
    index.mkdir()
    ids = ['vase_porcelain', 'vase_bronze', 'bowl_porcelain']
    vectors = np.asarray([[1, 1], [1, -1], [-1, 1]], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    np.save(index / 'image_embeddings.npy', vectors)
    (index / 'item_ids.json').write_text(json.dumps(ids))
    tasks = {}
    for domain, classes, weights in (
        ('form', ['bowl', 'vase'], [[4., 0.]]),
        ('fine_material', ['bronze', 'porcelain'], [[0., 4.]]),
    ):
        path = heads / f'{domain}.npz'
        np.savez(path, weights=np.asarray(weights), bias=np.zeros(1))
        tasks[domain] = {
            'status': 'trained', 'classes': classes, 'artifact': path.name,
            'artifact_sha256': digest(path), 'thresholds': dict.fromkeys(classes, .55),
        }
    (heads / 'report.json').write_text(json.dumps({
        'version': 'catalogue-multiclass-heads-v5', 'tasks': tasks,
        'index_sha256': digest(index / 'image_embeddings.npy'),
        'item_ids_sha256': digest(index / 'item_ids.json'),
    }))
    policy = heads / 'set_policy.json'
    policy.write_text(json.dumps({
        'version': 'catalogue-sets-v5', 'head_report_sha256': digest(heads / 'report.json'),
    }))
    items = [{'item_id': item, 'title': 'unrelated display text'} for item in reversed(ids)]
    return heads, index, policy, items, vectors[::-1]


def test_multi_image_set_is_variable_size_and_conjunction_is_over_whole_gallery(tmp_path):
    heads, _, policy, items, vectors = runtime(tmp_path)
    service = ProbeSetService(items, vectors, probes_path=heads, set_report_path=policy)
    broad = service.search('vase', top_k=1)
    assert set(broad['matched_item_ids']) == {'vase_porcelain', 'vase_bronze'}
    assert broad['matched_count'] == 2 and broad['returned'] == 1
    specific = service.search('porcelain vase')
    assert specific['matched_item_ids'] == ['vase_porcelain']
    assert all(atom['yes'] for atom in specific['results'][0]['atoms'])
    empty = service.search('bronze bowl')
    assert empty['matched_item_ids'] == [] and empty['no_match'] == 1
    assert broad['gate'] is None


def test_set_api_uses_independent_base_index_and_aligns_scrambled_catalogue(tmp_path, monkeypatch):
    heads, index, policy, items, _ = runtime(tmp_path)
    monkeypatch.setenv('CURIO_SET_PROBES', str(heads))
    monkeypatch.setenv('CURIO_SET_REPORT', str(policy))
    monkeypatch.setenv('CURIO_SET_INDEX', str(index))
    # Deliberately unrelated active LoRA vectors must have no effect.
    active = SearchService(items, embeddings=np.ones((3, 5)), root=tmp_path)
    client = TestClient(create_app(service=active, root=tmp_path))
    result = client.post('/api/search/set', json={'query': 'porcelain vase'})
    assert result.status_code == 200
    assert result.json()['matched_item_ids'] == ['vase_porcelain']
    assert client.post('/api/search/set', json={'query': 'two vases'}).status_code == 422
    assert client.post('/api/search/set', json={'query': 'Qianlong porcelain vase'}).status_code == 422


def test_set_api_rejects_changed_image_index(tmp_path, monkeypatch):
    heads, index, policy, items, _ = runtime(tmp_path)
    monkeypatch.setenv('CURIO_SET_PROBES', str(heads))
    monkeypatch.setenv('CURIO_SET_REPORT', str(policy))
    monkeypatch.setenv('CURIO_SET_INDEX', str(index))
    np.save(index / 'image_embeddings.npy', np.ones((3, 2)))
    client = TestClient(create_app(service=SearchService(items, root=tmp_path), root=tmp_path))
    response = client.post('/api/search/set', json={'query': 'vase'})
    assert response.status_code == 503
    assert 'does not match' in response.json()['detail']
