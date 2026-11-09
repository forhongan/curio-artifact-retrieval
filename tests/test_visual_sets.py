import numpy as np
import pytest

from curio.retrieve.visual_sets import VisualSetService, visual_boundary


def test_boundary_returns_variable_sets_and_rejects_weak_out_of_domain_peak():
    rng = np.random.default_rng(5523)
    scores = np.r_[rng.normal(.01, .002, 70), rng.normal(.15, .005, 30)]
    generic = np.full(100, .04)
    selected, _, _, report = visual_boundary(scores, generic)
    assert 20 <= selected.sum() <= 30
    assert not selected[:70].any()
    assert report['status'] == 'boundary_found'
    weak, _, _, report = visual_boundary(scores * .1, generic * .1)
    assert not weak.any()
    assert report['status'] == 'weak_absolute_support'


def test_graph_filter_cannot_add_an_image_without_query_support():
    scores = np.r_[np.linspace(-.1, -.08, 30), np.linspace(.16, .18, 10)]
    base, *_ = visual_boundary(scores, np.zeros(40))
    neighbors = np.tile(np.arange(10), (40, 1))
    graph, *_ = visual_boundary(scores, np.zeros(40), neighbors=neighbors)
    assert np.all(~graph | base)
    assert not graph.any()


class FakeSearch:
    def __init__(self, root):
        self.root = root
        self.items = [{'item_id': str(i), 'title': 'arbitrary'} for i in range(40)]
        self.vector_rows = list(range(40))
        self.by_id = {item['item_id']: item for item in self.items}
        self.embeddings = np.ones((40, 2))
        self.index_sha256 = 'same-pixels'

    def _dense(self, query):
        if query == 'a photograph of a museum object':
            return np.zeros(40)
        return np.r_[np.linspace(-.1, -.08, 30), np.linspace(.16, .18, 10)]


def test_full_set_is_returned_and_catalogue_text_does_not_change_decisions(tmp_path):
    search = FakeSearch(tmp_path)
    service = VisualSetService(search)
    first = service.search('a vessel with unfamiliar decoration', top_k=2)
    assert first['matched_count'] > 2
    assert first['returned'] == first['matched_count'] == len(first['results'])
    assert len(first['preview']) == 2
    for item in search.items:
        item['title'] = 'conflicting label text'
        item['material'] = ['wrong value']
    second = service.search('a vessel with unfamiliar decoration', top_k=1)
    assert first['matched_item_ids'] == second['matched_item_ids']
    assert second['provenance']['candidate_catalogue_used_for_decisions'] is False


def test_visible_motif_contradiction_removes_whole_query_match(tmp_path):
    search = FakeSearch(tmp_path)
    original = search._dense

    def score(query):
        if 'decorated with' in query:
            values = np.full(40, .1)
            if 'lotus flowers' in query:
                values[:] = .2
                values[39] = .05  # Whole-query score is high, motif support is low.
            return values
        return original(query)

    search._dense = score
    service = VisualSetService(search)
    base = service.search('porcelain bowl with lotus flowers', method='adaptive_margin')
    result = service.search('porcelain bowl with lotus flowers')
    assert '39' in base['matched_item_ids']
    assert '39' not in result['matched_item_ids']
    assert result['matched_count'] == base['matched_count'] - 1
    assert any(e['signal'].endswith('/lotus') for e in result['results'][0]['auxiliary_evidence'])


def test_negative_query_is_not_silently_rewritten_to_positive(tmp_path):
    service = VisualSetService(FakeSearch(tmp_path))
    query = '没有龙纹的青花瓷碗'
    assert service.search(query)['encoded_query'] == query
    with pytest.raises(ValueError, match='non-whitespace'):
        service.search('   ')
