import numpy as np

from curio.represent.train import make_batches


def test_hard_batches_cover_each_object_once_and_pack_neighbors():
    vectors = np.eye(7, dtype=np.float32)
    vectors[1] = vectors[0] + 0.01
    neighbors = np.argsort(-(vectors @ vectors.T), axis=1, kind="stable")
    batches = make_batches(np.arange(7), 3, neighbors)
    assert sorted(np.concatenate(batches).tolist()) == list(range(7))
    assert {0, 1}.issubset(set(batches[0]))
    assert [len(batch) for batch in batches] == [3, 3, 1]


def test_random_batches_preserve_seeded_order():
    order = np.random.default_rng(5523).permutation(17)
    batches = make_batches(order, 4)
    np.testing.assert_array_equal(np.concatenate(batches), order)
