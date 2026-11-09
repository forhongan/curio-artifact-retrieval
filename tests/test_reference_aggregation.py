import numpy as np
import pytest

from curio.retrieve.reference_aggregation import METHODS, aggregate_references


def example(query=None, **options):
    gallery = np.array([[1., 0.], [0., 1.], [.8, .2]])
    return aggregate_references(gallery, gallery, np.eye(2), np.eye(2), query, **options)


def test_soft_formula_and_dino_transfer_are_explicit():
    result = aggregate_references(np.eye(2), np.eye(2), np.eye(2), np.eye(2)[::-1],
                                  [1, 0], temperature=1, uniform_floor=.1)
    expected = .9 * np.exp([1, 0]) / np.exp([1, 0]).sum() + .05
    np.testing.assert_allclose(result["references"]["query_cosines"], [1, 0])
    np.testing.assert_allclose(result["references"]["soft_weights"], expected)
    method = result["methods"]["query_aligned"]
    np.testing.assert_allclose(method["scores"]["siglip"], expected)
    np.testing.assert_allclose(method["scores"]["dino"], expected[::-1])
    assert method["text_assisted_reference_selection_or_weighting"] is True


def test_query_does_not_add_direct_query_gallery_similarity():
    gallery = np.array([[0, 0, 1], [0, 0, -1]])
    refs = np.array([[1, 0, 0], [0, 1, 0]])
    first = aggregate_references(gallery, gallery, refs, refs, [1, 1, 1])
    second = aggregate_references(gallery, gallery, refs, refs, [1, 1, -1])
    # Direct text/gallery similarities would reverse sign; reference alignment
    # is unchanged, so reference-weighted candidate scores must be identical.
    np.testing.assert_allclose(first["references"]["soft_weights"], [.5, .5])
    for channel in ("siglip", "dino", "fusion"):
        np.testing.assert_allclose(first["methods"]["query_aligned"]["scores"][channel],
                                   second["methods"]["query_aligned"]["scores"][channel])


def test_all_reference_weights_stay_positive_even_at_extreme_alignment():
    result = aggregate_references(np.eye(2), np.eye(2), [[1, 0], [-1, 0]],
                                  [[1, 0], [-1, 0]], [1, 0], temperature=1e-12)
    np.testing.assert_allclose(result["references"]["soft_weights"], [.95, .05])
    assert (result["references"]["soft_weights"] > 0).all()
    assert np.isclose(result["references"]["soft_weights"].sum(), 1)


def test_uniform_floor_one_disables_query_weight_preference():
    result = example([1, 0], uniform_floor=1)
    for channel in ("siglip", "dino", "fusion"):
        np.testing.assert_allclose(result["methods"]["query_aligned"]["scores"][channel],
                                   result["methods"]["uniform_mean"]["scores"][channel])


def test_missing_query_is_unavailable_without_faking_uniform_query_alignment():
    result = example()
    assert result["references"]["query_cosines"] is None
    assert result["references"]["soft_weights"] is None
    assert result["methods"]["query_aligned"]["status"] == "unavailable_query_vector_missing"
    assert result["methods"]["query_aligned"]["scores"] is None
    assert result["methods"]["uniform_mean"]["status"] == "available"


def test_single_best_uses_query_or_reference_medoid_never_gallery_oracle():
    refs = np.array([[1, 0], [1, 0], [0, 1]])
    without = aggregate_references(np.eye(2), np.eye(2), refs, refs)
    with_query = aggregate_references(np.eye(2), np.eye(2), refs, refs, [0, 1])
    assert without["references"]["single_best_index"] == 0
    assert without["references"]["single_best_selection"] == "joint_image_reference_medoid"
    assert with_query["references"]["single_best_index"] == 2
    assert with_query["references"]["single_best_selection"].startswith("highest_original_query")
    np.testing.assert_allclose(with_query["methods"]["single_best"]["reference_weights"], [0, 0, 1])


def test_one_reference_all_aggregations_collapse_to_same_scores():
    result = aggregate_references(np.eye(2), np.eye(2), [[1, 0]], [[0, 1]], [0, 1])
    assert result["references"]["single_best_index"] == 0
    np.testing.assert_array_equal(result["references"]["soft_weights"], [1])
    for method in METHODS:
        for channel in ("siglip", "dino", "fusion"):
            np.testing.assert_allclose(result["methods"][method]["scores"][channel],
                                       result["methods"]["single_best"]["scores"][channel])


def test_identical_references_have_uniform_weights_and_no_artificial_consensus_gain():
    refs = np.tile([1., 0.], (4, 1))
    result = aggregate_references(np.eye(2), np.eye(2), refs, refs, [1, 0])
    np.testing.assert_allclose(result["references"]["soft_weights"], [.25] * 4)
    assert result["references"]["visual_clusters"] == [[0, 1, 2, 3]]
    for method in METHODS:
        np.testing.assert_allclose(result["methods"][method]["scores"]["siglip"], [1, 0])


def test_empty_reference_set_never_fabricates_ranked_evidence():
    result = aggregate_references(np.eye(2), np.eye(2), np.empty((0, 2)),
                                  np.empty((0, 2)), [1, 0])
    assert result["status"] == "no_references"
    assert result["pairwise_scores"]["siglip"].shape == (2, 0)
    assert result["references"]["uniform_weights"].size == 0
    for method in METHODS:
        assert result["methods"][method]["status"] == "unavailable_no_references"
        assert result["methods"][method]["scores"] is None
        assert result["methods"][method]["orders"] is None


def test_empty_gallery_has_empty_arrays_but_valid_reference_weights():
    result = aggregate_references(np.empty((0, 2)), np.empty((0, 3)), np.eye(2),
                                  [[1, 0, 0], [0, 1, 0]], [1, 0])
    assert result["status"] == "empty_gallery"
    assert result["references"]["soft_weights"].shape == (2,)
    for method in METHODS:
        assert result["methods"][method]["scores"]["fusion"].shape == (0,)
        assert result["methods"][method]["orders"]["fusion"].size == 0


def test_max_reports_candidate_specific_reference_contributors():
    result = example([1, 0])
    assert result["methods"]["max"]["reference_weights"] is None
    np.testing.assert_array_equal(result["methods"]["max"]["argmax_reference_indices"]["siglip"], [0, 1, 0])


def test_tie_aware_fusion_does_not_turn_ids_into_evidence():
    refs = [[1, 0]]
    result = aggregate_references([[1, 0]] * 3, [[1, 0]] * 3, refs, refs,
                                  gallery_ids=["c", "a", "b"])
    scores = result["methods"]["uniform_mean"]["scores"]["fusion"]
    np.testing.assert_allclose(scores, [61 / 62] * 3)
    np.testing.assert_array_equal(result["methods"]["uniform_mean"]["orders"]["fusion"], [1, 2, 0])


def test_query_weighted_scores_are_equivariant_to_reference_order():
    rng = np.random.default_rng(7)
    sg, dg, sr, dr, q = rng.normal(size=(6, 4)), rng.normal(size=(6, 3)), rng.normal(size=(4, 4)), rng.normal(size=(4, 3)), rng.normal(size=4)
    permutation = np.array([2, 0, 3, 1])
    first = aggregate_references(sg, dg, sr, dr, q)
    second = aggregate_references(sg, dg, sr[permutation], dr[permutation], q)
    np.testing.assert_allclose(second["references"]["soft_weights"], first["references"]["soft_weights"][permutation])
    for channel in ("siglip", "dino", "fusion"):
        np.testing.assert_allclose(first["methods"]["query_aligned"]["scores"][channel],
                                   second["methods"]["query_aligned"]["scores"][channel])


def test_diverse_control_matches_existing_runtime_numerically():
    from curio.retrieve.dynamic_reference_runtime import score_gallery
    rng = np.random.default_rng(21)
    sg, dg = rng.normal(size=(7, 4)), rng.normal(size=(7, 3))
    sr, dr = rng.normal(size=(4, 4)), rng.normal(size=(4, 3))
    ids = [f"item_{i}" for i in range(7)]
    result = aggregate_references(sg, dg, sr, dr, gallery_ids=ids)
    old = score_gallery(ids, sg, dg, sr, dr)
    np.testing.assert_allclose(result["methods"]["diverse"]["scores"]["siglip"], old["channels"]["siglip_diverse"], atol=1e-6)
    np.testing.assert_allclose(result["methods"]["diverse"]["scores"]["dino"], old["channels"]["dino_diverse"], atol=1e-6)
    np.testing.assert_allclose(result["methods"]["diverse"]["scores"]["fusion"], old["channels"]["fusion"], atol=1e-6)


def test_opposing_references_have_finite_mean_without_centroid_normalization():
    result = aggregate_references(np.eye(2), np.eye(2), [[1, 0], [-1, 0]],
                                  [[1, 0], [-1, 0]], [1, 0])
    np.testing.assert_array_equal(result["methods"]["uniform_mean"]["scores"]["siglip"], [0, 0])
    assert np.isfinite(result["methods"]["query_aligned"]["scores"]["fusion"]).all()


def test_input_scaling_and_single_row_query_shape_do_not_change_scores_or_mutate_inputs():
    gallery, refs, query = np.eye(2), np.eye(2), np.array([1., 1.])
    original = gallery.copy()
    first = aggregate_references(gallery, gallery, refs, refs, query)
    second = aggregate_references(gallery * 1e300, gallery * 1e-300,
                                   refs * 7, refs * 9, query[None, :] * 13)
    np.testing.assert_array_equal(gallery, original)
    np.testing.assert_allclose(first["methods"]["query_aligned"]["scores"]["siglip"],
                               second["methods"]["query_aligned"]["scores"]["siglip"])


@pytest.mark.parametrize("options", [
    {"temperature": 0}, {"temperature": float("nan")}, {"uniform_floor": 0},
    {"uniform_floor": 1.1}, {"siglip_weight": -.1}, {"siglip_weight": float("inf")},
    {"gallery_ids": ["a", "a", "b"]}, {"gallery_ids": ["a"]},
])
def test_invalid_parameters_fail_explicitly(options):
    with pytest.raises(ValueError):
        example(**options)


@pytest.mark.parametrize("query", [[0, 0], [float("nan"), 0], [1, 0, 0], [[1, 0], [0, 1]]])
def test_invalid_query_vectors_fail_explicitly(query):
    with pytest.raises(ValueError):
        example(query)


def test_misaligned_channel_rows_dimensions_and_zero_vectors_fail():
    with pytest.raises(ValueError, match="Gallery channel row"):
        aggregate_references(np.eye(2), [[1, 0]], np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match="Reference channel row"):
        aggregate_references(np.eye(2), np.eye(2), [[1, 0]], np.eye(2))
    with pytest.raises(ValueError, match="dimensions"):
        aggregate_references(np.eye(2), np.eye(2), [[1, 0, 0]], [[1, 0]])
    with pytest.raises(ValueError, match="zero vector"):
        aggregate_references([[0, 0]], [[1, 0]], [[1, 0]], [[1, 0]])
