from types import SimpleNamespace

import numpy as np
import pytest

from curio.represent.image_encoder import DINO_ID, DINO_REVISION
from curio.represent.patch_match import (
    DinoPatchMatcher, foreground_patch_features, match_patch_features, rerank_patch_candidates,
)


def patch_grid(kind="vertical"):
    tokens = np.zeros((16, 16, 4), dtype=np.float32)
    tokens[..., 0] = 1
    if kind == "vertical":
        tokens[3:13, 6:10] = [0, 1, 0, 0]
    elif kind == "horizontal":
        tokens[6:10, 3:13] = [0, 1, 0, 0]
    elif kind == "different_texture":
        tokens[3:13, 6:10] = [0, 0, 1, 0]
    elif kind == "edge":
        tokens[:, :4] = [0, 1, 0, 0]
    elif kind == "fragmented":
        tokens[2:5, 2:5] = [0, 1, 0, 0]
        tokens[11:14, 11:14] = [0, 1, 0, 0]
    return tokens.reshape(-1, 4)


def feature(kind="vertical"):
    return foreground_patch_features(patch_grid(kind))


def test_foreground_profile_is_repeatable_and_preserves_aspect_discrimination():
    first, repeated = feature(), feature()
    assert first["diagnostics"]["status"] == "valid"
    assert first["diagnostics"]["learned_segmentation"] is False
    assert first["diagnostics"]["complete_object_coverage_verified"] is False
    np.testing.assert_array_equal(first["foreground_weights"], repeated["foreground_weights"])
    np.testing.assert_array_equal(first["profile"], repeated["profile"])
    matching = match_patch_features(first, repeated)
    horizontal = match_patch_features(first, feature("horizontal"))
    assert matching["local_score"] == pytest.approx(1)
    assert matching["shape_similarity"] == pytest.approx(1)
    assert horizontal["local_score"] == pytest.approx(1)
    assert horizontal["shape_similarity"] < matching["shape_similarity"]


def test_local_descriptor_distinguishes_same_shape_with_different_patch_features():
    same = match_patch_features(feature(), feature())
    different = match_patch_features(feature(), feature("different_texture"))
    assert same["shape_similarity"] == pytest.approx(different["shape_similarity"])
    assert same["local_score"] > different["local_score"]
    assert same["combined_score"] > different["combined_score"]


@pytest.mark.parametrize("kind", ["blank", "edge", "fragmented"])
def test_unreliable_foregrounds_abstain_instead_of_matching_background(kind):
    rejected = feature(kind)
    assert rejected["diagnostics"]["status"] == "unknown"
    scores = match_patch_features(feature(), rejected)
    assert scores == {"status": "unknown", "local_score": None, "shape_similarity": None, "combined_score": None}


def test_unknown_candidates_keep_original_slots_and_scores_do_not_fill_unknown_values():
    candidates = [feature("horizontal"), feature("blank"), feature()]
    result = rerank_patch_candidates(feature(), candidates, [.9, .8, .7], ["a", "b", "c"])
    assert result["order"] == [2, 1, 0]
    assert result["coverage"] == pytest.approx(2 / 3)
    assert result["details"][1]["combined_score"] is None
    assert result["accepted_candidate_count"] == 2


def test_unknown_reference_returns_explicit_cls_fallback():
    result = rerank_patch_candidates(feature("blank"), [feature(), feature()], [.7, .9], ["a", "b"])
    assert result["status"] == "fallback_cls"
    assert result["order"] == [1, 0]
    assert result["coverage"] == 0
    assert result["reference_status"] == "unknown"


def test_reranking_is_deterministic_under_repeated_calls_and_score_ties():
    result = rerank_patch_candidates(feature(), [feature(), feature()], [.8, .8], ["z", "a"])
    assert result["order"] == [1, 0]
    again = rerank_patch_candidates(feature(), [feature(), feature()], [.8, .8], ["z", "a"])
    assert result == again


@pytest.mark.parametrize("tokens,grid", [
    (np.zeros((16, 4)), None), (np.ones((15, 4)), None),
    (np.full((16, 4), np.nan), None), (np.ones((16, 4)), (2, 8)),
])
def test_invalid_patch_matrices_are_rejected(tokens, grid):
    with pytest.raises(ValueError):
        foreground_patch_features(tokens, grid)


def test_reuses_existing_pinned_encoder_without_constructing_new_model():
    encoder = SimpleNamespace(metadata={"model_id": DINO_ID, "revision": DINO_REVISION})
    matcher = DinoPatchMatcher(encoder=encoder)
    assert matcher.encoder is encoder
    assert matcher.metadata["feature"] == "last_hidden_state_patch_tokens"
    assert matcher.encode_images([]) == []
    with pytest.raises(ValueError):
        DinoPatchMatcher(encoder=SimpleNamespace(metadata={"model_id": DINO_ID, "revision": "wrong"}))


def test_reranker_validates_candidate_alignment():
    with pytest.raises(ValueError):
        rerank_patch_candidates(feature(), [feature()], [], ["a"])
    with pytest.raises(ValueError):
        rerank_patch_candidates(feature(), [feature(), feature()], [.8, .7], ["a", "a"])
    with pytest.raises(ValueError):
        rerank_patch_candidates(feature(), [feature()], [np.nan], ["a"])
