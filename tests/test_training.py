import numpy as np
import pytest
import torch

from curio.represent.train import multi_positive_loss, paired_retrieval


def test_repeated_captions_are_positives():
    features = torch.tensor([[1.0, 0], [1.0, 0], [0, 1.0]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=torch.bool)
    loss = multi_positive_loss(features, features, mask)
    # Equal duplicated positives are rewarded equally; irreducible log(2) mass.
    assert loss == pytest.approx(2 / 3 * np.log(2), abs=1e-5)
    with pytest.raises(AssertionError):
        assert float(loss) == 0
    metrics = paired_retrieval(features.numpy(), features.numpy(), ["vase", "vase", "bowl"])
    assert metrics["r@1"] == 1
    assert metrics["unique_captions"] == 2
