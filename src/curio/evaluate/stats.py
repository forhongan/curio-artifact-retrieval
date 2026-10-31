import numpy as np


def bootstrap_ci(values, repetitions=2000, seed=5523):
    values = np.asarray(values, dtype=float)
    if not len(values):
        raise ValueError("Cannot bootstrap an empty sample")
    rng = np.random.default_rng(seed)
    means = np.mean(values[rng.integers(len(values), size=(repetitions, len(values)))], axis=1)
    return {
        "mean": float(values.mean()),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "n": len(values),
    }


def paired_permutation(a, b, repetitions=10000, seed=5523):
    delta = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    if not len(delta):
        raise ValueError("Paired observations are required")
    rng = np.random.default_rng(seed)
    null = np.mean(delta * rng.choice([-1, 1], size=(repetitions, len(delta))), axis=1)
    p = (np.sum(np.abs(null) >= abs(delta.mean())) + 1) / (repetitions + 1)
    return {"difference": bootstrap_ci(delta, seed=seed), "p_value": float(p)}


def holm_adjust(p_values):
    order = np.argsort(p_values)
    adjusted, previous = np.empty(len(order)), 0.0
    for rank, index in enumerate(order):
        previous = max(previous, min(1.0, p_values[index] * (len(order) - rank)))
        adjusted[index] = previous
    return adjusted.tolist()
