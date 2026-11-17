"""Constrained evidence combiner: stronger opposition cannot raise acceptance."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def fit_monotone(x, y, *, regularization=1.0):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or len(x) != len(y) or not np.isfinite(x).all():
        raise ValueError("Invalid training matrix")
    if set(y.tolist()) != {0.0, 1.0} or regularization <= 0:
        raise ValueError("Two classes and positive regularization are required")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    z = (x - mean) / scale
    weights = np.where(y == 1, len(y) / (2 * y.sum()), len(y) / (2 * (len(y) - y.sum())))

    def objective(theta):
        coefficients, intercept = theta[:-1], theta[-1]
        logits = z @ coefficients + intercept
        residual = weights * (expit(logits) - y)
        loss = (
            np.sum(weights * (np.logaddexp(0, logits) - y * logits))
            + 0.5 * regularization * coefficients @ coefficients
        )
        gradient = np.r_[z.T @ residual + regularization * coefficients, residual.sum()]
        return loss, gradient

    result = minimize(
        objective,
        np.zeros(z.shape[1] + 1),
        method="L-BFGS-B",
        jac=True,
        bounds=[(0.0, None)] * z.shape[1] + [(None, None)],
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError("Constrained fit failed: " + result.message)
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficient": result.x[:-1].tolist(),
        "intercept": float(result.x[-1]),
        "regularization": regularization,
        "optimizer_success": True,
    }


def predict_monotone(head, x):
    x = np.asarray(x, dtype=float)
    if not np.isfinite(x).all():
        raise ValueError("Invalid inference features")
    coefficients = np.asarray(head["coefficient"])
    scale = np.asarray(head["scale"])
    if (coefficients < 0).any() or (scale <= 0).any():
        raise ValueError("Invalid monotone head")
    return expit(((x - head["mean"]) / scale) @ coefficients + head["intercept"])
