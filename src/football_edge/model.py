"""A small dependency-light logistic model for transparent research."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LogisticModel:
    coefficients: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    iterations: int


def fit_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    l2: float = 1.0,
    max_iter: int = 100,
    tolerance: float = 1e-9,
) -> LogisticModel:
    """Fit standardized L2-regularized logistic regression via Newton steps."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2 or len(X) != len(y) or len(X) == 0:
        raise ValueError("X and y must contain the same non-zero number of rows")

    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale == 0] = 1.0
    design = np.column_stack([np.ones(len(X)), (X - mean) / scale])

    coefficients = np.zeros(design.shape[1])
    penalty = np.r_[0.0, np.ones(design.shape[1] - 1)]

    converged = False
    for iteration in range(1, max_iter + 1):
        linear = np.clip(design @ coefficients, -35, 35)
        probability = 1 / (1 + np.exp(-linear))
        weights = np.maximum(probability * (1 - probability), 1e-9)
        gradient = design.T @ (y - probability) - l2 * penalty * coefficients
        information = design.T @ (design * weights[:, None])
        information += l2 * np.diag(penalty)
        update = np.linalg.solve(information, gradient)
        coefficients += update
        if np.max(np.abs(update)) < tolerance:
            converged = True
            break

    if not converged:
        raise RuntimeError(
            f"Logistic regression did not converge within {max_iter} iterations"
        )

    return LogisticModel(
        coefficients=coefficients,
        mean=mean,
        scale=scale,
        iterations=iteration,
    )


def predict_probability(model: LogisticModel, X: np.ndarray) -> np.ndarray:
    """Return positive-class probabilities."""
    X = np.asarray(X, dtype=float)
    design = np.column_stack(
        [np.ones(len(X)), (X - model.mean) / model.scale]
    )
    linear = np.clip(design @ model.coefficients, -35, 35)
    return 1 / (1 + np.exp(-linear))
