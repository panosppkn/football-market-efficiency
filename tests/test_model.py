import numpy as np

from football_edge.model import fit_logistic_regression, predict_probability


def test_logistic_probabilities_are_finite_and_ordered() -> None:
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    model = fit_logistic_regression(X, y)
    probability = predict_probability(model, X)

    assert np.isfinite(probability).all()
    assert ((probability > 0) & (probability < 1)).all()
    assert np.all(np.diff(probability) > 0)
    assert model.iterations <= 100
