from football_edge.backtest import (
    build_research_dataset,
    probability_performance,
    run_regularization_grid,
    run_walk_forward_with_coefficients,
)
from football_edge.config import REGULARIZATION_GRID
from football_edge.data import discover_datasets


def test_full_research_pipeline_is_chronological() -> None:
    datasets = discover_datasets()
    all_matches = build_research_dataset(datasets)
    predictions, coefficients = run_walk_forward_with_coefficients(all_matches)

    assert len(datasets) == 20
    assert len(all_matches) == 7_230
    assert len(predictions) == 5_628
    assert predictions["model_probability"].between(0, 1).all()
    assert (predictions["train_end_date"] < predictions["test_start_date"]).all()
    assert len(coefficients) == 16 * 6
    assert coefficients.groupby(["league", "test_season"]).size().eq(6).all()

    pooled_predictions, pooled_coefficients = run_regularization_grid(
        all_matches,
        REGULARIZATION_GRID,
        training_window=2,
    )
    assert pooled_predictions.groupby("model_name").size().nunique() == 1
    assert set(pooled_predictions["season"]) == {"23_24", "24_25", "25_26"}
    assert (
        pooled_predictions["train_end_date"]
        < pooled_predictions["test_start_date"]
    ).all()
    assert len(pooled_coefficients) == len(REGULARIZATION_GRID) * 3 * 9

    metrics = probability_performance(
        pooled_predictions,
        probability_column="model_probability",
        group_by=["model_name"],
    )
    assert len(metrics) == len(REGULARIZATION_GRID)
    assert metrics["brier_score"].between(0, 1).all()
