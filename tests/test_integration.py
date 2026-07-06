from pathlib import Path

import pandas as pd

from football_edge.backtest import (
    build_research_dataset,
    probability_performance,
    run_regularization_grid,
    run_walk_forward_with_coefficients,
)
from football_edge.config import REGULARIZATION_GRID
from football_edge.data import Dataset


def _synthetic_datasets(tmp_path: Path) -> list[Dataset]:
    """Create deterministic league-season CSVs for a portable pipeline test."""
    seasons = ["21_22", "22_23", "23_24", "24_25", "25_26"]
    datasets = []

    for league_index in range(4):
        league = f"League {league_index + 1}"
        teams = [f"L{league_index} Team {index}" for index in range(4)]
        for season_index, season in enumerate(seasons):
            start = pd.Timestamp(f"20{season[:2]}-08-01")
            rows = []
            for match_index in range(12):
                home_index = match_index % len(teams)
                away_index = (match_index + 1 + match_index // 4) % len(teams)
                rows.append(
                    {
                        "Date": (start + pd.Timedelta(days=7 * match_index)).strftime(
                            "%d/%m/%Y"
                        ),
                        "Time": "15:00",
                        "HomeTeam": teams[home_index],
                        "AwayTeam": teams[away_index],
                        "FTHG": (match_index + league_index + season_index) % 4,
                        "FTAG": (2 * match_index + season_index) % 3,
                        "Avg>2.5": 1.85 + 0.02 * (match_index % 4),
                        "Avg<2.5": 1.90 + 0.02 * (match_index % 3),
                    }
                )

            path = tmp_path / f"league_{league_index}_{season}.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            datasets.append(Dataset(path=path, league=league, season=season))

    return datasets


def test_full_research_pipeline_is_chronological(tmp_path: Path) -> None:
    datasets = _synthetic_datasets(tmp_path)
    all_matches = build_research_dataset(datasets)
    predictions, coefficients = run_walk_forward_with_coefficients(all_matches)

    assert len(datasets) == 20
    assert len(all_matches) == 240
    assert not predictions.empty
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
