import numpy as np
import pandas as pd

from football_edge.market_efficiency import (
    build_source_panel,
    normalize_season_key,
    prepare_kelly_input,
    run_walk_forward_paper_rule,
    season_to_parquet_name,
    simulate_fractional_kelly_path,
)


def test_season_helpers_normalize_common_formats():
    assert normalize_season_key("2006/2007") == "06_07"
    assert normalize_season_key("06_07") == "06_07"
    assert season_to_parquet_name("06_07") == "all_euro_2006_2007.parquet"


def test_build_source_panel_uses_raw_implied_probability_and_outcome():
    matches = pd.DataFrame(
        {
            "league": ["A", "A"],
            "season": ["20_21", "20_21"],
            "date": pd.to_datetime(["2020-08-01", "2020-08-02"]),
            "home_team": ["H1", "H2"],
            "away_team": ["A1", "A2"],
            "total_ft_goals": [3, 1],
            "FTR": ["H", "A"],
            "AvgH": [2.0, 3.0],
            "AvgD": [3.5, 3.2],
            "AvgA": [4.0, 2.2],
        }
    )
    outcome_config = {
        "home_win": {
            "label": "Home win",
            "odds_key": "home",
            "outcome_function": lambda frame: frame["FTR"].eq("H").astype(int),
        }
    }
    source_specs = {
        "average_market": {
            "odds": {"home": "AvgH"},
            "margin_groups": {"1x2": ["AvgH", "AvgD", "AvgA"]},
            "kind": "market_aggregate",
        }
    }

    panel = build_source_panel(
        matches,
        source_specs,
        market_outcome="home_win",
        outcome_config=outcome_config,
    )

    assert panel["outcome"].tolist() == [1, 0]
    np.testing.assert_allclose(panel["raw_implied_probability"], [0.5, 1 / 3])
    np.testing.assert_allclose(panel["flat_profit"], [1.0, -1.0])


def test_walk_forward_uses_only_prior_seasons():
    rows = []
    for season_index, season in enumerate(["20_21", "21_22", "22_23"]):
        for match_index in range(10):
            p = 0.2 + match_index * 0.05
            rows.append(
                {
                    "source": "market_maximum",
                    "league": "A",
                    "season": season,
                    "date": pd.Timestamp(2020 + season_index, 8, 1)
                    + pd.Timedelta(days=match_index),
                    "home_team": f"H{match_index}",
                    "away_team": f"A{match_index}",
                    "raw_implied_probability": p,
                    "forecast_error_raw": (match_index % 2) - p,
                    "outcome_odds": 1 / p,
                    "outcome": match_index % 2,
                    "flat_profit": (1 / p - 1) if match_index % 2 else -1.0,
                    "market_outcome": "home_win",
                }
            )
    panel = pd.DataFrame(rows)

    result = run_walk_forward_paper_rule(
        panel,
        training_window_seasons=None,
        min_train_seasons=1,
        min_train_matches=5,
        confidence_z=1.96,
        estimation_method="ols_hc1",
    )

    assert set(result["season"]) == {"21_22", "22_23"}
    training_by_season = result.groupby("season")["training_seasons"].first().to_dict()
    assert training_by_season["21_22"] == "20_21"
    assert training_by_season["22_23"] == "20_21,21_22"


def test_kelly_preparation_filters_source_and_simulates_non_negative_stakes():
    bets = pd.DataFrame(
        {
            "source": ["market_maximum", "average_market"],
            "date": pd.to_datetime(["2022-08-01", "2022-08-01"]),
            "season": ["22_23", "22_23"],
            "league": ["A", "A"],
            "home_team": ["H1", "H2"],
            "away_team": ["A1", "A2"],
            "corrected_probability": [0.60, 0.60],
            "outcome_odds": [2.2, 2.2],
            "expected_value": [0.32, 0.32],
            "flat_profit": [1.2, -1.0],
            "outcome": [1, 0],
        }
    )

    kelly_input = prepare_kelly_input(bets, source="market_maximum")
    assert len(kelly_input) == 1
    assert (kelly_input["kelly_full_fraction"] >= 0).all()

    results, path = simulate_fractional_kelly_path(
        kelly_input,
        fraction=0.25,
        initial_bankroll=1.0,
        max_bet_fraction=0.05,
        max_date_exposure_fraction=0.25,
    )
    assert len(results) == 1
    assert results["stake_fraction"].iloc[0] <= 0.05
    assert path["bankroll"].iloc[-1] > 1.0
