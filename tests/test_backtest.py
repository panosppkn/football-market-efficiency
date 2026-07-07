import numpy as np
import pandas as pd

from football_edge.backtest import (
    build_execution_candidates,
    bookmaker_coverage,
    ev_bucket_summary,
    monthly_roi,
    select_bets,
    select_bookmaker_bets,
    summarize_bets,
)


def test_bet_settlement_and_roi() -> None:
    predictions = pd.DataFrame(
        {
            "model_probability": [0.75, 0.75],
            "over_2_5": [1, 0],
            "Avg>2.5": [2.0, 2.0],
            "AvgC>2.5": [1.9, 1.9],
            "Max>2.5": [2.1, 2.1],
            "MaxC>2.5": [2.0, 2.0],
        }
    )

    bets = select_bets(predictions, minimum_expected_value=0.0)
    average_bets = bets.loc[bets["price_scenario"].eq("average_preclosing")]
    summary = summarize_bets(average_bets, ["price_scenario"]).iloc[0]

    assert np.allclose(average_bets["profit"], [1.0, -1.0])
    assert summary["bets"] == 2
    assert summary["profit_units"] == 0
    assert summary["roi_pct"] == 0


def test_aggregate_quotes_at_or_below_one_are_unavailable() -> None:
    predictions = pd.DataFrame(
        {
            "model_probability": [0.75, 0.75],
            "over_2_5": [1, 1],
            "Avg>2.5": [2.0, 0.0],
            "AvgC>2.5": [1.9, 1.9],
            "Max>2.5": [2.1, 2.1],
            "MaxC>2.5": [2.0, 0.0],
        }
    )

    bets = select_bets(predictions, minimum_expected_value=0.0)

    assert bets.groupby("price_scenario").size().eq(1).all()


def _bookmaker_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_probability": [0.75, 0.75],
            "over_2_5": [1, 0],
            "B365>2.5": [2.0, 2.0],
            "B365C>2.5": [1.9, 1.9],
            "P>2.5": [2.1, 2.1],
            "PC>2.5": [2.0, 2.0],
            "BFE>2.5": [2.2, np.nan],
            "BFEC>2.5": [2.1, np.nan],
        }
    )


def test_named_bookmakers_share_probabilities_but_use_own_prices() -> None:
    bets = select_bookmaker_bets(
        _bookmaker_predictions(), minimum_expected_value=0.0
    )

    assert set(bets["bookmaker"]) == {
        "bet365",
        "pinnacle",
        "betfair_exchange",
    }
    assert bets.groupby("bookmaker")["model_probability"].nunique().eq(1).all()

    betfair = bets.loc[bets["bookmaker"].eq("betfair_exchange")].iloc[0]
    assert np.isclose(betfair["bet_odds"], 2.2)
    assert np.isclose(betfair["profit"], 1.2)


def test_common_sample_requires_all_three_sources() -> None:
    predictions = _bookmaker_predictions()
    coverage = bookmaker_coverage(predictions, common_sample=True)
    bets = select_bookmaker_bets(
        predictions,
        minimum_expected_value=0.0,
        common_sample=True,
    )

    assert coverage["comparison_matches"].eq(1).all()
    assert coverage["coverage_pct"].eq(100).all()
    assert bets.groupby("bookmaker").size().eq(1).all()


def test_monthly_roi_uses_profit_over_flat_stakes() -> None:
    bets = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-08-01", "2024-08-20", "2024-09-02"]
            ),
            "league": ["League A"] * 3,
            "execution_source": ["source"] * 3,
            "over_2_5": [1, 0, 1],
            "profit": [1.0, -1.0, 0.8],
        }
    )

    result = monthly_roi(
        bets, ["league", "execution_source"]
    ).set_index("month")

    assert result.loc[pd.Timestamp("2024-08-01"), "bets"] == 2
    assert result.loc[pd.Timestamp("2024-08-01"), "profit_units"] == 0
    assert result.loc[pd.Timestamp("2024-08-01"), "roi_pct"] == 0
    assert result.loc[pd.Timestamp("2024-09-01"), "staked_units"] == 1
    assert result.loc[pd.Timestamp("2024-09-01"), "roi_pct"] == 80


def test_drawdown_is_computed_in_chronological_order() -> None:
    bets = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-08-02", "2024-08-01", "2024-08-03"]
            ),
            "source": ["test"] * 3,
            "over_2_5": [0, 1, 0],
            "bet_odds": [2.0, 2.0, 2.0],
            "closing_odds": [2.0, 2.0, 2.0],
            "profit": [-1.0, 1.0, -1.0],
        }
    )

    summary = summarize_bets(bets, ["source"]).iloc[0]

    assert summary["max_drawdown_units"] == 2.0


def test_ev_buckets_include_all_valid_prices_without_thresholding() -> None:
    predictions = _bookmaker_predictions().assign(
        league="League A",
        season="24_25",
        date=pd.to_datetime(["2024-08-01", "2024-08-02"]),
        **{
            "Avg>2.5": [2.0, 2.0],
            "AvgC>2.5": [1.9, 1.9],
            "Max>2.5": [2.1, 2.1],
            "MaxC>2.5": [2.0, 2.0],
        },
    )

    candidates = build_execution_candidates(predictions)
    summary = ev_bucket_summary(candidates)

    assert set(candidates["execution_source"]) == {
        "bet365",
        "pinnacle",
        "betfair_exchange",
        "average_preclosing",
        "best_preclosing",
    }
    assert len(candidates.loc[candidates["execution_source"].eq("bet365")]) == 2
    assert summary["candidates"].sum() == len(candidates)


def test_execution_candidates_can_include_closing_market_maximum() -> None:
    predictions = _bookmaker_predictions().assign(
        league="League A",
        season="24_25",
        date=pd.to_datetime(["2024-08-01", "2024-08-02"]),
        **{
            "Avg>2.5": [2.0, 2.0],
            "AvgC>2.5": [1.9, 1.9],
            "Max>2.5": [2.1, 2.1],
            "MaxC>2.5": [2.2, 2.2],
        },
    )

    candidates = build_execution_candidates(
        predictions, include_closing_prices=True
    )
    closing = candidates.loc[
        candidates["execution_source"].eq("best_closing")
    ]

    assert len(closing) == 2
    assert closing["bet_odds"].eq(2.2).all()
    assert closing["closing_odds"].isna().all()
    assert closing["clv_pct"].isna().all()


def test_pooled_walk_forward_accepts_custom_feature_columns() -> None:
    from football_edge.backtest import run_pooled_rolling_walk_forward

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2021-08-01",
                    "2021-08-02",
                    "2022-08-01",
                    "2022-08-02",
                    "2023-08-01",
                    "2023-08-02",
                ]
            ),
            "league": ["League A", "League B"] * 3,
            "season": ["21_22", "21_22", "22_23", "22_23", "23_24", "23_24"],
            "market_logit": [-2.0, 2.0, -1.5, 1.5, -1.0, 1.0],
            "custom_feature": [0.1, 1.0, 0.2, 0.9, 0.3, 0.8],
            "over_2_5": [0, 1, 0, 1, 0, 1],
        }
    )

    predictions, coefficients = run_pooled_rolling_walk_forward(
        frame,
        l2=10.0,
        model_name="custom",
        training_window=2,
        feature_columns=["market_logit", "custom_feature"],
    )

    assert len(predictions) == 2
    assert predictions["model_probability"].between(0, 1).all()
    assert "custom_feature" in set(coefficients["coefficient"])
