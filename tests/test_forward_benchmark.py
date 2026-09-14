import numpy as np
import pandas as pd
import pytest

from football_edge.forward_benchmark import (
    BenchmarkConfig,
    evaluate_benchmark_bets,
    prepare_benchmark_dataset,
)
from football_edge.forward_polymarket import CurrentSeasonFixtureSelection


def _fixture(
    *,
    match_date: str = "2026-09-05",
    release: str = "2026-09-04T16:00:00Z",
    kickoff: str = "2026-09-05T14:00:00Z",
    odds: float = 2.0,
    q_hat: float = 0.60,
    result: float = 1.0,
    home_team: str = "A",
    accepted: bool = True,
) -> dict:
    p_fd = 1.0 / odds if odds > 1 else np.nan
    return {
        "match_date": match_date,
        "date": pd.Timestamp(match_date),
        "season": "26_27",
        "league": "La Liga",
        "home_team": home_team,
        "away_team": "B",
        "football_data_release_utc": pd.Timestamp(release),
        "kickoff_utc": pd.Timestamp(kickoff),
        "football_data_max_home_odds": odds,
        "p_fd": p_fd,
        "g_hat_fd": q_hat - p_fd if pd.notna(p_fd) else np.nan,
        "q_hat_fd": q_hat,
        "accepted_probability_min": 0.48,
        "accepted_probability_max": 0.99,
        "accepted_fd": accepted,
        "valid_selection_input": odds > 1,
        "home_goals": 2 if result == 1 else 0,
        "away_goals": 0 if result == 1 else 1,
        "home_win_result": result,
    }


def _selection(rows: list[dict]) -> CurrentSeasonFixtureSelection:
    fixtures = pd.DataFrame(rows)
    return CurrentSeasonFixtureSelection(
        season="26_27",
        frozen_parameters=pd.DataFrame(),
        observed_fixtures=fixtures,
        selected_fixtures=fixtures.loc[fixtures["accepted_fd"]].copy(),
        skipped_sheets=pd.DataFrame(),
    )


def test_benchmark_includes_only_settled_frozen_rule_matches() -> None:
    dataset = prepare_benchmark_dataset(
        _selection(
            [
                _fixture(home_team="selected"),
                _fixture(home_team="rejected", accepted=False),
                _fixture(home_team="unsettled", result=np.nan),
            ]
        )
    )

    assert dataset.audit["qualifies_frozen_rule"].sum() == 2
    assert list(dataset.bets["home_team"]) == ["selected"]
    total = dataset.coverage.iloc[0]
    assert total["matches_loaded"] == 3
    assert total["qualifying_matches"] == 2
    assert total["settled_qualifying_bets"] == 1


def test_analysis_end_date_is_inclusive() -> None:
    dataset = prepare_benchmark_dataset(
        _selection(
            [
                _fixture(match_date="2026-08-30", home_team="before"),
                _fixture(match_date="2026-08-31", home_team="on cutoff"),
                _fixture(match_date="2026-09-01", home_team="after"),
            ]
        ),
        analysis_end_date="2026-08-31",
    )

    assert list(dataset.audit["home_team"]) == ["before", "on cutoff"]
    assert list(dataset.bets["home_team"]) == ["before", "on cutoff"]


def test_flat_and_kelly_stakes_and_standard_pnl_are_correct() -> None:
    dataset = prepare_benchmark_dataset(_selection([_fixture()]))
    config = BenchmarkConfig(
        initial_capital_usd=100.0,
        flat_stake_usd=10.0,
        kelly_fractions=(0.10, 0.25),
        maximum_kelly_stake_fraction=0.05,
    )
    evaluation = evaluate_benchmark_bets(dataset.bets, config=config)
    settled = evaluation.simulation.bets.set_index("strategy")

    assert settled.loc["flat stake", "total_stake_usd"] == pytest.approx(10.0)
    assert settled.loc["flat stake", "profit_usd"] == pytest.approx(10.0)
    # q=0.60 and odds=2 imply full Kelly=0.20.
    assert settled.loc["10% Kelly", "total_stake_usd"] == pytest.approx(2.0)
    assert settled.loc["25% Kelly", "total_stake_usd"] == pytest.approx(5.0)
    assert settled.loc["10% Kelly", "profit_usd"] == pytest.approx(2.0)
    assert settled.loc["25% Kelly", "profit_usd"] == pytest.approx(5.0)


def test_kelly_sizing_uses_only_bankroll_realized_before_order_time() -> None:
    rows = [
        _fixture(
            home_team="first",
            release="2026-09-04T10:00:00Z",
            kickoff="2026-09-04T12:00:00Z",
        ),
        _fixture(
            home_team="overlapping",
            release="2026-09-04T11:00:00Z",
            kickoff="2026-09-04T15:00:00Z",
        ),
        _fixture(
            home_team="after settlement",
            release="2026-09-04T14:30:00Z",
            kickoff="2026-09-04T18:00:00Z",
        ),
    ]
    dataset = prepare_benchmark_dataset(_selection(rows))
    evaluation = evaluate_benchmark_bets(
        dataset.bets,
        config=BenchmarkConfig(
            initial_capital_usd=100.0,
            flat_stake_usd=10.0,
            kelly_fractions=(0.10,),
        ),
    )
    kelly = (
        evaluation.simulation.bets.loc[
            evaluation.simulation.bets["strategy"].eq("10% Kelly")
        ]
        .set_index("home_team")
    )

    assert kelly.loc["first", "bankroll_at_order_usd"] == pytest.approx(100.0)
    assert kelly.loc["overlapping", "bankroll_at_order_usd"] == pytest.approx(100.0)
    assert kelly.loc["after settlement", "bankroll_at_order_usd"] == pytest.approx(102.0)


def test_period_aggregation_uses_total_stake_and_standard_returns() -> None:
    dataset = prepare_benchmark_dataset(
        _selection(
            [
                _fixture(home_team="win", result=1.0),
                _fixture(
                    home_team="loss",
                    match_date="2026-09-12",
                    release="2026-09-11T16:00:00Z",
                    kickoff="2026-09-12T14:00:00Z",
                    result=0.0,
                ),
            ]
        )
    )
    evaluation = evaluate_benchmark_bets(
        dataset.bets,
        config=BenchmarkConfig(initial_capital_usd=100.0, flat_stake_usd=10.0),
    )
    flat = evaluation.metrics.set_index("strategy").loc["flat stake"]

    assert flat["total_stake_usd"] == pytest.approx(20.0)
    assert flat["profit_usd"] == pytest.approx(0.0)
    assert flat["stake_roi_pct"] == pytest.approx(0.0)
    assert flat["final_bankroll_usd"] == pytest.approx(100.0)
    assert evaluation.monthly_returns.loc[
        evaluation.monthly_returns["strategy"].eq("flat stake"), "bets"
    ].sum() == 2
    assert evaluation.weekly_returns.loc[
        evaluation.weekly_returns["strategy"].eq("flat stake"), "bets"
    ].sum() == 2
