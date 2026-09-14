from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_edge.forward_execution import (
    ExecutionConfig,
    evaluate_execution_by_league,
    evaluate_execution_trades,
    load_finalized_holdout_cache,
    period_return_summary,
    prepare_execution_trades,
    raw_price_for_all_in_limit,
    separate_common_strategy_statistics,
)


def collected_trade(
    *,
    tx_hash: str = "0xtrade",
    taker_side: str = "SELL",
    maker_side: str = "BUY",
    raw_price: float = 0.40,
    shares: float = 100.0,
    result: float = 1.0,
    league: str = "La Liga",
) -> dict:
    return {
        "match_date": "2026-09-05",
        "season": "26_27",
        "league": league,
        "home_team": "A",
        "away_team": "B",
        "condition_id": "0xabc",
        "asset_id": "asset",
        "tx_hash": tx_hash,
        "evt_index": 1,
        "is_taker_side": True,
        "maker_side": maker_side,
        "taker_side": taker_side,
        "football_data_release_utc": "2026-09-04T16:00:00Z",
        "kickoff_uk_time": "2026-09-05T15:00:00+01:00",
        "dune_trade_time_utc": "2026-09-05T10:00:00Z",
        "football_data_max_home_odds": 2.0,
        "q_hat_fd": 0.60,
        "dune_home_win_price": raw_price,
        "dune_trade_amount_usd": raw_price * shares,
        "dune_trade_shares": shares,
        "dune_recorded_fee_usd": 0.0,
        "home_goals": 2,
        "away_goals": 0,
        "home_win_result": result,
    }


def test_raw_limit_inverts_the_configured_buyer_fee_once() -> None:
    all_in = pd.Series([0.50, 0.25])
    raw = raw_price_for_all_in_limit(all_in, 0.05)

    reconstructed = raw + 0.05 * raw * (1.0 - raw)
    np.testing.assert_allclose(reconstructed, all_in)


def test_execution_prices_preserve_both_canonical_flow_definitions() -> None:
    config = ExecutionConfig()
    trades = pd.DataFrame(
        [
            collected_trade(tx_hash="0xseller"),
            collected_trade(
                tx_hash="0xbuyer",
                taker_side="BUY",
                maker_side="SELL",
                raw_price=0.45,
            ),
        ]
    )

    prepared, diagnostics = prepare_execution_trades(trades, config=config)
    seller = prepared.loc[prepared["tx_hash"].eq("0xseller")].iloc[0]
    buyer = prepared.loc[prepared["tx_hash"].eq("0xbuyer")].iloc[0]

    assert seller["execution_price_source"] == "predefined_limit"
    assert seller["execution_raw_price"] > seller["observed_raw_price"]
    assert seller["hypothetical_effective_price"] == pytest.approx(1 / 2.02)
    assert buyer["execution_price_source"] == "observed_trade_price"
    assert buyer["execution_raw_price"] == pytest.approx(0.45)
    assert buyer["hypothetical_effective_price"] == pytest.approx(
        0.45 + 0.05 * 0.45 * 0.55
    )
    assert prepared["execution_eligible"].all()
    assert {
        "match_date",
        "league",
        "home_team",
        "away_team",
        "football_data_max_home_odds",
        "condition_id",
    }.issubset(diagnostics.columns)


def test_premium_and_fee_are_applied_without_changing_frozen_probability() -> None:
    trade = collected_trade(
        taker_side="BUY",
        maker_side="SELL",
        raw_price=0.49,
        result=np.nan,
    )
    prepared, _ = prepare_execution_trades(
        pd.DataFrame([trade]), config=ExecutionConfig(minimum_odds_premium=0.01)
    )

    assert prepared.iloc[0]["q_hat_fd"] == 0.60
    assert not bool(prepared.iloc[0]["execution_eligible"])


def test_chronological_flat_stake_fill_and_settlement_are_correct() -> None:
    config = ExecutionConfig(
        initial_capital_usd=100.0,
        flat_stake_usd=10.0,
        kelly_fractions=(0.10, 0.25),
        maximum_trade_participation=0.50,
        maximum_open_exposure_fraction=0.50,
    )
    prepared, _ = prepare_execution_trades(
        pd.DataFrame([collected_trade()]), config=config
    )
    evaluation = evaluate_execution_trades(prepared, config=config)
    flat_bet = evaluation.simulation.bets.loc[
        evaluation.simulation.bets["strategy"].eq("flat stake")
    ].iloc[0]
    flat_fill = evaluation.simulation.fills.loc[
        evaluation.simulation.fills["strategy"].eq("flat stake")
    ].iloc[0]

    assert flat_bet["filled_stake_usd"] == pytest.approx(10.0)
    assert flat_fill["fill_cost_usd"] == pytest.approx(
        flat_fill["fill_raw_cost_usd"] + flat_fill["fill_fee_usd"]
    )
    assert flat_bet["profit_usd"] == pytest.approx(10.2)
    metric = evaluation.metrics.loc[evaluation.metrics["strategy"].eq("flat stake")].iloc[0]
    assert metric["stake_roi_pct"] == pytest.approx(102.0)
    assert metric["final_bankroll_usd"] == pytest.approx(110.2)
    monthly = evaluation.monthly_returns.loc[
        evaluation.monthly_returns["strategy"].eq("flat stake")
    ].iloc[0]
    assert monthly["filled_stake_usd"] == pytest.approx(10.0)
    assert monthly["stake_roi_pct"] == pytest.approx(102.0)


def test_overlapping_positions_respect_cash_and_open_exposure_before_settlement() -> None:
    config = ExecutionConfig(
        initial_capital_usd=100.0,
        flat_stake_usd=80.0,
        maximum_trade_participation=1.0,
        maximum_open_exposure_fraction=1.0,
    )
    second_trade = {
        **collected_trade(
            tx_hash="0xsecond",
            shares=1_000.0,
            result=0.0,
        ),
        "home_team": "C",
        "away_team": "D",
        "condition_id": "0xdef",
        "asset_id": "asset-2",
        "dune_trade_time_utc": "2026-09-05T11:00:00Z",
        "kickoff_uk_time": "2026-09-05T18:00:00+01:00",
    }
    trades = pd.DataFrame(
        [
            collected_trade(tx_hash="0xfirst", shares=1_000.0),
            second_trade,
        ]
    )
    prepared, _ = prepare_execution_trades(trades, config=config)
    evaluation = evaluate_execution_trades(prepared, config=config)

    flat_fills = evaluation.simulation.fills.loc[
        evaluation.simulation.fills["strategy"].eq("flat stake")
    ].sort_values("fill_time_utc")
    flat_path = evaluation.simulation.bankroll_paths.loc[
        evaluation.simulation.bankroll_paths["strategy"].eq("flat stake")
    ]

    assert flat_fills["fill_cost_usd"].tolist() == pytest.approx([80.0, 20.0])
    assert flat_fills["fill_cost_usd"].sum() <= config.initial_capital_usd
    assert flat_path["cash_usd"].ge(-1e-12).all()
    assert flat_path["open_exposure_usd"].le(
        config.maximum_open_exposure_fraction * flat_path["bankroll_usd"] + 1e-12
    ).all()


def test_period_stake_roi_is_nan_when_no_stake_is_filled() -> None:
    paths = pd.DataFrame(
        [
            {"strategy": "flat stake", "event_time_utc": "2026-08-31T12:00:00Z", "bankroll_usd": 100.0},
            {"strategy": "flat stake", "event_time_utc": "2026-09-01T12:00:00Z", "bankroll_usd": 100.0},
        ]
    )
    summary = period_return_summary(
        paths,
        pd.DataFrame(columns=["strategy", "settlement_time_utc", "profit_usd", "filled_stake_usd"]),
        frequency="M",
        period_column="month",
        initial_capital_usd=100.0,
        timezone="Europe/London",
    )

    assert summary["filled_stake_usd"].eq(0.0).all()
    assert summary["stake_roi_pct"].isna().all()


def test_common_sample_statistics_are_removed_only_when_verified_equal() -> None:
    metrics = pd.DataFrame(
        [
            {"strategy": "flat stake", "bets": 3, "wins": 2, "hit_rate_pct": 2 / 3 * 100, "profit_usd": 1.0},
            {"strategy": "10% Kelly", "bets": 3, "wins": 2, "hit_rate_pct": 2 / 3 * 100, "profit_usd": 2.0},
        ]
    )
    common, strategy_metrics = separate_common_strategy_statistics(metrics)

    assert common.loc[0, "bets"] == 3
    assert common.loc[0, "wins"] == 2
    assert common.loc[0, "hit_rate_pct"] == pytest.approx(2 / 3 * 100)
    assert not {"bets", "wins", "hit_rate_pct"}.intersection(strategy_metrics.columns)

    differing = metrics.copy()
    differing.loc[differing["strategy"].eq("10% Kelly"), "bets"] = 2
    common, strategy_metrics = separate_common_strategy_statistics(differing)

    assert common.empty
    assert {"bets", "wins", "hit_rate_pct"}.issubset(strategy_metrics.columns)


def test_trade_participation_can_create_a_partial_fill() -> None:
    config = ExecutionConfig(
        initial_capital_usd=100.0,
        flat_stake_usd=50.0,
        maximum_trade_participation=0.01,
        maximum_open_exposure_fraction=1.0,
    )
    prepared, _ = prepare_execution_trades(
        pd.DataFrame([collected_trade(shares=10.0)]), config=config
    )
    evaluation = evaluate_execution_trades(prepared, config=config)
    flat_bet = evaluation.simulation.bets.loc[
        evaluation.simulation.bets["strategy"].eq("flat stake")
    ].iloc[0]

    assert 0 < flat_bet["filled_stake_usd"] < 50.0
    assert 0 < flat_bet["fill_fraction"] < 1.0


def test_loader_discovers_all_months_and_ignores_unrelated_cache_names(
    tmp_path: Path,
) -> None:
    suffix = "market_maximum_home_win"
    august = tmp_path / f"la_liga_26_27_2026_08_{suffix}_eligible_trades.csv"
    september = tmp_path / f"serie_a_26_27_2026_09_{suffix}_eligible_trades.csv"
    pd.DataFrame([collected_trade()]).to_csv(august, index=False)
    pd.DataFrame(columns=pd.DataFrame([collected_trade()]).columns).to_csv(
        september, index=False
    )
    pd.DataFrame([collected_trade()]).to_csv(
        tmp_path / f"la_liga_26_27_2026_08_{suffix}_accepted_trades.csv",
        index=False,
    )
    for league, month in [("la_liga", "2026_08"), ("serie_a", "2026_09")]:
        summary = pd.DataFrame(
            [
                {
                    "match_date": f"2026-{month[-2:]}-05",
                    "season": "26_27",
                    "league": "La Liga" if league == "la_liga" else "Serie A",
                    "home_team": "A" if league == "la_liga" else "C",
                    "away_team": "B" if league == "la_liga" else "D",
                }
            ]
        )
        summary.to_csv(
            tmp_path / f"{league}_26_27_{month}_{suffix}_match_summary.csv",
            index=False,
        )

    cache = load_finalized_holdout_cache(tmp_path, season="26_27")

    assert len(cache.eligible_trade_files) == 2
    assert len(cache.match_summary_files) == 2
    assert len(cache.eligible_trades) == 1
    assert len(cache.match_summary) == 2


def test_empty_holdout_inputs_are_handled_without_failure(tmp_path: Path) -> None:
    cache = load_finalized_holdout_cache(tmp_path, season="26_27")
    assert cache.eligible_trades.empty
    assert cache.match_summary.empty

    evaluation = evaluate_execution_by_league(
        pd.DataFrame(), config=ExecutionConfig()
    )
    assert evaluation.metrics.empty


def test_league_results_use_independent_initial_capital_sleeves() -> None:
    config = ExecutionConfig(initial_capital_usd=100.0, flat_stake_usd=10.0)
    trades = pd.DataFrame(
        [
            collected_trade(tx_hash="0xla", league="La Liga"),
            {
                **collected_trade(tx_hash="0xit", league="Serie A"),
                "home_team": "C",
                "away_team": "D",
                "condition_id": "0xdef",
            },
        ]
    )
    prepared, _ = prepare_execution_trades(trades, config=config)

    league_evaluation = evaluate_execution_by_league(prepared, config=config)
    flat = league_evaluation.metrics.loc[
        league_evaluation.metrics["strategy"].eq("flat stake")
    ]

    assert set(flat["analysis_league"]) == {"La Liga", "Serie A"}
    np.testing.assert_allclose(flat["final_bankroll_usd"], 110.2)
