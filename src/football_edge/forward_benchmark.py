"""Football-Data reference-price benchmark for frozen Football-Data ``MaxH`` signals.

This module evaluates the current-season matches selected by the already-frozen
market-maximum home-win rule.  Every qualifying, settled match is assumed to be
fully executable at the recorded Football-Data ``MaxH``.  It deliberately does
not model venue discovery, fees, liquidity, partial fills, or open exposure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from football_edge.forward_execution import maximum_drawdown_pct
from football_edge.forward_polymarket import CurrentSeasonFixtureSelection
from football_edge.staking import full_kelly_fraction


@dataclass(frozen=True)
class BenchmarkConfig:
    """Simple staking and reporting assumptions for the Football-Data reference-price benchmark."""

    initial_capital_usd: float = 100_000.0
    flat_stake_usd: float = 1_000.0
    kelly_fractions: tuple[float, ...] = (0.10, 0.25)
    maximum_kelly_stake_fraction: float = 0.05
    assumed_match_duration_hours: float = 2.0
    reporting_timezone: str = "Europe/London"
    weekly_frequency: str = "W-MON"
    monthly_frequency: str = "M"
    weekly_risk_free_rate: float = 0.0

    def __post_init__(self) -> None:
        if self.initial_capital_usd <= 0:
            raise ValueError("initial_capital_usd must be positive")
        if self.flat_stake_usd <= 0:
            raise ValueError("flat_stake_usd must be positive")
        if not self.kelly_fractions or any(not 0 < value <= 1 for value in self.kelly_fractions):
            raise ValueError("kelly_fractions must contain values in (0, 1]")
        if not 0 < self.maximum_kelly_stake_fraction <= 1:
            raise ValueError("maximum_kelly_stake_fraction must be in (0, 1]")
        if self.assumed_match_duration_hours <= 0:
            raise ValueError("assumed_match_duration_hours must be positive")


@dataclass(frozen=True)
class BenchmarkDataset:
    """Coverage audit and settled bets prepared from the frozen selection."""

    audit: pd.DataFrame
    coverage: pd.DataFrame
    bets: pd.DataFrame


@dataclass(frozen=True)
class BenchmarkSimulation:
    """Settled bet records and chronological bankroll paths."""

    bets: pd.DataFrame
    bankroll_paths: pd.DataFrame


@dataclass(frozen=True)
class BenchmarkEvaluation:
    """Simulation and weekly, monthly, and full-sample summaries."""

    simulation: BenchmarkSimulation
    weekly_returns: pd.DataFrame
    monthly_returns: pd.DataFrame
    metrics: pd.DataFrame


def _strategy_specifications(config: BenchmarkConfig) -> tuple[tuple[str, float | None], ...]:
    return (
        ("flat stake", None),
        *tuple((f"{fraction:.0%} Kelly", fraction) for fraction in config.kelly_fractions),
    )


def prepare_benchmark_dataset(
    selection: CurrentSeasonFixtureSelection,
    *,
    analysis_end_date: str | pd.Timestamp | None = None,
) -> BenchmarkDataset:
    """Create a coverage audit and settled bet set through an inclusive cutoff."""
    fixtures = selection.observed_fixtures.copy()
    if analysis_end_date is not None:
        cutoff = pd.to_datetime(analysis_end_date, errors="raise")
        if cutoff.tzinfo is not None:
            raise ValueError("analysis_end_date must be a timezone-naive calendar date")
        cutoff = cutoff.normalize()
        match_dates = pd.to_datetime(fixtures["date"], errors="coerce").dt.normalize()
        fixtures = fixtures.loc[match_dates.le(cutoff)].copy()
    if fixtures.empty:
        return BenchmarkDataset(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    fixtures["completed_match"] = fixtures["home_win_result"].notna()
    fixtures["valid_maxh"] = fixtures["football_data_max_home_odds"].gt(1)
    fixtures["qualifies_frozen_rule"] = fixtures["accepted_fd"].fillna(False).astype(bool)
    fixtures["included_benchmark_bet"] = (
        fixtures["completed_match"] & fixtures["qualifies_frozen_rule"]
    )

    coverage = (
        fixtures.groupby("league", sort=True, dropna=False)
        .agg(
            matches_loaded=("league", "size"),
            completed_matches=("completed_match", "sum"),
            valid_maxh_matches=("valid_maxh", "sum"),
            qualifying_matches=("qualifies_frozen_rule", "sum"),
            settled_qualifying_bets=("included_benchmark_bet", "sum"),
        )
        .reset_index()
    )
    coverage["missing_or_invalid_maxh"] = (
        coverage["matches_loaded"] - coverage["valid_maxh_matches"]
    )
    coverage["unresolved_matches"] = (
        coverage["matches_loaded"] - coverage["completed_matches"]
    )
    coverage["unresolved_qualifying_matches"] = (
        coverage["qualifying_matches"] - coverage["settled_qualifying_bets"]
    )
    total = {
        "league": "All frozen leagues",
        **{
            column: int(coverage[column].sum())
            for column in coverage.columns
            if column != "league"
        },
    }
    coverage = pd.concat([pd.DataFrame([total]), coverage], ignore_index=True)

    audit_columns = [
        "match_date",
        "league",
        "home_team",
        "away_team",
        "football_data_max_home_odds",
        "p_fd",
        "g_hat_fd",
        "q_hat_fd",
        "accepted_probability_min",
        "accepted_probability_max",
        "home_goals",
        "away_goals",
        "home_win_result",
        "completed_match",
        "valid_maxh",
        "qualifies_frozen_rule",
        "included_benchmark_bet",
    ]
    audit = fixtures.loc[:, audit_columns].copy()

    bets = fixtures.loc[fixtures["included_benchmark_bet"]].copy()
    if not bets.empty:
        bets["benchmark_odds"] = bets["football_data_max_home_odds"].astype(float)
        bets["full_kelly_fraction"] = full_kelly_fraction(
            bets["q_hat_fd"], bets["benchmark_odds"]
        )
        bets["profit_per_unit_stake"] = np.where(
            bets["home_win_result"].eq(1.0), bets["benchmark_odds"] - 1.0, -1.0
        )
        bets["order_time_utc"] = pd.to_datetime(
            bets["football_data_release_utc"], utc=True
        )
        bets["settlement_time_utc"] = pd.to_datetime(
            bets["kickoff_utc"], utc=True
        ) + pd.to_timedelta(2.0, unit="h")
        bets = bets.sort_values(
            ["order_time_utc", "kickoff_utc", "league", "home_team", "away_team"],
            kind="stable",
        ).reset_index(drop=True)
        bets.insert(0, "benchmark_bet_id", np.arange(len(bets), dtype=int))
    return BenchmarkDataset(audit=audit, coverage=coverage, bets=bets)


def _simulate_strategy(
    bets: pd.DataFrame,
    *,
    strategy: str,
    kelly_fraction: float | None,
    config: BenchmarkConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Size at release time and recognize P&L only at settlement time."""
    if bets.empty:
        return pd.DataFrame(), pd.DataFrame()

    working = bets.copy()
    working["settlement_time_utc"] = pd.to_datetime(
        working["kickoff_utc"], utc=True
    ) + pd.to_timedelta(config.assumed_match_duration_hours, unit="h")
    events: list[tuple[pd.Timestamp, int, int, str]] = []
    for row in working.itertuples(index=False):
        bet_id = int(row.benchmark_bet_id)
        # Settlement precedes an order at an identical timestamp because its
        # outcome would then already be known when the new stake is submitted.
        events.append((pd.Timestamp(row.settlement_time_utc), 0, bet_id, "settlement"))
        events.append((pd.Timestamp(row.order_time_utc), 1, bet_id, "order"))
    events.sort(key=lambda item: (item[0], item[1], item[2]))

    lookup = working.set_index("benchmark_bet_id", drop=False)
    bankroll = float(config.initial_capital_usd)
    positions: dict[int, dict[str, float]] = {}
    settled_rows: list[dict[str, Any]] = []
    path_rows = [
        {
            "strategy": strategy,
            "event_time_utc": events[0][0],
            "event_type": "initial",
            "benchmark_bet_id": np.nan,
            "bankroll_usd": bankroll,
        }
    ]

    for event_time, _, bet_id, event_type in events:
        row = lookup.loc[bet_id]
        if event_type == "order":
            if kelly_fraction is None:
                target_stake = float(config.flat_stake_usd)
                applied_fraction = np.nan
            else:
                applied_fraction = min(
                    float(kelly_fraction) * float(row["full_kelly_fraction"]),
                    config.maximum_kelly_stake_fraction,
                )
                target_stake = bankroll * applied_fraction
            positions[bet_id] = {
                "stake_usd": float(target_stake),
                "bankroll_at_order_usd": bankroll,
                "applied_bankroll_fraction": applied_fraction,
            }
            continue

        position = positions.pop(bet_id)
        stake = position["stake_usd"]
        profit = stake * float(row["profit_per_unit_stake"])
        bankroll += profit
        settled = row.to_dict()
        settled.update(
            {
                "strategy": strategy,
                "total_stake_usd": stake,
                "profit_usd": profit,
                "bankroll_at_order_usd": position["bankroll_at_order_usd"],
                "applied_bankroll_fraction": position["applied_bankroll_fraction"],
                "bankroll_after_settlement_usd": bankroll,
            }
        )
        settled_rows.append(settled)
        path_rows.append(
            {
                "strategy": strategy,
                "event_time_utc": event_time,
                "event_type": "settlement",
                "benchmark_bet_id": bet_id,
                "bankroll_usd": bankroll,
            }
        )

    return pd.DataFrame(settled_rows), pd.DataFrame(path_rows)


def run_benchmark_strategy_set(
    bets: pd.DataFrame,
    *,
    config: BenchmarkConfig,
) -> BenchmarkSimulation:
    """Run flat stake and configured Kelly fractions on one common bet set."""
    bet_frames: list[pd.DataFrame] = []
    path_frames: list[pd.DataFrame] = []
    for strategy, fraction in _strategy_specifications(config):
        strategy_bets, strategy_path = _simulate_strategy(
            bets,
            strategy=strategy,
            kelly_fraction=fraction,
            config=config,
        )
        if not strategy_bets.empty:
            bet_frames.append(strategy_bets)
            path_frames.append(strategy_path)
    return BenchmarkSimulation(
        bets=pd.concat(bet_frames, ignore_index=True, sort=False) if bet_frames else pd.DataFrame(),
        bankroll_paths=(
            pd.concat(path_frames, ignore_index=True, sort=False)
            if path_frames
            else pd.DataFrame()
        ),
    )


def _local_period(
    frame: pd.DataFrame,
    column: str,
    *,
    frequency: str,
    timezone: str,
) -> pd.Series:
    timestamps = pd.to_datetime(frame[column], utc=True).dt.tz_convert(timezone)
    return timestamps.dt.tz_localize(None).dt.to_period(frequency)


def benchmark_period_summary(
    paths: pd.DataFrame,
    bets: pd.DataFrame,
    *,
    frequency: str,
    period_column: str,
    initial_capital_usd: float,
    timezone: str,
) -> pd.DataFrame:
    """Compute non-overlapping bankroll returns, Stake ROI, and bet counts."""
    if paths.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for strategy, strategy_path in paths.groupby("strategy", sort=True):
        strategy_path = strategy_path.sort_values("event_time_utc", kind="stable").copy()
        strategy_path["period"] = _local_period(
            strategy_path, "event_time_utc", frequency=frequency, timezone=timezone
        )
        periods = pd.period_range(
            strategy_path["period"].min(), strategy_path["period"].max(), freq=frequency
        )
        end_bankroll = (
            strategy_path.groupby("period", sort=True)["bankroll_usd"]
            .last()
            .reindex(periods)
            .ffill()
        )
        start_bankroll = end_bankroll.shift(1).fillna(initial_capital_usd)
        period_bets = bets.loc[bets["strategy"].eq(strategy)].copy()
        period_bets["period"] = _local_period(
            period_bets, "settlement_time_utc", frequency=frequency, timezone=timezone
        )
        counts = period_bets.groupby("period").size().reindex(periods, fill_value=0)
        stakes = period_bets.groupby("period")["total_stake_usd"].sum().reindex(
            periods, fill_value=0.0
        )
        profits = period_bets.groupby("period")["profit_usd"].sum().reindex(
            periods, fill_value=0.0
        )
        for period in periods:
            stake = float(stakes.loc[period])
            profit = float(profits.loc[period])
            rows.append(
                {
                    "strategy": strategy,
                    period_column: str(period),
                    "bets": int(counts.loc[period]),
                    "total_stake_usd": stake,
                    "profit_usd": profit,
                    "stake_roi_pct": profit / stake * 100.0 if stake > 0 else np.nan,
                    "start_bankroll_usd": float(start_bankroll.loc[period]),
                    "end_bankroll_usd": float(end_bankroll.loc[period]),
                    "bankroll_return_pct": (
                        float(end_bankroll.loc[period] / start_bankroll.loc[period] - 1.0)
                        * 100.0
                    ),
                }
            )
    return pd.DataFrame(rows)


def summarize_benchmark_metrics(
    paths: pd.DataFrame,
    bets: pd.DataFrame,
    weekly_returns: pd.DataFrame,
    *,
    config: BenchmarkConfig,
) -> pd.DataFrame:
    """Summarize full-sample economics and weekly risk by strategy."""
    if paths.empty or bets.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for strategy, strategy_path in paths.groupby("strategy", sort=True):
        strategy_path = strategy_path.sort_values("event_time_utc", kind="stable")
        strategy_bets = bets.loc[bets["strategy"].eq(strategy)]
        weekly = (
            weekly_returns.loc[
                weekly_returns["strategy"].eq(strategy), "bankroll_return_pct"
            ].astype(float)
            / 100.0
        )
        weekly_mean = weekly.mean()
        weekly_volatility = weekly.std(ddof=1)
        annualized_volatility = (
            weekly_volatility * np.sqrt(52) if pd.notna(weekly_volatility) else np.nan
        )
        annualized_sharpe = (
            (weekly_mean - config.weekly_risk_free_rate) / weekly_volatility * np.sqrt(52)
            if pd.notna(weekly_volatility) and weekly_volatility > 0
            else np.nan
        )
        final_bankroll = float(strategy_path["bankroll_usd"].iloc[-1])
        total_stake = float(strategy_bets["total_stake_usd"].sum())
        profit = float(strategy_bets["profit_usd"].sum())
        rows.append(
            {
                "strategy": strategy,
                "bets": len(strategy_bets),
                "wins": int(strategy_bets["home_win_result"].sum()),
                "hit_rate_pct": strategy_bets["home_win_result"].mean() * 100.0,
                "cumulative_return_pct": (
                    final_bankroll / config.initial_capital_usd - 1.0
                ) * 100.0,
                "stake_roi_pct": profit / total_stake * 100.0 if total_stake > 0 else np.nan,
                "max_drawdown_pct": maximum_drawdown_pct(strategy_path["bankroll_usd"]),
                "annualized_weekly_sharpe": annualized_sharpe,
                "annualized_weekly_volatility_pct": annualized_volatility * 100.0,
                "mean_weekly_return_pct": weekly_mean * 100.0,
                "weeks_observed": len(weekly),
                "total_stake_usd": total_stake,
                "profit_usd": profit,
                "final_bankroll_usd": final_bankroll,
            }
        )
    return pd.DataFrame(rows)


def evaluate_benchmark_bets(
    bets: pd.DataFrame,
    *,
    config: BenchmarkConfig,
) -> BenchmarkEvaluation:
    """Run all benchmark strategies and construct their return summaries."""
    simulation = run_benchmark_strategy_set(bets, config=config)
    weekly = benchmark_period_summary(
        simulation.bankroll_paths,
        simulation.bets,
        frequency=config.weekly_frequency,
        period_column="week",
        initial_capital_usd=config.initial_capital_usd,
        timezone=config.reporting_timezone,
    )
    monthly = benchmark_period_summary(
        simulation.bankroll_paths,
        simulation.bets,
        frequency=config.monthly_frequency,
        period_column="month",
        initial_capital_usd=config.initial_capital_usd,
        timezone=config.reporting_timezone,
    )
    metrics = summarize_benchmark_metrics(
        simulation.bankroll_paths, simulation.bets, weekly, config=config
    )
    return BenchmarkEvaluation(simulation, weekly, monthly, metrics)


def evaluate_benchmark_by_league(
    bets: pd.DataFrame,
    *,
    config: BenchmarkConfig,
) -> BenchmarkEvaluation:
    """Evaluate every league as an independent full-capital sleeve."""
    if bets.empty or "league" not in bets:
        empty = pd.DataFrame()
        return BenchmarkEvaluation(BenchmarkSimulation(empty, empty), empty, empty, empty)

    metric_frames: list[pd.DataFrame] = []
    bet_frames: list[pd.DataFrame] = []
    path_frames: list[pd.DataFrame] = []
    weekly_frames: list[pd.DataFrame] = []
    monthly_frames: list[pd.DataFrame] = []
    for league, league_bets in bets.groupby("league", sort=True):
        evaluation = evaluate_benchmark_bets(league_bets.copy(), config=config)
        for frame, destination in (
            (evaluation.metrics, metric_frames),
            (evaluation.simulation.bets, bet_frames),
            (evaluation.simulation.bankroll_paths, path_frames),
            (evaluation.weekly_returns, weekly_frames),
            (evaluation.monthly_returns, monthly_frames),
        ):
            if not frame.empty:
                annotated = frame.copy()
                annotated.insert(0, "analysis_league", str(league))
                destination.append(annotated)

    def concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    return BenchmarkEvaluation(
        simulation=BenchmarkSimulation(concat(bet_frames), concat(path_frames)),
        weekly_returns=concat(weekly_frames),
        monthly_returns=concat(monthly_frames),
        metrics=concat(metric_frames),
    )
