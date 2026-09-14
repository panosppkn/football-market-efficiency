"""Build the two public figures for the current-season holdout report.

The script delegates selection, staking, chronology, and return calculations to
the same tested helpers used by public notebooks 02 and 04.  It only changes the
presentation: cumulative bankroll return is expressed as a percentage, and
full-period Stake ROI is shown as a compact comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from football_edge.forward_benchmark import (  # noqa: E402
    BenchmarkConfig,
    evaluate_benchmark_bets,
    prepare_benchmark_dataset,
)
from football_edge.forward_execution import (  # noqa: E402
    ExecutionConfig,
    evaluate_execution_trades,
    load_finalized_holdout_cache,
    prepare_execution_trades,
)
from football_edge.forward_polymarket import (  # noqa: E402
    prepare_current_season_fixture_selection,
)


CURRENT_SEASON = "26_27"
ANALYSIS_END_DATE = "2026-08-31"
CURRENT_SEASON_FILE = (
    PROJECT_ROOT / "data" / "holdout" / "all-euro-data-2026-2027.xlsx"
)
FROZEN_PARAMETERS_FILE = (
    PROJECT_ROOT
    / "docs"
    / "parameter_snapshots"
    / "market_maximum_home_win_asof_2025_26.csv"
)
HOLDOUT_CACHE_DIR = PROJECT_ROOT / "data" / "holdout" / "dune_cache"
FIGURE_DIR = Path(__file__).resolve().parent / "figures"

STRATEGIES = ("flat stake", "10% Kelly", "25% Kelly")
COLORS = {
    "flat stake": "#F58518",
    "10% Kelly": "#4C78A8",
    "25% Kelly": "#54A24B",
}
MARKERS = {"flat stake": "s", "10% Kelly": "o", "25% Kelly": "^"}

BENCHMARK_CONFIG = BenchmarkConfig(
    initial_capital_usd=100_000.0,
    flat_stake_usd=1_000.0,
    kelly_fractions=(0.10, 0.25),
    maximum_kelly_stake_fraction=0.05,
    assumed_match_duration_hours=2.0,
    reporting_timezone="Europe/London",
    weekly_frequency="W-MON",
    monthly_frequency="M",
    weekly_risk_free_rate=0.0,
)

EXECUTION_CONFIG = ExecutionConfig(
    initial_capital_usd=100_000.0,
    flat_stake_usd=1_000.0,
    kelly_fractions=(0.10, 0.25),
    buyer_fee_rate=0.05,
    minimum_odds_premium=0.01,
    allow_partial_fills=True,
    minimum_fill_fraction=0.0,
    maximum_trade_participation=0.50,
    maximum_kelly_stake_fraction=0.05,
    maximum_open_exposure_fraction=1.0,
    assumed_match_duration_hours=2.0,
    reporting_timezone="Europe/London",
    weekly_frequency="W-MON",
    monthly_frequency="M",
    weekly_risk_free_rate=0.0,
)


def load_report_evaluations():
    """Reproduce the saved notebook 02 and 04 evaluations without new queries."""
    selection = prepare_current_season_fixture_selection(
        CURRENT_SEASON_FILE,
        FROZEN_PARAMETERS_FILE,
        season=CURRENT_SEASON,
        market_outcome="home_win",
        source="market_maximum",
    )
    benchmark_data = prepare_benchmark_dataset(
        selection,
        analysis_end_date=ANALYSIS_END_DATE,
    )
    benchmark = evaluate_benchmark_bets(
        benchmark_data.bets,
        config=BENCHMARK_CONFIG,
    )

    holdout_cache = load_finalized_holdout_cache(
        HOLDOUT_CACHE_DIR,
        season=CURRENT_SEASON,
        source="market_maximum",
        market_outcome="home_win",
    )
    execution_trades, _ = prepare_execution_trades(
        holdout_cache.eligible_trades,
        config=EXECUTION_CONFIG,
    )
    execution = evaluate_execution_trades(
        execution_trades,
        config=EXECUTION_CONFIG,
    )
    return benchmark, execution


def plot_cumulative_bankroll_return(benchmark, execution) -> plt.Figure:
    """Plot benchmark and execution-aware cumulative returns in separate panels."""
    figure, axes = plt.subplots(
        2,
        1,
        figsize=(11, 8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    panels = (
        (
            axes[0],
            benchmark.simulation.bankroll_paths,
            BENCHMARK_CONFIG.initial_capital_usd,
            "Football-Data reference-price benchmark",
        ),
        (
            axes[1],
            execution.simulation.bankroll_paths,
            EXECUTION_CONFIG.initial_capital_usd,
            "Polymarket transaction-based execution",
        ),
    )
    for axis, paths, initial_capital, title in panels:
        for strategy in STRATEGIES:
            strategy_path = paths.loc[paths["strategy"].eq(strategy)].sort_values(
                "event_time_utc", kind="stable"
            )
            if strategy_path.empty:
                continue
            event_time = pd.to_datetime(strategy_path["event_time_utc"], utc=True)
            cumulative_return = (
                strategy_path["bankroll_usd"].astype(float) / initial_capital - 1.0
            ) * 100.0
            axis.plot(
                event_time,
                cumulative_return,
                color=COLORS[strategy],
                marker=MARKERS[strategy],
                linewidth=2,
                markersize=4,
                drawstyle="steps-post",
                label=strategy,
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_ylabel("Cumulative bankroll return (%)")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")
    axes[-1].set_xlabel("Settlement timeline")
    figure.suptitle("Frozen 2026/27 forward evaluation through 31 August 2026", fontsize=14)
    return figure


def plot_stake_roi(benchmark, execution) -> plt.Figure:
    """Plot full-period Stake ROI for the two complementary evaluations."""
    benchmark_roi = benchmark.metrics.set_index("strategy")["stake_roi_pct"]
    execution_roi = execution.metrics.set_index("strategy")["stake_roi_pct"]
    x_positions = np.arange(len(STRATEGIES))
    width = 0.36

    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    benchmark_bars = axis.bar(
        x_positions - width / 2,
        [benchmark_roi.get(strategy, np.nan) for strategy in STRATEGIES],
        width,
        color="#4C78A8",
        label="Football-Data reference price",
    )
    execution_bars = axis.bar(
        x_positions + width / 2,
        [execution_roi.get(strategy, np.nan) for strategy in STRATEGIES],
        width,
        color="#E45756",
        label="Polymarket execution-aware",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(STRATEGIES)
    axis.set_ylabel("Stake ROI (%)")
    axis.set_title("Current-period profitability per dollar wagered")
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend(loc="best")
    axis.bar_label(benchmark_bars, fmt="%.2f%%", padding=3)
    axis.bar_label(execution_bars, fmt="%.2f%%", padding=3)
    return figure


def main() -> None:
    """Build the report figures from the current finalized local inputs."""
    benchmark, execution = load_report_evaluations()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    cumulative = plot_cumulative_bankroll_return(benchmark, execution)
    cumulative.savefig(
        FIGURE_DIR / "cumulative_bankroll_return.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(cumulative)

    stake_roi = plot_stake_roi(benchmark, execution)
    stake_roi.savefig(
        FIGURE_DIR / "stake_roi_comparison.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(stake_roi)


if __name__ == "__main__":
    main()
