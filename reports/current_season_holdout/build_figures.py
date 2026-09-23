"""Build the monthly Polymarket figures for the forward-holdout report.

The script delegates execution, staking, chronology, and return calculations to
the same tested helpers used by public notebook 04. It changes presentation
only: monthly bankroll return and Stake ROI are plotted for the three staking
specifications, with executed-bet counts shown alongside bankroll return.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from football_edge.forward_execution import (  # noqa: E402
    ExecutionConfig,
    evaluate_execution_trades,
    load_finalized_holdout_cache,
    prepare_execution_trades,
)
CURRENT_SEASON = "26_27"
HOLDOUT_CACHE_DIR = PROJECT_ROOT / "data" / "holdout" / "dune_cache"
FIGURE_DIR = Path(__file__).resolve().parent / "figures"

STRATEGIES = ("flat stake", "10% Kelly", "25% Kelly")
COLORS = {
    "flat stake": "#F58518",
    "10% Kelly": "#4C78A8",
    "25% Kelly": "#54A24B",
}
MARKERS = {"flat stake": "s", "10% Kelly": "o", "25% Kelly": "^"}

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


def load_report_evaluation():
    """Reproduce notebook 04's portfolio evaluation without new Dune queries."""
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
    return evaluate_execution_trades(
        execution_trades,
        config=EXECUTION_CONFIG,
    )


def plot_monthly_metric(
    execution,
    *,
    value_column: str,
    title: str,
    y_label: str,
    show_bet_counts: bool,
) -> plt.Figure:
    """Plot one monthly execution metric for the configured strategies."""
    monthly = execution.monthly_returns.loc[
        execution.monthly_returns["strategy"].isin(STRATEGIES)
    ].copy()
    if monthly.empty:
        raise ValueError("No monthly execution results are available")

    month_labels = sorted(monthly["month"].astype(str).unique())
    x_positions = np.arange(len(month_labels))
    figure, axis = plt.subplots(figsize=(11, 5.8), constrained_layout=True)

    for strategy in STRATEGIES:
        strategy_values = (
            monthly.loc[monthly["strategy"].eq(strategy)]
            .set_index("month")[value_column]
            .reindex(month_labels)
        )
        axis.plot(
            x_positions,
            strategy_values,
            color=COLORS[strategy],
            marker=MARKERS[strategy],
            linewidth=2.2,
            markersize=7,
            label=strategy,
        )

    axis.axhline(0.0, color="black", linewidth=0.9)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(month_labels)
    axis.set_xlabel("Evaluation month")
    axis.set_ylabel(y_label)
    axis.set_title(title)
    axis.grid(True, axis="y", alpha=0.3)

    handles, labels = axis.get_legend_handles_labels()
    if show_bet_counts:
        count_axis = axis.twinx()
        count_table = monthly.pivot(
            index="month", columns="strategy", values="bets"
        ).reindex(month_labels)
        common_counts = count_table.nunique(axis=1, dropna=False).le(1).all()
        if common_counts:
            count_axis.bar(
                x_positions,
                count_table.iloc[:, 0].fillna(0),
                width=0.55,
                color="lightgray",
                alpha=0.35,
                label="executed bets",
            )
        else:
            for strategy in STRATEGIES:
                if strategy not in count_table:
                    continue
                count_axis.plot(
                    x_positions,
                    count_table[strategy].fillna(0),
                    color=COLORS[strategy],
                    marker=MARKERS[strategy],
                    linestyle=":",
                    alpha=0.55,
                    label=f"{strategy} bets",
                )
        count_axis.set_ylabel("Executed bets")
        count_axis.set_ylim(bottom=0)
        bet_handles, bet_labels = count_axis.get_legend_handles_labels()
        handles += bet_handles
        labels += bet_labels

    axis.legend(handles, labels, loc="best")
    return figure


def main() -> None:
    """Build the report figures from the current finalized local inputs."""
    execution = load_report_evaluation()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    bankroll_return = plot_monthly_metric(
        execution,
        value_column="bankroll_return_pct",
        title="Polymarket monthly bankroll return",
        y_label="Monthly bankroll return (%)",
        show_bet_counts=True,
    )
    bankroll_return.savefig(
        FIGURE_DIR / "polymarket_monthly_bankroll_return.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(bankroll_return)

    stake_roi = plot_monthly_metric(
        execution,
        value_column="stake_roi_pct",
        title="Polymarket monthly Stake ROI",
        y_label="Monthly Stake ROI (%)",
        show_bet_counts=False,
    )
    stake_roi.savefig(
        FIGURE_DIR / "polymarket_monthly_stake_roi.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(stake_roi)


if __name__ == "__main__":
    main()
