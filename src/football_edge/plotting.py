"""Reusable presentation plots for the football edge analysis."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from football_edge.config import EV_BUCKET_LABELS

SOURCE_LABELS = {
    "bet365": "Bet365",
    "pinnacle": "Pinnacle",
    "betfair_exchange": "Betfair Exchange",
    "average_preclosing": "Market average",
    "best_preclosing": "Market maximum",
    "best_closing": "Closing market maximum",
}

SOURCE_COLORS = {
    "Bet365": "#2CA02C",
    "Pinnacle": "#1F77B4",
    "Betfair Exchange": "#FF7F0E",
    "Market average": "#9467BD",
    "Market maximum": "#D62728",
    "Closing market maximum": "#8C564B",
}

COEFFICIENT_LABELS = {
    "market_logit": "Market log-odds",
    "home_season_avg_goals": "Home season scoring",
    "away_season_avg_goals": "Away season scoring",
    "home_last_5_avg_goals": "Home trailing-five scoring",
    "away_last_5_avg_goals": "Away trailing-five scoring",
}

COEFFICIENT_COLORS = {
    "market_logit": "#111111",
    "home_season_avg_goals": "#1F77B4",
    "away_season_avg_goals": "#17BECF",
    "home_last_5_avg_goals": "#D62728",
    "away_last_5_avg_goals": "#FF7F0E",
}


def plot_overall_roi_by_execution(
    performance: pd.DataFrame,
    *,
    title: str = "Out-of-sample ROI across all test seasons",
) -> tuple[plt.Figure, plt.Axes]:
    """Plot full-period ROI and its approximate 95% interval by execution source."""
    required = {
        "execution_source",
        "roi_pct",
        "roi_95_low_pct",
        "roi_95_high_pct",
        "bets",
    }
    missing = required.difference(performance.columns)
    if missing:
        raise ValueError(
            "Overall ROI plot is missing columns: " + ", ".join(sorted(missing))
        )
    if performance.empty:
        raise ValueError("Overall ROI plot data cannot be empty")

    source_order = [
        source
        for source in SOURCE_LABELS
        if source in set(performance["execution_source"])
    ]
    plot_data = (
        performance.set_index("execution_source")
        .reindex(source_order)
        .reset_index()
    )
    labels = plot_data["execution_source"].map(SOURCE_LABELS)
    lower_error = plot_data["roi_pct"] - plot_data["roi_95_low_pct"]
    upper_error = plot_data["roi_95_high_pct"] - plot_data["roi_pct"]

    figure, axis = plt.subplots(figsize=(11, 5))
    bars = axis.bar(
        labels,
        plot_data["roi_pct"],
        yerr=np.vstack([lower_error, upper_error]),
        capsize=4,
        color=[SOURCE_COLORS[label] for label in labels],
        alpha=0.85,
    )
    axis.axhline(0, color="black", linewidth=0.9)
    axis.set(title=title, xlabel="Execution scenario", ylabel="ROI (%)")
    axis.tick_params(axis="x", rotation=20)
    axis.bar_label(
        bars,
        labels=[f"n={int(count):,}" for count in plot_data["bets"]],
        padding=4,
        fontsize=9,
    )
    figure.tight_layout()
    return figure, axis


def plot_monthly_metric(
    monthly_by_league: pd.DataFrame,
    monthly_all_leagues: pd.DataFrame,
    *,
    value_column: str,
    title: str,
    y_label: str,
    fill_missing: float | None = None,
    show_zero_line: bool = False,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot a monthly metric for each league and the combined universe.

    Missing calendar months remain gaps by default. Pass ``fill_missing=0`` for
    count metrics where absence means that no bets were placed.
    """
    required_by_league = {"league", "execution_source", "month", value_column}
    required_all = {"execution_source", "month", value_column}
    missing_by_league = required_by_league.difference(monthly_by_league.columns)
    missing_all = required_all.difference(monthly_all_leagues.columns)
    if missing_by_league or missing_all:
        missing = sorted(missing_by_league.union(missing_all))
        raise ValueError("Monthly plot is missing columns: " + ", ".join(missing))
    if monthly_by_league.empty or monthly_all_leagues.empty:
        raise ValueError("Monthly plot data cannot be empty")

    leagues = sorted(monthly_by_league["league"].unique())
    panel_names = [*leagues, "All leagues"]
    figure, axes_array = plt.subplots(
        len(panel_names),
        1,
        figsize=(16, 4.4 * len(panel_names)),
        sharex=True,
        sharey=True,
    )
    axes = list(axes_array)

    available_sources = set(monthly_by_league["execution_source"]).union(
        monthly_all_leagues["execution_source"]
    )
    source_keys = [
        source for source in SOURCE_LABELS if source in available_sources
    ]

    for axis, panel_name in zip(axes, panel_names):
        panel = (
            monthly_all_leagues
            if panel_name == "All leagues"
            else monthly_by_league.loc[
                monthly_by_league["league"].eq(panel_name)
            ]
        )
        calendar = pd.date_range(
            panel["month"].min(), panel["month"].max(), freq="MS"
        )

        for source_key in source_keys:
            source_label = SOURCE_LABELS[source_key]
            series = (
                panel.loc[panel["execution_source"].eq(source_key)]
                .set_index("month")[value_column]
                .reindex(calendar, fill_value=fill_missing)
            )
            axis.plot(
                series.index,
                series,
                label=source_label,
                color=SOURCE_COLORS[source_label],
                marker="o",
                markersize=3,
                linewidth=1.2,
                alpha=0.9,
            )

        if show_zero_line:
            axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(panel_name)
        axis.set_ylabel(y_label)

    axes[-1].set_xlabel("Calendar month")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.suptitle(title, fontsize=16, y=0.995)
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=len(source_keys),
        frameon=False,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.945])
    return figure, axes


def plot_ev_bucket_diagnostics(
    bucket_summary: pd.DataFrame,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Compare mean estimated EV with realized ROI across fixed EV buckets."""
    required = {
        "execution_source",
        "ev_bucket",
        "candidates",
        "mean_estimated_ev_pct",
        "realized_roi_pct",
    }
    missing = required.difference(bucket_summary.columns)
    if missing:
        raise ValueError(
            "EV bucket plot is missing columns: " + ", ".join(sorted(missing))
        )

    source_keys = list(SOURCE_LABELS)
    figure, axes_array = plt.subplots(
        len(source_keys),
        1,
        figsize=(13, 3.8 * len(source_keys)),
        sharex=True,
        sharey=True,
    )
    axes = list(axes_array)
    x_positions = np.arange(len(EV_BUCKET_LABELS))

    for axis, source_key in zip(axes, source_keys):
        source = (
            bucket_summary.loc[
                bucket_summary["execution_source"].eq(source_key)
            ]
            .set_index("ev_bucket")
            .reindex(EV_BUCKET_LABELS)
        )
        source_label = SOURCE_LABELS[source_key]
        bars = axis.bar(
            x_positions,
            source["realized_roi_pct"],
            color=SOURCE_COLORS[source_label],
            alpha=0.75,
            label="Realized ROI",
        )
        axis.plot(
            x_positions,
            source["mean_estimated_ev_pct"],
            color="black",
            linestyle="--",
            marker="D",
            markersize=4,
            label="Mean estimated EV",
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(source_label)
        axis.set_ylabel("Percent")

        for bar, count in zip(bars, source["candidates"]):
            if pd.notna(count):
                height = bar.get_height()
                offset = 4 if height >= 0 else -12
                axis.annotate(
                    f"n={int(count)}",
                    (bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if height >= 0 else "top",
                    fontsize=8,
                )

    axes[-1].set_xticks(x_positions, EV_BUCKET_LABELS)
    axes[-1].set_xlabel("Model-estimated expected-value bucket")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.suptitle(
        "Estimated value versus realized ROI by execution scenario",
        fontsize=16,
        y=0.995,
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.945])
    return figure, axes


def plot_ev_bucket_counts(
    bucket_summary: pd.DataFrame,
    *,
    include_negative: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot candidate counts by EV bucket for every execution scenario."""
    required = {"execution_source", "ev_bucket", "candidates"}
    missing = required.difference(bucket_summary.columns)
    if missing:
        raise ValueError(
            "EV bucket count plot is missing columns: "
            + ", ".join(sorted(missing))
        )

    bucket_labels = (
        EV_BUCKET_LABELS
        if include_negative
        else [label for label in EV_BUCKET_LABELS if label != "<0%"]
    )
    counts = (
        bucket_summary.pivot(
            index="ev_bucket",
            columns="execution_source",
            values="candidates",
        )
        .reindex(index=bucket_labels, columns=list(SOURCE_LABELS))
        .rename(columns=SOURCE_LABELS)
        .fillna(0)
    )
    figure, axis = plt.subplots(figsize=(14, 6))
    counts.plot(
        kind="bar",
        ax=axis,
        color=[SOURCE_COLORS[source] for source in counts.columns],
        width=0.82,
    )
    title_prefix = "Candidate" if include_negative else "Positive-EV candidate"
    axis.set_title(
        f"{title_prefix} count by estimated-EV bucket and execution scenario"
    )
    axis.set_xlabel("Model-estimated expected-value bucket")
    axis.set_ylabel("Candidate matches")
    axis.legend(title="Execution scenario", frameon=False)
    axis.tick_params(axis="x", rotation=0)
    figure.tight_layout()
    return figure, axis


def plot_coefficient_stability(
    coefficients: pd.DataFrame,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot standardized walk-forward coefficients by league and test season."""
    required = {"league", "test_season", "coefficient", "value"}
    missing = required.difference(coefficients.columns)
    if missing:
        raise ValueError(
            "Coefficient plot is missing columns: " + ", ".join(sorted(missing))
        )

    slopes = coefficients.loc[
        coefficients["coefficient"].isin(COEFFICIENT_LABELS)
    ].copy()
    if slopes.empty:
        raise ValueError("Coefficient plot contains no recognized slope coefficients")

    leagues = sorted(slopes["league"].unique())
    figure, axes_array = plt.subplots(
        len(leagues),
        1,
        figsize=(13, 3.8 * len(leagues)),
        sharex=True,
        sharey=True,
    )
    axes = list(np.atleast_1d(axes_array))

    for axis, league in zip(axes, leagues):
        league_data = slopes.loc[slopes["league"].eq(league)]
        seasons = sorted(league_data["test_season"].unique())
        x_positions = np.arange(len(seasons))

        for coefficient, label in COEFFICIENT_LABELS.items():
            series = (
                league_data.loc[league_data["coefficient"].eq(coefficient)]
                .set_index("test_season")["value"]
                .reindex(seasons)
            )
            axis.plot(
                x_positions,
                series,
                label=label,
                color=COEFFICIENT_COLORS[coefficient],
                marker="o",
                linewidth=1.5,
            )

        axis.axhline(0, color="grey", linewidth=0.8)
        axis.set_title(league)
        axis.set_ylabel("Standardized coefficient")
        axis.set_xticks(x_positions, seasons)

    axes[-1].set_xlabel("Out-of-sample test season")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.suptitle(
        "Walk-forward coefficient stability by league",
        fontsize=16,
        y=0.995,
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.972),
        ncol=len(COEFFICIENT_LABELS),
        frameon=False,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.94])
    return figure, axes


def plot_regularization_metrics(
    metrics: pd.DataFrame,
    model_order: list[str],
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Compare proper scoring and calibration across ridge settings."""
    required = {
        "model_name",
        "brier_score",
        "log_loss",
        "calibration_slope",
    }
    missing = required.difference(metrics.columns)
    if missing:
        raise ValueError(
            "Regularization metrics plot is missing columns: "
            + ", ".join(sorted(missing))
        )

    ordered = metrics.set_index("model_name").reindex(model_order)
    figure, axes_array = plt.subplots(1, 3, figsize=(15, 4.5))
    axes = list(axes_array)
    specifications = [
        ("brier_score", "Brier score", None),
        ("log_loss", "Log loss", None),
        ("calibration_slope", "Calibration slope", 1.0),
    ]
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    for axis, (column, title, reference), color in zip(
        axes, specifications, colors
    ):
        axis.bar(model_order, ordered[column], color=color, alpha=0.85)
        if reference is not None:
            axis.axhline(reference, color="black", linestyle="--", linewidth=1)
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=25)

    figure.suptitle(
        "Out-of-sample probability performance by ridge strength",
        fontsize=15,
    )
    figure.tight_layout()
    return figure, axes


def plot_regularization_coefficient_paths(
    coefficients: pd.DataFrame,
) -> tuple[plt.Figure, list[plt.Axes]]:
    """Plot standardized feature coefficients against ridge strength."""
    required = {
        "regularization_l2",
        "test_season",
        "coefficient",
        "value",
    }
    missing = required.difference(coefficients.columns)
    if missing:
        raise ValueError(
            "Regularization coefficient plot is missing columns: "
            + ", ".join(sorted(missing))
        )

    slopes = coefficients.loc[
        coefficients["coefficient"].isin(COEFFICIENT_LABELS)
    ].copy()
    seasons = sorted(slopes["test_season"].unique())
    figure, axes_array = plt.subplots(
        len(seasons),
        1,
        figsize=(13, 3.8 * len(seasons)),
        sharex=True,
        sharey=True,
    )
    axes = list(np.atleast_1d(axes_array))

    for axis, season in zip(axes, seasons):
        season_data = slopes.loc[slopes["test_season"].eq(season)]
        for coefficient, label in COEFFICIENT_LABELS.items():
            series = (
                season_data.loc[
                    season_data["coefficient"].eq(coefficient)
                ]
                .sort_values("regularization_l2")
            )
            axis.plot(
                series["regularization_l2"],
                series["value"],
                label=label,
                color=COEFFICIENT_COLORS[coefficient],
                marker="o",
                linewidth=1.5,
            )
        axis.axhline(0, color="grey", linewidth=0.8)
        axis.set_xscale("log")
        axis.set_title(f"Test season {season}")
        axis.set_ylabel("Standardized coefficient")

    axes[-1].set_xlabel("L2 penalty (log scale)")
    handles, labels = axes[-1].get_legend_handles_labels()
    figure.suptitle(
        "Coefficient shrinkage under stricter regularization",
        fontsize=16,
        y=0.995,
    )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=len(COEFFICIENT_LABELS),
        frameon=False,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.93])
    return figure, axes
