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


def plot_market_efficiency_margins(
    market_panel: pd.DataFrame,
    *,
    primary_sources: list[str],
    source_colors: dict[str, str],
) -> list[tuple[plt.Figure, plt.Axes]]:
    """Plot average quoted market margin by season and source."""
    required = {"source", "league", "season", "bookmaker_margin"}
    missing = required.difference(market_panel.columns)
    if missing:
        raise ValueError("Margin plot is missing columns: " + ", ".join(sorted(missing)))

    margin_by_season = (
        market_panel.groupby(["source", "league", "season"], sort=True)
        .agg(mean_margin_pct=("bookmaker_margin", lambda values: values.mean() * 100))
        .reset_index()
    )
    figures = []
    for source in [source for source in primary_sources if source in margin_by_season["source"].unique()]:
        source_frame = margin_by_season.loc[margin_by_season["source"].eq(source)].copy()
        source_average = (
            source_frame.groupby("season", sort=True)["mean_margin_pct"]
            .mean()
            .reset_index(name="cross_league_average_margin_pct")
        )

        figure, axis = plt.subplots(figsize=(12, 4.5))
        for league, league_frame in source_frame.groupby("league", sort=True):
            axis.plot(
                league_frame["season"],
                league_frame["mean_margin_pct"],
                marker="o",
                linewidth=1.2,
                alpha=0.45,
                label=league,
            )
        axis.plot(
            source_average["season"],
            source_average["cross_league_average_margin_pct"],
            color="black",
            linewidth=2.5,
            marker="o",
            label="Cross-league average",
        )
        axis.axhline(0, color="black", linewidth=0.8, linestyle="--")
        axis.set_title(f"Average quoted market margin by season - {source}")
        axis.set_xlabel("Season")
        axis.set_ylabel("Mean quoted margin (%)")
        axis.tick_params(axis="x", rotation=25)
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
        figure.tight_layout()
        figures.append((figure, axis))
    return figures


def plot_paper_rule_total_results(
    paper_summary: pd.DataFrame,
    *,
    primary_sources: list[str],
    source_colors: dict[str, str],
) -> tuple[plt.Figure, list[plt.Axes]] | None:
    """Plot total paper-rule ROI and number of selected bets."""
    if paper_summary.empty:
        print("No paper-rule bets are available for the selected configuration.")
        return None
    sources = [source for source in primary_sources if source in paper_summary["source"].unique()]
    total_plot = paper_summary.set_index("source").reindex(sources)
    colors = [source_colors[source] for source in sources]

    figure, axes_array = plt.subplots(1, 2, figsize=(14, 4.5))
    axes = list(axes_array)
    axes[0].bar(sources, total_plot["roi_pct"], color=colors)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_title("Total ROI")
    axes[0].set_ylabel("ROI (%)")
    axes[1].bar(sources, total_plot["bets"], color=colors)
    axes[1].set_title("Selected bets")
    axes[1].set_ylabel("Bets")
    for axis in axes:
        axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    return figure, axes


def plot_paper_rule_by_season(
    paper_season_summary: pd.DataFrame,
    *,
    primary_sources: list[str],
    source_colors: dict[str, str],
) -> tuple[plt.Figure, list[plt.Axes]] | None:
    """Plot paper-rule ROI and bet count by season."""
    if paper_season_summary.empty:
        print("No season-level paper-rule bets are available.")
        return None
    sources = [source for source in primary_sources if source in paper_season_summary["source"].unique()]
    figure, axes_array = plt.subplots(len(sources), 1, figsize=(12, 4.2 * len(sources)), sharex=False)
    axes = list(np.atleast_1d(axes_array))

    for axis, source in zip(axes, sources):
        subset = paper_season_summary.loc[paper_season_summary["source"].eq(source)].sort_values("season")
        axis.plot(
            subset["season"],
            subset["roi_pct"],
            marker="o",
            linewidth=2,
            color=source_colors[source],
            label="ROI",
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis2 = axis.twinx()
        axis2.bar(subset["season"], subset["bets"], alpha=0.25, color=source_colors[source], label="bets")
        axis.set_title(f"ROI and bet count by season - {source}")
        axis.set_ylabel("ROI (%)")
        axis2.set_ylabel("Bets")
        axis.tick_params(axis="x", rotation=25)
    axes[-1].set_xlabel("Season")
    figure.tight_layout()
    return figure, axes


def plot_paper_rule_by_league(
    paper_league_summary: pd.DataFrame,
    *,
    primary_sources: list[str],
    source_colors: dict[str, str],
) -> list[tuple[plt.Figure, list[plt.Axes]]]:
    """Plot paper-rule ROI and selected bets by league."""
    if paper_league_summary.empty:
        print("No league-level paper-rule bets are available.")
        return []
    figures = []
    for source in [source for source in primary_sources if source in paper_league_summary["source"].unique()]:
        source_frame = paper_league_summary.loc[paper_league_summary["source"].eq(source)].copy()
        source_frame = source_frame.sort_values("roi_pct", ascending=True, na_position="first")
        figure, axes_array = plt.subplots(1, 2, figsize=(14, max(4.5, 0.35 * len(source_frame))))
        axes = list(axes_array)
        color = source_colors[source]
        axes[0].barh(source_frame["league"], source_frame["roi_pct"], color=color)
        axes[0].axvline(0, color="black", linewidth=0.8)
        axes[0].set_title(f"ROI by league - {source}")
        axes[0].set_xlabel("ROI (%)")
        axes[1].barh(source_frame["league"], source_frame["bets"], color=color, alpha=0.75)
        axes[1].set_title(f"Selected bets by league - {source}")
        axes[1].set_xlabel("Bets")
        figure.tight_layout()
        figures.append((figure, axes))
    return figures


def plot_market_maximum_robustness(
    *,
    market_maximum_bets: pd.DataFrame,
    robustness_summary: pd.DataFrame,
    season_league_bootstrap_roi: pd.DataFrame,
    season_bootstrap_roi: pd.DataFrame,
    profit_contribution_summary: pd.DataFrame,
    season_bet_level_bootstrap_summary: pd.DataFrame,
    season_blocks: pd.DataFrame,
    color: str,
) -> list[plt.Figure]:
    """Plot compact market-maximum robustness diagnostics without leave-one-out."""
    if market_maximum_bets.empty:
        print("No market-maximum robustness plots are available.")
        return []

    figures = []
    figure, axes_array = plt.subplots(2, 1, figsize=(12, 7.5), sharex=False)
    axes = list(axes_array)
    for axis, (label, data) in zip(
        axes,
        [("season-league", season_league_bootstrap_roi), ("season", season_bootstrap_roi)],
    ):
        observed_roi = robustness_summary.loc[
            robustness_summary["group_label"].eq(label), "observed_roi_pct"
        ].iloc[0]
        axis.hist(data["roi_pct"], bins=50, color=color, alpha=0.70)
        axis.axvline(0, color="black", linewidth=0.8, linestyle="--", label="0% ROI")
        axis.axvline(observed_roi, color="#111111", linewidth=2, label=f"observed ROI = {observed_roi:.2f}%")
        axis.set_title(f"Market-maximum block-bootstrap ROI distribution - {label}")
        axis.set_xlabel("Bootstrap ROI (%)")
        axis.set_ylabel("Simulations")
        axis.legend()
    figure.tight_layout()
    figures.append(figure)

    if not profit_contribution_summary.empty:
        figure, axis = plt.subplots(figsize=(8, 4.5))
        subset = profit_contribution_summary.sort_values("top_winning_blocks_pct")
        axis.plot(
            subset["top_winning_blocks_pct"],
            subset["share_of_positive_profit_pct"],
            marker="o",
            linewidth=2,
            color=color,
            label="share of positive profit",
        )
        axis.axline((0, 0), slope=1, color="black", linewidth=0.8, linestyle="--", label="even contribution")
        axis.set_title("Market-maximum profit concentration by season-league block")
        axis.set_xlabel("Top winning blocks included (%)")
        axis.set_ylabel("Cumulative positive profit contribution (%)")
        axis.set_xlim(0, 100)
        axis.set_ylim(0, max(100, subset["share_of_positive_profit_pct"].max() * 1.05))
        axis.legend()
        figure.tight_layout()
        figures.append(figure)

    if not season_bet_level_bootstrap_summary.empty:
        season_bet_level_plot = season_bet_level_bootstrap_summary.merge(
            season_blocks[["season", "roi_pct"]],
            on="season",
            how="left",
            validate="one_to_one",
        ).rename(columns={"roi_pct": "season_result_roi_pct"})
        season_bet_level_plot = season_bet_level_plot.sort_values("season", kind="stable").copy()
        y_pos = np.arange(len(season_bet_level_plot))

        figure, axis = plt.subplots(figsize=(12, max(4.0, 0.50 * len(season_bet_level_plot))))
        axis.hlines(
            y=y_pos,
            xmin=season_bet_level_plot["bootstrap_p05_roi_pct"],
            xmax=season_bet_level_plot["bootstrap_p95_roi_pct"],
            color="#4C4C4C",
            linewidth=3,
            alpha=0.75,
            label="5-95% bet-level bootstrap interval",
        )
        axis.scatter(
            season_bet_level_plot["bootstrap_median_roi_pct"],
            y_pos,
            color="#6A5ACD",
            s=55,
            alpha=0.90,
            label="bootstrap median ROI",
        )
        axis.scatter(
            season_bet_level_plot["season_result_roi_pct"],
            y_pos,
            color=color,
            s=np.maximum(55, np.sqrt(season_bet_level_plot["bets"]) * 10),
            alpha=0.95,
            label="observed season ROI",
        )
        axis.axvline(0, color="black", linewidth=0.8, linestyle="--")
        axis.set_yticks(y_pos, season_bet_level_plot["season"])
        axis.set_title("Market-maximum bet-level bootstrap by season")
        axis.set_xlabel("ROI within season across all selected leagues (%)")
        axis.set_ylabel("Season")
        axis.legend()
        figure.tight_layout()
        figures.append(figure)
    return figures


def plot_kelly_diagnostics(
    *,
    kelly_results: pd.DataFrame,
    kelly_paths: pd.DataFrame,
    kelly_input: pd.DataFrame,
    kelly_summary: pd.DataFrame,
    kelly_season_summary: pd.DataFrame,
    kelly_league_summary: pd.DataFrame,
    kelly_bootstrap_results: pd.DataFrame,
    kelly_bootstrap_summary: pd.DataFrame,
    kelly_fractions: list[float],
    initial_bankroll: float,
) -> list[plt.Figure]:
    """Plot compact fractional-Kelly diagnostics."""
    if kelly_results.empty:
        print("No Kelly plots are available.")
        return []

    kelly_palette = {0.10: "#4C78A8", 0.25: "#54A24B", 0.50: "#B279A2", 1.00: "#E45756"}
    figures = []

    figure, axis = plt.subplots(figsize=(12, 4.8))
    for fraction, path in kelly_paths.groupby("kelly_fraction", sort=True):
        path = path.dropna(subset=["date"]).sort_values("date")
        axis.plot(
            path["date"],
            path["bankroll"],
            linewidth=2,
            color=kelly_palette.get(fraction),
            label=f"{fraction:.0%} Kelly",
        )
    axis.axhline(initial_bankroll, color="black", linewidth=0.8, linestyle="--")
    axis.set_title("Market-maximum Kelly bankroll path")
    axis.set_xlabel("Date")
    axis.set_ylabel("Bankroll")
    axis.legend()
    figure.tight_layout()
    figures.append(figure)

    if not kelly_season_summary.empty:
        figure, axis = plt.subplots(figsize=(12, 4.8))
        season_bet_counts = kelly_input.groupby("season", sort=True).size().rename("bets").reset_index()
        axis2 = axis.twinx()
        axis2.bar(season_bet_counts["season"], season_bet_counts["bets"], color="#9E9E9E", alpha=0.22, label="selected bets")
        for fraction, subset in kelly_season_summary.groupby("kelly_fraction", sort=True):
            subset = subset.sort_values("season")
            axis.plot(
                subset["season"],
                subset["roi_pct"],
                marker="o",
                linewidth=2,
                color=kelly_palette.get(fraction),
                label=f"{fraction:.0%} Kelly",
            )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title("Market-maximum Kelly return on staked capital by season")
        axis.set_xlabel("Season")
        axis.set_ylabel("Profit / stake (%)")
        axis2.set_ylabel("Selected bets")
        axis.tick_params(axis="x", rotation=25)
        lines, labels = axis.get_legend_handles_labels()
        bars, bar_labels = axis2.get_legend_handles_labels()
        axis.legend(lines + bars, labels + bar_labels, loc="best")
        figure.tight_layout()
        figures.append(figure)

    if not kelly_league_summary.empty:
        league_order = (
            kelly_league_summary.loc[kelly_league_summary["kelly_fraction"].eq(max(kelly_fractions))]
            .sort_values("roi_pct", ascending=True)["league"]
            .tolist()
        )
        y = np.arange(len(league_order))
        width = 0.22
        figure, axis = plt.subplots(figsize=(12, max(5.0, 0.35 * len(league_order))))
        for offset, fraction in zip(np.linspace(-width, width, len(kelly_fractions)), kelly_fractions):
            subset = (
                kelly_league_summary.loc[kelly_league_summary["kelly_fraction"].eq(fraction)]
                .set_index("league")
                .reindex(league_order)
            )
            axis.barh(
                y + offset,
                subset["roi_pct"],
                height=width,
                color=kelly_palette.get(fraction),
                alpha=0.85,
                label=f"{fraction:.0%} Kelly",
            )
        axis.axvline(0, color="black", linewidth=0.8)
        axis.set_yticks(y, league_order)
        axis.set_title("Market-maximum Kelly return on staked capital by league")
        axis.set_xlabel("Profit / stake (%)")
        axis.legend()
        figure.tight_layout()
        figures.append(figure)

    if not kelly_bootstrap_results.empty:
        figure, axes_array = plt.subplots(len(kelly_fractions), 1, figsize=(12, 3.4 * len(kelly_fractions)), sharex=True)
        axes = list(np.atleast_1d(axes_array))
        for axis, fraction in zip(axes, kelly_fractions):
            subset = kelly_bootstrap_results.loc[kelly_bootstrap_results["kelly_fraction"].eq(fraction)]
            observed = kelly_summary.loc[
                kelly_summary["kelly_fraction"].eq(fraction), "bankroll_return_pct"
            ].iloc[0]
            probability_losing = kelly_bootstrap_summary.loc[
                kelly_bootstrap_summary["kelly_fraction"].eq(fraction),
                "probability_losing_money_pct",
            ].iloc[0]
            axis.hist(subset["bankroll_return_pct"], bins=50, color=kelly_palette.get(fraction), alpha=0.70)
            axis.axvline(0, color="black", linewidth=0.8, linestyle="--", label="0% return")
            axis.axvline(observed, color="#111111", linewidth=2, label=f"observed = {observed:.2f}%")
            axis.set_title(
                f"Bet-level bootstrap bankroll return - {fraction:.0%} Kelly "
                f"(P(loss) = {probability_losing:.2f}%)"
            )
            axis.set_ylabel("Simulations")
            axis.legend()
        axes[-1].set_xlabel("Bootstrap bankroll return (%)")
        figure.tight_layout()
        figures.append(figure)
    return figures

