import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from football_edge.plotting import (
    COEFFICIENT_LABELS,
    SOURCE_LABELS,
    plot_coefficient_stability,
    plot_ev_bucket_counts,
    plot_ev_bucket_diagnostics,
    plot_monthly_metric,
    plot_overall_roi_by_execution,
    plot_regularization_coefficient_paths,
    plot_regularization_metrics,
)


def _monthly_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for league in ["League A", "League B"]:
        for source in SOURCE_LABELS:
            rows.append(
                {
                    "league": league,
                    "execution_source": source,
                    "month": pd.Timestamp("2024-08-01"),
                    "roi_pct": 5.0,
                    "bets": 2,
                }
            )
    by_league = pd.DataFrame(rows)
    all_leagues = (
        by_league.groupby(["execution_source", "month"], as_index=False)
        .agg(roi_pct=("roi_pct", "mean"), bets=("bets", "sum"))
    )
    return by_league, all_leagues


def test_monthly_plot_has_one_panel_per_league_plus_combined() -> None:
    by_league, all_leagues = _monthly_frames()

    figure, axes = plot_monthly_metric(
        by_league,
        all_leagues,
        value_column="roi_pct",
        title="Test title",
        y_label="ROI (%)",
        show_zero_line=True,
    )

    assert len(axes) == 3
    assert [axis.get_title() for axis in axes] == [
        "League A",
        "League B",
        "All leagues",
    ]
    assert len(axes[-1].lines) == len(SOURCE_LABELS) + 1
    assert figure._suptitle.get_text() == "Test title"
    plt.close(figure)


def test_monthly_plot_validates_required_columns() -> None:
    by_league, all_leagues = _monthly_frames()

    with pytest.raises(ValueError, match="missing columns"):
        plot_monthly_metric(
            by_league.drop(columns="roi_pct"),
            all_leagues,
            value_column="roi_pct",
            title="Test title",
            y_label="ROI (%)",
        )


def test_overall_roi_plot_has_one_bar_per_execution_source() -> None:
    performance = pd.DataFrame(
        {
            "execution_source": list(SOURCE_LABELS),
            "roi_pct": [1.0] * len(SOURCE_LABELS),
            "roi_95_low_pct": [-2.0] * len(SOURCE_LABELS),
            "roi_95_high_pct": [4.0] * len(SOURCE_LABELS),
            "bets": [100] * len(SOURCE_LABELS),
        }
    )

    figure, axis = plot_overall_roi_by_execution(performance)

    assert len(axis.patches) == len(SOURCE_LABELS)
    assert axis.get_ylabel() == "ROI (%)"
    plt.close(figure)


def test_ev_bucket_plot_has_one_panel_per_execution_source() -> None:
    rows = []
    for source in SOURCE_LABELS:
        rows.append(
            {
                "execution_source": source,
                "ev_bucket": "0-2%",
                "candidates": 10,
                "mean_estimated_ev_pct": 1.0,
                "realized_roi_pct": -2.0,
            }
        )

    figure, axes = plot_ev_bucket_diagnostics(pd.DataFrame(rows))

    assert len(axes) == len(SOURCE_LABELS)
    assert figure._suptitle.get_text().startswith("Estimated value")
    plt.close(figure)


def test_ev_bucket_count_plot_contains_all_execution_sources() -> None:
    rows = []
    for source in SOURCE_LABELS:
        rows.append(
            {
                "execution_source": source,
                "ev_bucket": "0-2%",
                "candidates": 10,
            }
        )

    figure, axis = plot_ev_bucket_counts(
        pd.DataFrame(rows), include_negative=False
    )

    assert len(axis.containers) == len(SOURCE_LABELS)
    assert axis.get_ylabel() == "Candidate matches"
    assert "<0%" not in [tick.get_text() for tick in axis.get_xticklabels()]
    plt.close(figure)


def test_coefficient_plot_has_one_panel_per_league() -> None:
    rows = []
    for league in ["League A", "League B"]:
        for season in ["23_24", "24_25"]:
            for coefficient in COEFFICIENT_LABELS:
                rows.append(
                    {
                        "league": league,
                        "test_season": season,
                        "coefficient": coefficient,
                        "value": 0.1,
                    }
                )

    figure, axes = plot_coefficient_stability(pd.DataFrame(rows))

    assert len(axes) == 2
    assert [axis.get_title() for axis in axes] == ["League A", "League B"]
    assert len(axes[-1].lines) == len(COEFFICIENT_LABELS) + 1
    plt.close(figure)


def test_regularization_plots_have_expected_panels() -> None:
    model_order = ["L2 = 1", "L2 = 10", "L2 = 100"]
    metrics = pd.DataFrame(
        {
            "model_name": model_order,
            "brier_score": [0.24, 0.23, 0.22],
            "log_loss": [0.69, 0.68, 0.67],
            "calibration_slope": [0.9, 1.0, 1.1],
        }
    )
    metric_figure, metric_axes = plot_regularization_metrics(
        metrics, model_order
    )

    coefficient_rows = []
    for season in ["24_25", "25_26"]:
        for l2 in [1.0, 10.0, 100.0]:
            for coefficient in COEFFICIENT_LABELS:
                coefficient_rows.append(
                    {
                        "test_season": season,
                        "regularization_l2": l2,
                        "coefficient": coefficient,
                        "value": 0.1,
                    }
                )
    coefficient_figure, coefficient_axes = (
        plot_regularization_coefficient_paths(
            pd.DataFrame(coefficient_rows)
        )
    )

    assert len(metric_axes) == 3
    assert len(coefficient_axes) == 2
    plt.close(metric_figure)
    plt.close(coefficient_figure)
