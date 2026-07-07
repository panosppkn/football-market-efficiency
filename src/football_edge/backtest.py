"""Walk-forward prediction and flat-stake betting evaluation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from football_edge.config import (
    BOOKMAKER_SCENARIOS,
    CLOSING_PRICE_SCENARIOS,
    EV_BUCKET_EDGES,
    EV_BUCKET_LABELS,
    MINIMUM_EXPECTED_VALUE,
    MODEL_FEATURES,
    PRICE_SCENARIOS,
)
from football_edge.data import Dataset
from football_edge.features import create_goal_features
from football_edge.model import fit_logistic_regression, predict_probability


def add_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add margin-free pre-closing probability without using closing information."""
    frame = frame.copy()
    valid = frame[["Avg>2.5", "Avg<2.5"]].gt(1).all(axis=1)
    over_implied = 1 / frame["Avg>2.5"].where(valid)
    under_implied = 1 / frame["Avg<2.5"].where(valid)
    frame["market_over_probability"] = over_implied / (
        over_implied + under_implied
    )
    probability = frame["market_over_probability"].clip(1e-6, 1 - 1e-6)
    frame["market_logit"] = np.log(probability / (1 - probability))
    return frame


def build_research_dataset(datasets: Iterable[Dataset]) -> pd.DataFrame:
    """Create one combined frame while retaining league-season boundaries."""
    frames = []
    for dataset in datasets:
        frame = add_market_features(create_goal_features(dataset.path))
        frame["league"] = dataset.league
        frame["season"] = dataset.season
        frames.append(frame)
    if not frames:
        raise ValueError("At least one dataset is required")
    return pd.concat(frames, ignore_index=True)


def run_walk_forward_with_coefficients(
    all_matches: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict future seasons and retain standardized fold coefficients."""
    required = MODEL_FEATURES + [
        "over_2_5",
        "Avg>2.5",
    ]
    model_data = all_matches.dropna(subset=required).copy()
    predictions = []
    coefficient_rows = []

    for league, league_data in model_data.groupby("league", sort=True):
        seasons = sorted(league_data["season"].unique())
        for test_season in seasons[1:]:
            train = league_data.loc[league_data["season"] < test_season]
            test = league_data.loc[
                league_data["season"].eq(test_season)
            ].copy()
            if train.empty or test.empty:
                continue
            train_end_date = train["date"].max()
            test_start_date = test["date"].min()
            if train_end_date >= test_start_date:
                raise ValueError(
                    f"Walk-forward overlap for {league} {test_season}: "
                    f"training ends {train_end_date}, test starts {test_start_date}"
                )

            model = fit_logistic_regression(
                train[MODEL_FEATURES].to_numpy(),
                train["over_2_5"].to_numpy(),
                l2=100.0,
            )
            coefficient_names = ["intercept", *MODEL_FEATURES]
            coefficient_rows.extend(
                {
                    "league": league,
                    "test_season": test_season,
                    "training_seasons": train["season"].nunique(),
                    "training_matches": len(train),
                    "coefficient": coefficient,
                    "value": value,
                }
                for coefficient, value in zip(
                    coefficient_names, model.coefficients
                )
            )
            test["model_probability"] = predict_probability(
                model, test[MODEL_FEATURES].to_numpy()
            )
            test["training_seasons"] = train["season"].nunique()
            test["training_matches"] = len(train)
            test["train_end_date"] = train_end_date
            test["test_start_date"] = test_start_date
            test["model_iterations"] = model.iterations
            predictions.append(test)

    if not predictions:
        raise ValueError("Walk-forward evaluation needs at least two seasons")
    prediction_frame = (
        pd.concat(predictions, ignore_index=True)
        .sort_values(["league", "season", "date"])
        .reset_index(drop=True)
    )
    coefficient_frame = pd.DataFrame(coefficient_rows).sort_values(
        ["league", "test_season", "coefficient"], kind="stable"
    )
    return prediction_frame, coefficient_frame.reset_index(drop=True)


def run_walk_forward(all_matches: pd.DataFrame) -> pd.DataFrame:
    """Predict each season using prior seasons from the same league only."""
    predictions, _ = run_walk_forward_with_coefficients(all_matches)
    return predictions


def coefficient_stability_summary(
    coefficients: pd.DataFrame,
    *,
    include_intercept: bool = False,
) -> pd.DataFrame:
    """Summarize coefficient dispersion and fold-to-fold sign changes."""
    required = {"league", "test_season", "coefficient", "value"}
    missing = required.difference(coefficients.columns)
    if missing:
        raise ValueError(
            "Coefficient stability is missing columns: "
            + ", ".join(sorted(missing))
        )

    frame = coefficients.copy()
    if not include_intercept:
        frame = frame.loc[frame["coefficient"].ne("intercept")]
    frame = frame.sort_values(["league", "coefficient", "test_season"])
    frame["sign"] = np.sign(frame["value"])
    frame["sign_change"] = (
        frame.groupby(["league", "coefficient"])["sign"]
        .diff()
        .fillna(0)
        .ne(0)
    )

    return (
        frame.groupby(["league", "coefficient"], sort=True)
        .agg(
            folds=("value", "size"),
            mean_coefficient=("value", "mean"),
            std_coefficient=("value", "std"),
            min_coefficient=("value", "min"),
            max_coefficient=("value", "max"),
            positive_share=("value", lambda values: values.gt(0).mean()),
            sign_changes=("sign_change", "sum"),
        )
        .reset_index()
    )


def run_pooled_rolling_walk_forward(
    all_matches: pd.DataFrame,
    *,
    l2: float,
    model_name: str,
    training_window: int = 2,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one all-league model using exactly the previous N seasons.

    League dummy variables permit different intercept levels while market and
    form slopes remain shared across leagues.
    """
    if training_window < 1:
        raise ValueError("training_window must be at least one season")
    if l2 < 0:
        raise ValueError("l2 must be non-negative")

    base_features = (
        list(feature_columns) if feature_columns is not None else MODEL_FEATURES
    )
    if not base_features:
        raise ValueError("feature_columns must contain at least one feature")

    required = [*base_features, "over_2_5", "league", "season", "date"]
    missing_columns = set(required).difference(all_matches.columns)
    if missing_columns:
        raise ValueError(
            "Pooled walk-forward data is missing columns: "
            + ", ".join(sorted(missing_columns))
        )
    model_data = all_matches.dropna(subset=required).copy()
    league_dummies = pd.get_dummies(
        model_data["league"], prefix="league", drop_first=True, dtype=float
    )
    league_features = list(league_dummies.columns)
    model_data = pd.concat([model_data, league_dummies], axis=1)
    model_feature_columns = [*base_features, *league_features]

    seasons = sorted(model_data["season"].unique())
    if len(seasons) <= training_window:
        raise ValueError("Not enough seasons for the requested training window")

    predictions = []
    coefficient_rows = []
    for test_index in range(training_window, len(seasons)):
        test_season = seasons[test_index]
        training_seasons = seasons[
            test_index - training_window : test_index
        ]
        train = model_data.loc[
            model_data["season"].isin(training_seasons)
        ]
        test = model_data.loc[model_data["season"].eq(test_season)].copy()
        if train.empty or test.empty:
            continue

        train_end_date = train["date"].max()
        test_start_date = test["date"].min()
        if train_end_date >= test_start_date:
            raise ValueError(
                f"Pooled walk-forward overlap for {test_season}: "
                f"training ends {train_end_date}, test starts {test_start_date}"
            )

        model = fit_logistic_regression(
            train[model_feature_columns].to_numpy(),
            train["over_2_5"].to_numpy(),
            l2=l2,
        )
        test["model_probability"] = predict_probability(
            model, test[model_feature_columns].to_numpy()
        )
        test["model_name"] = model_name
        test["regularization_l2"] = l2
        test["training_window"] = training_window
        test["training_seasons"] = ",".join(training_seasons)
        test["training_matches"] = len(train)
        test["train_end_date"] = train_end_date
        test["test_start_date"] = test_start_date
        test["model_iterations"] = model.iterations
        predictions.append(test)

        coefficient_names = ["intercept", *model_feature_columns]
        coefficient_rows.extend(
            {
                "model_name": model_name,
                "regularization_l2": l2,
                "test_season": test_season,
                "training_seasons": ",".join(training_seasons),
                "training_matches": len(train),
                "coefficient": coefficient,
                "value": value,
            }
            for coefficient, value in zip(
                coefficient_names, model.coefficients
            )
        )

    if not predictions:
        raise ValueError("Pooled walk-forward evaluation produced no predictions")

    prediction_frame = (
        pd.concat(predictions, ignore_index=True)
        .sort_values(["season", "league", "date"], kind="stable")
        .reset_index(drop=True)
    )
    coefficient_frame = pd.DataFrame(coefficient_rows).sort_values(
        ["test_season", "coefficient"], kind="stable"
    )
    return prediction_frame, coefficient_frame.reset_index(drop=True)


def run_regularization_grid(
    all_matches: pd.DataFrame,
    configurations: dict[str, float],
    *,
    training_window: int = 2,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a fixed ridge grid on identical pooled rolling test periods."""
    prediction_frames = []
    coefficient_frames = []
    for model_name, l2 in configurations.items():
        predictions, coefficients = run_pooled_rolling_walk_forward(
            all_matches,
            l2=l2,
            model_name=model_name,
            training_window=training_window,
            feature_columns=feature_columns,
        )
        prediction_frames.append(predictions)
        coefficient_frames.append(coefficients)

    combined_predictions = pd.concat(prediction_frames, ignore_index=True)
    combined_coefficients = pd.concat(coefficient_frames, ignore_index=True)
    sample_counts = combined_predictions.groupby("model_name").size()
    if sample_counts.nunique() != 1:
        raise ValueError("Regularization configurations use unequal test samples")
    return combined_predictions, combined_coefficients


MONTHLY_RECALIBRATION_WINDOWS = {"12M", "18M", "24M", "all_history"}


def run_pooled_monthly_recalibration_walk_forward(
    all_matches: pd.DataFrame,
    window: str,
    *,
    l2: float = 100.0,
    model_name: str | None = None,
    feature_columns: list[str] | None = None,
    min_training_matches: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit one pooled model recalibrated at the start of each test month.

    ``window`` must be one of ``12M``, ``18M``, ``24M``, or ``all_history``.
    Rolling windows train on matches in ``[test_start - window, test_start)``;
    the expanding window trains on all matches with ``date < test_start``.
    League dummy variables are built once on the full model dataset, matching
    the season-level pooled walk-forward implementation.
    """
    if window not in MONTHLY_RECALIBRATION_WINDOWS:
        raise ValueError(
            "window must be one of: " + ", ".join(sorted(MONTHLY_RECALIBRATION_WINDOWS))
        )
    if l2 < 0:
        raise ValueError("l2 must be non-negative")

    base_features = (
        list(feature_columns) if feature_columns is not None else MODEL_FEATURES
    )
    if not base_features:
        raise ValueError("feature_columns must contain at least one feature")

    required = [*base_features, "over_2_5", "league", "season", "date"]
    missing_columns = set(required).difference(all_matches.columns)
    if missing_columns:
        raise ValueError(
            "Monthly recalibration data is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    model_data = all_matches.dropna(subset=required).copy()
    model_data["date"] = pd.to_datetime(model_data["date"], errors="raise")
    model_data = model_data.sort_values("date", kind="stable").reset_index(drop=True)
    league_dummies = pd.get_dummies(
        model_data["league"], prefix="league", drop_first=True, dtype=float
    )
    league_features = list(league_dummies.columns)
    model_data = pd.concat([model_data, league_dummies], axis=1)
    model_feature_columns = [*base_features, *league_features]

    if min_training_matches is None:
        min_training_matches = max(100, 10 * len(model_feature_columns))
    if min_training_matches < 1:
        raise ValueError("min_training_matches must be positive")

    first_available_month = model_data["date"].min().to_period("M").to_timestamp()
    month_starts = pd.date_range(
        first_available_month,
        model_data["date"].max().to_period("M").to_timestamp(),
        freq="MS",
    )
    model_label = model_name or f"monthly_{window}"

    predictions = []
    coefficient_rows = []
    for test_start in month_starts:
        test_end = test_start + pd.DateOffset(months=1)
        test = model_data.loc[
            model_data["date"].ge(test_start) & model_data["date"].lt(test_end)
        ].copy()
        if test.empty:
            continue

        if window == "all_history":
            train_start = pd.NaT
            train_mask = model_data["date"].lt(test_start)
        else:
            months = int(window.removesuffix("M"))
            train_start = test_start - pd.DateOffset(months=months)
            if train_start < first_available_month:
                continue
            train_mask = model_data["date"].ge(train_start) & model_data[
                "date"
            ].lt(test_start)
        train = model_data.loc[train_mask]
        if len(train) < min_training_matches:
            continue

        train_end_date = train["date"].max()
        test_start_date = test["date"].min()
        if train_end_date >= test_start_date:
            raise ValueError(
                f"Monthly recalibration overlap for {window} {test_start:%Y-%m}: "
                f"training ends {train_end_date}, test starts {test_start_date}"
            )

        model = fit_logistic_regression(
            train[model_feature_columns].to_numpy(),
            train["over_2_5"].to_numpy(),
            l2=l2,
        )
        test["model_probability"] = predict_probability(
            model, test[model_feature_columns].to_numpy()
        )
        test["model_name"] = model_label
        test["regularization_l2"] = l2
        test["training_window"] = window
        test["recalibration_month"] = test_start
        test["training_matches"] = len(train)
        test["test_matches"] = len(test)
        test["train_start_date"] = train_start
        test["train_end_date"] = train_end_date
        test["test_start_date"] = test_start_date
        test["test_end_date"] = test_end
        test["model_iterations"] = model.iterations
        predictions.append(test)

        coefficient_names = ["intercept", *model_feature_columns]
        coefficient_rows.extend(
            {
                "model_name": model_label,
                "training_window": window,
                "recalibration_month": test_start,
                "training_matches": len(train),
                "test_matches": len(test),
                "regularization_l2": l2,
                "coefficient": coefficient,
                "value": value,
            }
            for coefficient, value in zip(coefficient_names, model.coefficients)
        )

    if not predictions:
        raise ValueError(
            "Monthly recalibration produced no predictions; reduce min_training_matches "
            "or provide more historical data"
        )

    prediction_frame = (
        pd.concat(predictions, ignore_index=True)
        .sort_values(["recalibration_month", "league", "date"], kind="stable")
        .reset_index(drop=True)
    )
    coefficient_frame = pd.DataFrame(coefficient_rows).sort_values(
        ["recalibration_month", "coefficient"], kind="stable"
    )
    return prediction_frame, coefficient_frame.reset_index(drop=True)


def probability_performance(
    predictions: pd.DataFrame,
    *,
    probability_column: str,
    group_by: list[str],
) -> pd.DataFrame:
    """Calculate proper scoring and calibration metrics by group."""
    required = {probability_column, "over_2_5", *group_by}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            "Probability performance is missing columns: "
            + ", ".join(sorted(missing))
        )

    rows = []
    for keys, group in predictions.groupby(group_by, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        outcome = group["over_2_5"].to_numpy(float)
        probability = np.clip(
            group[probability_column].to_numpy(float), 1e-9, 1 - 1e-9
        )
        logit = np.log(probability / (1 - probability)).reshape(-1, 1)

        calibration_model = fit_logistic_regression(
            logit,
            outcome,
            l2=1e-8,
        )
        calibration_slope = (
            calibration_model.coefficients[1]
            / calibration_model.scale[0]
        )
        calibration_intercept = (
            calibration_model.coefficients[0]
            - calibration_model.mean[0] * calibration_slope
        )

        row = dict(zip(group_by, keys))
        row.update(
            {
                "observations": len(group),
                "brier_score": np.mean((probability - outcome) ** 2),
                "log_loss": -np.mean(
                    outcome * np.log(probability)
                    + (1 - outcome) * np.log(1 - probability)
                ),
                "accuracy": np.mean((probability >= 0.5) == outcome),
                "calibration_intercept": calibration_intercept,
                "calibration_slope": calibration_slope,
                "mean_probability": probability.mean(),
                "event_rate": outcome.mean(),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def probability_scores(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare model and market Brier scores; lower values are better."""
    rows = []
    for (league, season), group in predictions.groupby(["league", "season"]):
        outcome = group["over_2_5"]
        rows.append(
            {
                "league": league,
                "season": season,
                "matches": len(group),
                "brier_model": np.mean(
                    (group["model_probability"] - outcome) ** 2
                ),
                "brier_market": np.mean(
                    (group["market_over_probability"] - outcome) ** 2
                ),
            }
        )
    return pd.DataFrame(rows)


def select_bets(
    predictions: pd.DataFrame,
    *,
    minimum_expected_value: float = MINIMUM_EXPECTED_VALUE,
) -> pd.DataFrame:
    """Select model bets independently for average and best-price scenarios."""
    frames = []
    for scenario, (price_column, closing_column) in PRICE_SCENARIOS.items():
        valid = (
            predictions[[price_column, closing_column]].notna().all(axis=1)
            & predictions[price_column].gt(1)
            & predictions[closing_column].gt(1)
        )
        candidates = predictions.loc[valid].copy()
        candidates["expected_value"] = (
            candidates["model_probability"] * candidates[price_column] - 1
        )
        bets = candidates.loc[
            candidates["expected_value"] >= minimum_expected_value
        ].copy()
        bets["price_scenario"] = scenario
        bets["bet_odds"] = bets[price_column]
        bets["closing_odds"] = bets[closing_column]
        bets["profit"] = np.where(
            bets["over_2_5"].eq(1), bets["bet_odds"] - 1, -1.0
        )
        frames.append(bets)
    return pd.concat(frames, ignore_index=True)


def bookmaker_coverage(
    predictions: pd.DataFrame,
    *,
    common_sample: bool = False,
) -> pd.DataFrame:
    """Report usable quote coverage for each named execution venue."""
    common_mask = pd.Series(True, index=predictions.index)
    if common_sample:
        for scenario in BOOKMAKER_SCENARIOS.values():
            columns = [scenario["price_column"], scenario["closing_column"]]
            common_mask &= predictions[columns].notna().all(axis=1)
            common_mask &= predictions[columns].gt(1).all(axis=1)

    rows = []
    comparison_sample = predictions.loc[common_mask]
    for bookmaker, scenario in BOOKMAKER_SCENARIOS.items():
        columns = [scenario["price_column"], scenario["closing_column"]]
        valid = comparison_sample[columns].notna().all(axis=1)
        valid &= comparison_sample[columns].gt(1).all(axis=1)
        rows.append(
            {
                "bookmaker": bookmaker,
                "available_matches": int(valid.sum()),
                "comparison_matches": len(comparison_sample),
                "coverage_pct": valid.mean() * 100,
                "common_sample": common_sample,
            }
        )
    return pd.DataFrame(rows)


def _net_decimal_odds(odds: pd.Series, commission: float) -> pd.Series:
    """Convert quoted odds to net odds after commission on winnings."""
    return 1 + (odds - 1) * (1 - commission)


def select_bookmaker_bets(
    predictions: pd.DataFrame,
    *,
    minimum_expected_value: float = MINIMUM_EXPECTED_VALUE,
    common_sample: bool = False,
) -> pd.DataFrame:
    """Select bets at each named source using one shared model probability.

    Bet sets are source-specific because expected value depends on the offered
    price. If ``common_sample`` is true, every source must have both a
    pre-closing and closing quote for the match before any source is evaluated.
    """
    common_mask = pd.Series(True, index=predictions.index)
    if common_sample:
        for scenario in BOOKMAKER_SCENARIOS.values():
            columns = [scenario["price_column"], scenario["closing_column"]]
            common_mask &= predictions[columns].notna().all(axis=1)
            common_mask &= predictions[columns].gt(1).all(axis=1)

    frames = []
    for bookmaker, scenario in BOOKMAKER_SCENARIOS.items():
        price_column = scenario["price_column"]
        closing_column = scenario["closing_column"]
        commission = float(scenario["commission"])

        valid = (
            common_mask
            & predictions[[price_column, closing_column]].notna().all(axis=1)
            & predictions[price_column].gt(1)
            & predictions[closing_column].gt(1)
        )
        candidates = predictions.loc[valid].copy()
        candidates["quoted_odds"] = candidates[price_column]
        candidates["quoted_closing_odds"] = candidates[closing_column]
        candidates["bet_odds"] = _net_decimal_odds(
            candidates["quoted_odds"], commission
        )
        candidates["closing_odds"] = _net_decimal_odds(
            candidates["quoted_closing_odds"], commission
        )
        candidates["expected_value"] = (
            candidates["model_probability"] * candidates["bet_odds"] - 1
        )

        bets = candidates.loc[
            candidates["expected_value"] >= minimum_expected_value
        ].copy()
        bets["bookmaker"] = bookmaker
        bets["commission"] = commission
        bets["common_sample"] = common_sample
        bets["profit"] = np.where(
            bets["over_2_5"].eq(1), bets["bet_odds"] - 1, -1.0
        )
        frames.append(bets)

    return pd.concat(frames, ignore_index=True)


def build_execution_candidates(
    predictions: pd.DataFrame,
    *,
    include_closing_prices: bool = False,
    probability_column: str = "model_probability",
) -> pd.DataFrame:
    """Build every valid pre-closing candidate for all execution scenarios.

    Candidate eligibility depends only on the pre-closing quote. Closing odds
    are retained when available for CLV diagnostics but never determine whether
    a candidate enters the sample. Set ``include_closing_prices=True`` to add
    explicitly labelled closing-price execution sensitivities. Their CLV is
    undefined because the dataset contains no subsequent reference quote.
    """
    if probability_column not in predictions.columns:
        raise ValueError(
            f"Probability column {probability_column!r} is not available"
        )

    frames = []

    for source, (price_column, closing_column) in PRICE_SCENARIOS.items():
        valid_price = predictions[price_column].notna() & predictions[
            price_column
        ].gt(1)
        candidates = predictions.loc[valid_price].copy()
        valid_close = candidates[closing_column].notna() & candidates[
            closing_column
        ].gt(1)
        candidates["execution_source"] = source
        candidates["commission"] = 0.0
        candidates["quoted_odds"] = candidates[price_column]
        candidates["quoted_closing_odds"] = candidates[closing_column].where(
            valid_close
        )
        candidates["bet_odds"] = candidates["quoted_odds"]
        candidates["closing_odds"] = candidates["quoted_closing_odds"]
        frames.append(candidates)

    for source, scenario in BOOKMAKER_SCENARIOS.items():
        price_column = scenario["price_column"]
        closing_column = scenario["closing_column"]
        commission = float(scenario["commission"])
        valid_price = predictions[price_column].notna() & predictions[
            price_column
        ].gt(1)
        candidates = predictions.loc[valid_price].copy()
        valid_close = candidates[closing_column].notna() & candidates[
            closing_column
        ].gt(1)
        candidates["execution_source"] = source
        candidates["commission"] = commission
        candidates["quoted_odds"] = candidates[price_column]
        candidates["quoted_closing_odds"] = candidates[closing_column].where(
            valid_close
        )
        candidates["bet_odds"] = _net_decimal_odds(
            candidates["quoted_odds"], commission
        )
        candidates["closing_odds"] = _net_decimal_odds(
            candidates["quoted_closing_odds"], commission
        )
        frames.append(candidates)

    if include_closing_prices:
        for source, price_column in CLOSING_PRICE_SCENARIOS.items():
            valid_price = predictions[price_column].notna() & predictions[
                price_column
            ].gt(1)
            candidates = predictions.loc[valid_price].copy()
            candidates["execution_source"] = source
            candidates["commission"] = 0.0
            candidates["quoted_odds"] = candidates[price_column]
            candidates["quoted_closing_odds"] = np.nan
            candidates["bet_odds"] = candidates["quoted_odds"]
            candidates["closing_odds"] = np.nan
            frames.append(candidates)

    result = pd.concat(frames, ignore_index=True)
    result["signal_probability"] = result[probability_column]
    result["expected_value"] = (
        result["signal_probability"] * result["bet_odds"] - 1
    )
    result["profit"] = np.where(
        result["over_2_5"].eq(1), result["bet_odds"] - 1, -1.0
    )
    result["clv_pct"] = (
        result["bet_odds"] / result["closing_odds"] - 1
    ) * 100
    return result.sort_values(
        ["execution_source", "league", "season", "date"], kind="stable"
    ).reset_index(drop=True)


def ev_bucket_summary(
    candidates: pd.DataFrame,
    *,
    edges: list[float] = EV_BUCKET_EDGES,
    labels: list[str] = EV_BUCKET_LABELS,
) -> pd.DataFrame:
    """Summarize predicted value, realized ROI, and CLV by fixed EV bucket."""
    if len(edges) != len(labels) + 1:
        raise ValueError("EV bucket edges must contain one more value than labels")
    required = {
        "execution_source",
        "expected_value",
        "profit",
        "over_2_5",
        "bet_odds",
        "closing_odds",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(
            "EV bucket summary is missing columns: " + ", ".join(sorted(missing))
        )

    frame = candidates.copy()
    frame["ev_bucket"] = pd.cut(
        frame["expected_value"],
        bins=edges,
        labels=labels,
        right=False,
        ordered=True,
    )
    summary = (
        frame.groupby(
            ["execution_source", "ev_bucket"], observed=True, sort=False
        )
        .agg(
            candidates=("profit", "size"),
            wins=("over_2_5", "sum"),
            mean_estimated_ev=("expected_value", "mean"),
            average_odds=("bet_odds", "mean"),
            profit_units=("profit", "sum"),
            realized_roi=("profit", "mean"),
            clv_observations=("closing_odds", "count"),
            mean_clv_pct=("clv_pct", "mean"),
        )
        .reset_index()
    )
    summary["win_rate_pct"] = summary["wins"] / summary["candidates"] * 100
    summary["mean_estimated_ev_pct"] = summary["mean_estimated_ev"] * 100
    summary["realized_roi_pct"] = summary["realized_roi"] * 100
    return summary.drop(columns=["mean_estimated_ev", "realized_roi"])


def _maximum_drawdown(profit: np.ndarray) -> float:
    cumulative = np.cumsum(profit)
    running_peak = np.maximum.accumulate(np.r_[0.0, cumulative])
    return float(np.max(running_peak[1:] - cumulative))


def summarize_bets(bets: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    """Summarize flat one-unit stakes with approximate 95% ROI intervals."""
    rows = []
    for keys, group in bets.groupby(group_by, sort=True, observed=True):
        if "date" in group.columns:
            group = group.sort_values("date", kind="stable")
        keys = keys if isinstance(keys, tuple) else (keys,)
        profit = group["profit"].to_numpy(float)
        roi = profit.mean()
        standard_error = (
            profit.std(ddof=1) / np.sqrt(len(profit))
            if len(profit) > 1
            else np.nan
        )
        row = dict(zip(group_by, keys))
        row.update(
            {
                "bets": len(group),
                "wins": int(group["over_2_5"].sum()),
                "win_rate_pct": group["over_2_5"].mean() * 100,
                "average_odds": group["bet_odds"].mean(),
                "profit_units": profit.sum(),
                "roi_pct": roi * 100,
                "roi_95_low_pct": (roi - 1.96 * standard_error) * 100,
                "roi_95_high_pct": (roi + 1.96 * standard_error) * 100,
                "mean_clv_pct": (
                    group["bet_odds"] / group["closing_odds"] - 1
                ).mean()
                * 100,
                "max_drawdown_units": _maximum_drawdown(profit),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def monthly_roi(bets: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
    """Aggregate flat-stake realized ROI by calendar month.

    ROI is total profit divided by total one-unit stakes placed during the
    month. Months with no bets are absent rather than represented as zero.
    ``group_by`` controls whether results are separated by league, execution
    source, or both.
    """
    required = {"date", "profit", "over_2_5", *group_by}
    missing = required.difference(bets.columns)
    if missing:
        raise ValueError(
            "Monthly ROI is missing columns: " + ", ".join(sorted(missing))
        )
    if bets.empty:
        return pd.DataFrame(
            columns=[
                *group_by,
                "month",
                "bets",
                "wins",
                "staked_units",
                "profit_units",
                "roi_pct",
            ]
        )

    frame = bets.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp()

    monthly = (
        frame.groupby([*group_by, "month"], sort=True, observed=True)
        .agg(
            bets=("profit", "size"),
            wins=("over_2_5", "sum"),
            profit_units=("profit", "sum"),
        )
        .reset_index()
    )
    monthly["staked_units"] = monthly["bets"].astype(float)
    monthly["roi_pct"] = (
        monthly["profit_units"] / monthly["staked_units"] * 100
    )
    return monthly[
        [
            *group_by,
            "month",
            "bets",
            "wins",
            "staked_units",
            "profit_units",
            "roi_pct",
        ]
    ]
