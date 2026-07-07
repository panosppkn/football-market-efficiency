"""Two-sided Over/Under 2.5 execution and flat-stake evaluation."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from football_edge.config import MINIMUM_EXPECTED_VALUE


TWO_SIDED_EXECUTION_SCENARIOS = {
    "average_preclosing": {
        "over_price": "Avg>2.5",
        "under_price": "Avg<2.5",
        "over_close": "AvgC>2.5",
        "under_close": "AvgC<2.5",
        "commission": 0.0,
        "is_executable_proxy": True,
    },
    "best_preclosing": {
        "over_price": "Max>2.5",
        "under_price": "Max<2.5",
        "over_close": "MaxC>2.5",
        "under_close": "MaxC<2.5",
        "commission": 0.0,
        "is_executable_proxy": True,
    },
    "bet365": {
        "over_price": "B365>2.5",
        "under_price": "B365<2.5",
        "over_close": "B365C>2.5",
        "under_close": "B365C<2.5",
        "commission": 0.0,
        "is_executable_proxy": True,
    },
    "pinnacle": {
        "over_price": "P>2.5",
        "under_price": "P<2.5",
        "over_close": "PC>2.5",
        "under_close": "PC<2.5",
        "commission": 0.0,
        "is_executable_proxy": True,
    },
    "betfair_exchange": {
        "over_price": "BFE>2.5",
        "under_price": "BFE<2.5",
        "over_close": "BFEC>2.5",
        "under_close": "BFEC<2.5",
        "commission": 0.0,
        "is_executable_proxy": True,
    },
    "best_closing": {
        "over_price": "MaxC>2.5",
        "under_price": "MaxC<2.5",
        "over_close": None,
        "under_close": None,
        "commission": 0.0,
        "is_executable_proxy": False,
    },
}


def _net_decimal_odds(odds: pd.Series, commission: float) -> pd.Series:
    return 1 + (odds - 1) * (1 - commission)


def build_two_sided_candidates(
    predictions: pd.DataFrame,
    *,
    probability_column: str = "model_probability",
    include_closing_maximum: bool = False,
    sources: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build valid Over and Under candidates from one Over probability.

    The Under probability is exactly ``1 - p_over``. Candidate eligibility uses
    only the side-specific execution quote; a missing closing quote affects CLV
    coverage but never removes an otherwise valid candidate.
    """
    required = {probability_column, "over_2_5", "date"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(
            "Two-sided candidates are missing columns: "
            + ", ".join(sorted(missing))
        )

    probability = predictions[probability_column]
    if probability.isna().any() or not probability.between(0, 1).all():
        raise ValueError("Prediction probabilities must be non-missing and in [0, 1]")

    requested_sources = (
        list(sources)
        if sources is not None
        else list(TWO_SIDED_EXECUTION_SCENARIOS)
    )
    unknown = set(requested_sources).difference(TWO_SIDED_EXECUTION_SCENARIOS)
    if unknown:
        raise ValueError("Unknown execution sources: " + ", ".join(sorted(unknown)))
    if not include_closing_maximum:
        requested_sources = [
            source for source in requested_sources if source != "best_closing"
        ]

    base = predictions.copy()
    base["match_row_id"] = np.arange(len(base))
    frames = []
    for source in requested_sources:
        scenario = TWO_SIDED_EXECUTION_SCENARIOS[source]
        commission = float(scenario["commission"])
        for side in ("over", "under"):
            price_column = scenario[f"{side}_price"]
            close_column = scenario[f"{side}_close"]
            if price_column not in base.columns:
                continue

            valid_price = base[price_column].notna() & base[price_column].gt(1)
            candidates = base.loc[valid_price].copy()
            if candidates.empty:
                continue

            candidates["execution_source"] = source
            candidates["bet_side"] = side
            candidates["commission"] = commission
            candidates["is_executable_proxy"] = bool(
                scenario["is_executable_proxy"]
            )
            candidates["quoted_odds"] = candidates[price_column]
            candidates["bet_odds"] = _net_decimal_odds(
                candidates["quoted_odds"], commission
            )

            if close_column is not None and close_column in candidates.columns:
                valid_close = (
                    candidates[close_column].notna()
                    & candidates[close_column].gt(1)
                )
                candidates["quoted_closing_odds"] = candidates[
                    close_column
                ].where(valid_close)
                candidates["closing_odds"] = _net_decimal_odds(
                    candidates["quoted_closing_odds"], commission
                )
            else:
                candidates["quoted_closing_odds"] = np.nan
                candidates["closing_odds"] = np.nan

            if side == "over":
                candidates["signal_probability"] = candidates[
                    probability_column
                ]
                candidates["bet_won"] = candidates["over_2_5"].eq(1)
            else:
                candidates["signal_probability"] = 1 - candidates[
                    probability_column
                ]
                candidates["bet_won"] = candidates["over_2_5"].eq(0)

            candidates["expected_value"] = (
                candidates["signal_probability"] * candidates["bet_odds"] - 1
            )
            candidates["profit"] = np.where(
                candidates["bet_won"], candidates["bet_odds"] - 1, -1.0
            )
            candidates["clv_pct"] = (
                candidates["bet_odds"] / candidates["closing_odds"] - 1
            ) * 100
            frames.append(candidates)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["execution_source", "match_row_id", "bet_side"], kind="stable"
    ).reset_index(drop=True)


def select_two_sided_bets(
    candidates: pd.DataFrame,
    *,
    minimum_expected_value: float = MINIMUM_EXPECTED_VALUE,
    policy: str = "best_side",
) -> pd.DataFrame:
    """Apply a fixed EV threshold and select at most one side per source-match.

    ``policy`` may be ``over_only``, ``under_only``, or ``best_side``. For the
    combined policy, the qualifying side with higher estimated EV is selected.
    Exact EV ties are resolved in favor of Over only to make the rule
    deterministic; ties should be reported as quote-quality diagnostics.
    """
    required = {
        "execution_source",
        "match_row_id",
        "bet_side",
        "expected_value",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(
            "Two-sided selection is missing columns: "
            + ", ".join(sorted(missing))
        )
    if minimum_expected_value < 0:
        raise ValueError("minimum_expected_value must be non-negative")
    if policy not in {"over_only", "under_only", "best_side"}:
        raise ValueError("policy must be over_only, under_only, or best_side")

    eligible = candidates.loc[
        candidates["expected_value"].ge(minimum_expected_value)
    ].copy()
    group_columns = ["execution_source", "match_row_id"]
    eligible["qualifying_sides"] = eligible.groupby(group_columns)[
        "bet_side"
    ].transform("size")
    eligible["both_sides_qualify"] = eligible["qualifying_sides"].gt(1)

    if policy != "best_side":
        side = policy.removesuffix("_only")
        selected = eligible.loc[eligible["bet_side"].eq(side)].copy()
    else:
        eligible["side_priority"] = eligible["bet_side"].map(
            {"over": 0, "under": 1}
        )
        selected = (
            eligible.sort_values(
                [
                    "execution_source",
                    "match_row_id",
                    "expected_value",
                    "side_priority",
                ],
                ascending=[True, True, False, True],
                kind="stable",
            )
            .drop_duplicates(group_columns, keep="first")
            .drop(columns="side_priority")
        )

    selected["selection_policy"] = policy
    return selected.sort_values(
        ["execution_source", "date", "match_row_id"], kind="stable"
    ).reset_index(drop=True)


def summarize_two_sided_bets(
    bets: pd.DataFrame,
    group_by: list[str],
) -> pd.DataFrame:
    """Summarize flat one-unit two-sided bets with approximate ROI intervals."""
    required = {"profit", "bet_won", "bet_odds", "closing_odds", *group_by}
    missing = required.difference(bets.columns)
    if missing:
        raise ValueError(
            "Two-sided summary is missing columns: "
            + ", ".join(sorted(missing))
        )

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
        cumulative = np.r_[0.0, np.cumsum(profit)]
        running_peak = np.maximum.accumulate(cumulative)
        row = dict(zip(group_by, keys))
        row.update(
            {
                "bets": len(group),
                "wins": int(group["bet_won"].sum()),
                "win_rate_pct": group["bet_won"].mean() * 100,
                "average_odds": group["bet_odds"].mean(),
                "profit_units": profit.sum(),
                "roi_pct": roi * 100,
                "roi_95_low_pct": (roi - 1.96 * standard_error) * 100,
                "roi_95_high_pct": (roi + 1.96 * standard_error) * 100,
                "mean_clv_pct": (
                    group["bet_odds"] / group["closing_odds"] - 1
                ).mean()
                * 100,
                "max_drawdown_units": np.max(running_peak - cumulative),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
