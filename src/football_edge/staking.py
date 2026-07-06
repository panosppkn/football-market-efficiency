"""Market-consensus signals and path-dependent bankroll experiments."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from football_edge.backtest import add_market_features, build_execution_candidates
from football_edge.config import CONSENSUS_KELLY_MINIMUM_EXPECTED_VALUE


CONSENSUS_EXECUTION_SOURCES = (
    "bet365",
    "pinnacle",
    "betfair_exchange",
    "best_preclosing",
    "best_closing",
)

# Prespecified sensitivity set. The quarter-Kelly variant is the primary
# specification; the others show how estimation error is amplified by leverage.
DEFAULT_KELLY_CONFIGURATIONS = {
    "10% Kelly, 1% cap": (0.10, 0.01),
    "25% Kelly, 1% cap": (0.25, 0.01),
    "50% Kelly, 2% cap": (0.50, 0.02),
    "Full Kelly, 5% cap": (1.00, 0.05),
}


def build_consensus_candidates(
    matches: pd.DataFrame,
    *,
    minimum_expected_value: float = CONSENSUS_KELLY_MINIMUM_EXPECTED_VALUE,
) -> pd.DataFrame:
    """Select value bets using the no-vig average market probability.

    Average pre-closing over and under odds determine the consensus probability.
    They are never used as the execution price. Named bookmakers, the
    pre-closing market maximum, and the closing maximum are evaluated
    separately. The closing maximum is explicitly marked non-executable.
    """
    if minimum_expected_value < 0:
        raise ValueError("minimum_expected_value must be non-negative")

    frame = add_market_features(matches)
    candidates = build_execution_candidates(
        frame,
        include_closing_prices=True,
        probability_column="market_over_probability",
    )
    candidates = candidates.loc[
        candidates["execution_source"].isin(CONSENSUS_EXECUTION_SOURCES)
        & candidates["expected_value"].ge(minimum_expected_value)
    ].copy()
    candidates["probability_source"] = "no_vig_average_preclosing"
    candidates["is_executable_proxy"] = candidates["execution_source"].ne(
        "best_closing"
    )
    return candidates.reset_index(drop=True)


def full_kelly_fraction(
    probability: pd.Series | np.ndarray,
    decimal_odds: pd.Series | np.ndarray,
) -> np.ndarray:
    """Return the non-negative binary-outcome Kelly fraction."""
    probability = np.asarray(probability, dtype=float)
    decimal_odds = np.asarray(decimal_odds, dtype=float)
    if np.any((probability <= 0) | (probability >= 1)):
        raise ValueError("probabilities must lie strictly between zero and one")
    if np.any(decimal_odds <= 1):
        raise ValueError("decimal odds must be greater than one")
    return np.maximum(
        (probability * decimal_odds - 1) / (decimal_odds - 1),
        0.0,
    )


def simulate_kelly_bankroll(
    bets: pd.DataFrame,
    *,
    configurations: Mapping[str, tuple[float, float]] = (
        DEFAULT_KELLY_CONFIGURATIONS
    ),
    initial_bankroll: float = 100.0,
) -> pd.DataFrame:
    """Simulate independent bankrolls by source and Kelly configuration.

    Bets sharing an exact timestamp are sized from the same pre-event bankroll
    and settled together, avoiding artificial within-timestamp compounding.
    If simultaneous stakes would exceed available capital, all stakes in that
    batch are scaled proportionally.
    """
    required = {
        "date",
        "execution_source",
        "signal_probability",
        "bet_odds",
        "profit",
    }
    missing = required.difference(bets.columns)
    if missing:
        raise ValueError(
            "Kelly simulation is missing columns: "
            + ", ".join(sorted(missing))
        )
    if initial_bankroll <= 0:
        raise ValueError("initial_bankroll must be positive")

    for name, (multiplier, cap) in configurations.items():
        if not 0 < multiplier <= 1:
            raise ValueError(f"{name}: Kelly multiplier must be in (0, 1]")
        if not 0 < cap <= 1:
            raise ValueError(f"{name}: stake cap must be in (0, 1]")

    frames = []
    for source, source_bets in bets.groupby("execution_source", sort=True):
        source_bets = source_bets.sort_values("date", kind="stable")
        for strategy, (multiplier, cap) in configurations.items():
            bankroll = float(initial_bankroll)
            rows = []
            for timestamp, batch in source_bets.groupby("date", sort=True):
                batch = batch.copy()
                full_fraction = full_kelly_fraction(
                    batch["signal_probability"], batch["bet_odds"]
                )
                stake_fraction = np.minimum(multiplier * full_fraction, cap)
                total_fraction = stake_fraction.sum()
                capital_scale = min(1.0, 1.0 / total_fraction)
                stake_fraction *= capital_scale
                stakes = bankroll * stake_fraction
                pnl = stakes * batch["profit"].to_numpy(float)
                bankroll_after = bankroll + pnl.sum()

                batch["strategy"] = strategy
                batch["kelly_multiplier"] = multiplier
                batch["stake_cap"] = cap
                batch["full_kelly_fraction"] = full_fraction
                batch["stake_fraction"] = stake_fraction
                batch["stake"] = stakes
                batch["pnl"] = pnl
                batch["bankroll_before"] = bankroll
                batch["bankroll_after_batch"] = bankroll_after
                batch["capital_scale"] = capital_scale
                rows.append(batch)
                bankroll = bankroll_after
            if rows:
                frames.append(pd.concat(rows, ignore_index=True))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values(
        ["execution_source", "strategy", "date"], kind="stable"
    ).reset_index(drop=True)


def summarize_kelly_paths(
    paths: pd.DataFrame,
    *,
    initial_bankroll: float = 100.0,
) -> pd.DataFrame:
    """Summarize growth, drawdown, staking intensity, and concentration."""
    required = {
        "execution_source",
        "strategy",
        "date",
        "stake",
        "pnl",
        "bankroll_after_batch",
        "stake_fraction",
    }
    missing = required.difference(paths.columns)
    if missing:
        raise ValueError(
            "Kelly path summary is missing columns: "
            + ", ".join(sorted(missing))
        )

    rows = []
    for (source, strategy), group in paths.groupby(
        ["execution_source", "strategy"], sort=True
    ):
        by_time = (
            group.groupby("date", sort=True)
            .agg(
                pnl=("pnl", "sum"),
                bankroll=("bankroll_after_batch", "last"),
            )
            .reset_index()
        )
        wealth = np.r_[initial_bankroll, by_time["bankroll"].to_numpy(float)]
        peaks = np.maximum.accumulate(wealth)
        drawdowns = 1 - wealth / peaks
        total_staked = group["stake"].sum()
        final_bankroll = wealth[-1]
        rows.append(
            {
                "execution_source": source,
                "strategy": strategy,
                "bets": len(group),
                "final_bankroll": final_bankroll,
                "bankroll_return_pct": (
                    final_bankroll / initial_bankroll - 1
                )
                * 100,
                "total_staked": total_staked,
                "profit": group["pnl"].sum(),
                "turnover": total_staked / initial_bankroll,
                "return_on_stakes_pct": (
                    group["pnl"].sum() / total_staked * 100
                    if total_staked > 0
                    else np.nan
                ),
                "max_drawdown_pct": drawdowns.max() * 100,
                "average_stake_pct": group["stake_fraction"].mean() * 100,
                "maximum_stake_pct": group["stake_fraction"].max() * 100,
                "largest_bet_share_pct": (
                    group["stake"].max() / total_staked * 100
                    if total_staked > 0
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def cluster_bootstrap_roi(
    bets: pd.DataFrame,
    *,
    n_bootstrap: int = 2000,
    random_seed: int = 42,
    minimum_clusters_for_inference: int = 30,
) -> pd.DataFrame:
    """Estimate flat-stake ROI uncertainty by resampling match-date clusters.

    Resampling dates rather than individual bets preserves same-day dependence
    across leagues within each execution source. The procedure diagnoses
    sampling uncertainty; it does not resolve quote-timing or selection bias.
    """
    required = {"date", "execution_source", "profit"}
    missing = required.difference(bets.columns)
    if missing:
        raise ValueError(
            "Cluster bootstrap is missing columns: "
            + ", ".join(sorted(missing))
        )
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    if minimum_clusters_for_inference < 2:
        raise ValueError("minimum_clusters_for_inference must be at least two")

    rng = np.random.default_rng(random_seed)
    rows = []
    for source, group in bets.groupby("execution_source", sort=True):
        clusters = (
            group.assign(match_date=pd.to_datetime(group["date"]).dt.normalize())
            .groupby("match_date", sort=True)
            .agg(profit=("profit", "sum"), bets=("profit", "size"))
        )
        values = clusters[["profit", "bets"]].to_numpy(float)
        indices = rng.integers(
            0, len(values), size=(n_bootstrap, len(values))
        )
        samples = values[indices].sum(axis=1)
        roi = samples[:, 0] / samples[:, 1]
        sufficient_clusters = len(values) >= minimum_clusters_for_inference
        rows.append(
            {
                "execution_source": source,
                "date_clusters": len(values),
                "bets": len(group),
                "sufficient_clusters_for_inference": sufficient_clusters,
                "observed_roi_pct": group["profit"].mean() * 100,
                "bootstrap_mean_roi_pct": roi.mean() * 100,
                "roi_95_low_pct": (
                    np.quantile(roi, 0.025) * 100
                    if sufficient_clusters
                    else np.nan
                ),
                "roi_95_high_pct": (
                    np.quantile(roi, 0.975) * 100
                    if sufficient_clusters
                    else np.nan
                ),
                "probability_positive_roi_pct": (
                    np.mean(roi > 0) * 100
                    if sufficient_clusters
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)
