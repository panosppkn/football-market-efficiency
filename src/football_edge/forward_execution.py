"""Forward execution simulation for finalized Polymarket transaction files.

The functions here consume outputs created by the public monthly collection
workflow.  They do not query Dune, discover markets, select Football-Data
fixtures, or estimate model parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_edge.forward_polymarket import MATCH_CACHE_KEY, TRADE_IDENTITY_KEY
from football_edge.staking import full_kelly_fraction


@dataclass(frozen=True)
class ExecutionConfig:
    """Frozen assumptions for the transaction-based execution simulation."""

    initial_capital_usd: float = 10_000.0
    flat_stake_usd: float = 100.0
    kelly_fractions: tuple[float, ...] = (0.10, 0.25)
    buyer_fee_rate: float = 0.05
    minimum_odds_premium: float = 0.01
    allow_partial_fills: bool = True
    minimum_fill_fraction: float = 0.0
    maximum_trade_participation: float = 0.50
    maximum_kelly_stake_fraction: float = 0.05
    maximum_open_exposure_fraction: float = 1.0
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
        if not 0 <= self.buyer_fee_rate <= 0.20:
            raise ValueError("buyer_fee_rate must be between 0 and 0.20")
        if not 0 <= self.minimum_odds_premium <= 1:
            raise ValueError("minimum_odds_premium must be between 0 and 1")
        if not self.allow_partial_fills:
            raise ValueError("This execution model is defined for partial fills")
        if not 0 <= self.minimum_fill_fraction <= 1:
            raise ValueError("minimum_fill_fraction must be between 0 and 1")
        if not 0 < self.maximum_trade_participation <= 1:
            raise ValueError("maximum_trade_participation must be in (0, 1]")
        if not 0 < self.maximum_kelly_stake_fraction <= 1:
            raise ValueError("maximum_kelly_stake_fraction must be in (0, 1]")
        if not 0 < self.maximum_open_exposure_fraction <= 1:
            raise ValueError("maximum_open_exposure_fraction must be in (0, 1]")
        if self.assumed_match_duration_hours <= 0:
            raise ValueError("assumed_match_duration_hours must be positive")


@dataclass(frozen=True)
class HoldoutCache:
    """All finalized collection files discovered for one season."""

    eligible_trades: pd.DataFrame
    match_summary: pd.DataFrame
    eligible_trade_files: tuple[Path, ...]
    match_summary_files: tuple[Path, ...]


@dataclass(frozen=True)
class StrategySimulation:
    """Bet, event-path, and fill-level outputs for a set of strategies."""

    bets: pd.DataFrame
    bankroll_paths: pd.DataFrame
    fills: pd.DataFrame


@dataclass(frozen=True)
class ExecutionEvaluation:
    """Simulation and return summaries for one portfolio definition."""

    simulation: StrategySimulation
    weekly_returns: pd.DataFrame
    monthly_returns: pd.DataFrame
    metrics: pd.DataFrame


DATETIME_COLUMNS = (
    "match_date",
    "date",
    "football_data_release_utc",
    "football_data_release_uk_time",
    "kickoff_utc",
    "kickoff_uk_time",
    "dune_trade_time_utc",
    "dune_trade_time_uk",
    "first_eligible_trade_uk",
    "last_eligible_trade_uk",
)

NUMERIC_COLUMNS = (
    "football_data_max_home_odds",
    "p_fd",
    "q_hat_fd",
    "dune_home_win_price",
    "dune_price_from_amount",
    "dune_raw_decimal_odds",
    "dune_recorded_taker_buy_effective_odds",
    "dune_trade_amount_usd",
    "dune_trade_shares",
    "dune_recorded_fee_usd",
    "eligible_trade_rows",
    "evt_index",
    "home_goals",
    "away_goals",
    "home_win_result",
)


def _read_cache_file(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(
            path,
            dtype={
                "condition_id": "string",
                "asset_id": "string",
                "tx_hash": "string",
                "order_hash": "string",
                "unique_key": "string",
            },
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    frame["cache_file"] = str(path)
    return frame


def _load_cache_files(files: tuple[Path, ...]) -> pd.DataFrame:
    frames = [_read_cache_file(path) for path in files]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame()
    output = pd.concat(frames, ignore_index=True, sort=False)
    for column in DATETIME_COLUMNS:
        if column in output:
            output[column] = pd.to_datetime(output[column], utc=True, errors="coerce")
    for column in NUMERIC_COLUMNS:
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def load_finalized_holdout_cache(
    cache_dir: str | Path,
    *,
    season: str,
    source: str = "market_maximum",
    market_outcome: str = "home_win",
) -> HoldoutCache:
    """Discover and load every finalized league-month file for one season."""
    cache_dir = Path(cache_dir)
    suffix = f"{source}_{market_outcome}"
    eligible_files = tuple(sorted(cache_dir.glob(f"*_{season}_*_{suffix}_eligible_trades.csv")))
    summary_files = tuple(sorted(cache_dir.glob(f"*_{season}_*_{suffix}_match_summary.csv")))
    eligible = _load_cache_files(eligible_files)
    summary = _load_cache_files(summary_files)

    if not summary.empty:
        required = ["season", "league", "home_team", "away_team"]
        date_column = "match_date" if "match_date" in summary else "date"
        duplicate_key = [date_column, *required]
        duplicates = summary.duplicated(subset=duplicate_key, keep=False)
        if duplicates.any():
            raise ValueError(
                "Finalized match-summary files overlap: "
                f"{int(duplicates.sum())} duplicate match rows were found"
            )
    return HoldoutCache(
        eligible_trades=eligible,
        match_summary=summary,
        eligible_trade_files=eligible_files,
        match_summary_files=summary_files,
    )


def cache_coverage_by_league(cache: HoldoutCache) -> pd.DataFrame:
    """Summarize collection coverage independently of execution assumptions."""
    summary = cache.match_summary
    if summary.empty:
        return pd.DataFrame()
    summary = summary.copy()
    if "eligible_trade_rows" not in summary:
        summary["eligible_trade_rows"] = 0
    return (
        summary.groupby("league", sort=True)
        .agg(
            football_data_selected_matches=("home_team", "size"),
            polymarket_markets_resolved=(
                "market_resolution_status",
                lambda values: int(pd.Series(values).eq("resolved").sum()),
            ),
            matches_with_base_eligible_trades=(
                "eligible_trade_rows",
                lambda values: int(pd.to_numeric(values, errors="coerce").fillna(0).gt(0).sum()),
            ),
            base_eligible_trade_rows=("eligible_trade_rows", "sum"),
        )
        .reset_index()
    )


def holdout_evaluation_period(cache: HoldoutCache) -> dict[str, Any]:
    """Return concise file and match coverage metadata."""
    summary = cache.match_summary
    eligible = cache.eligible_trades
    date_source = summary if not summary.empty else eligible
    date_column = "match_date" if "match_date" in date_source else "date"
    dates = (
        pd.to_datetime(date_source[date_column], utc=True, errors="coerce").dropna()
        if not date_source.empty and date_column in date_source
        else pd.Series(dtype="datetime64[ns, UTC]")
    )
    return {
        "evaluation_start": dates.min().date().isoformat() if not dates.empty else None,
        "evaluation_end": dates.max().date().isoformat() if not dates.empty else None,
        "last_included_month": dates.max().strftime("%Y-%m") if not dates.empty else None,
        "eligible_trade_files": len(cache.eligible_trade_files),
        "match_summary_files": len(cache.match_summary_files),
        "football_data_selected_matches": len(summary),
        "cached_base_eligible_trade_rows": len(eligible),
    }


def parse_boolean(series: pd.Series) -> pd.Series:
    """Parse booleans written by pandas CSV serialization."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def raw_price_for_all_in_limit(
    all_in_price: pd.Series | np.ndarray,
    fee_rate: float,
) -> pd.Series:
    """Invert ``p + fee_rate * p * (1-p)`` for an all-in price limit."""
    target = pd.Series(pd.to_numeric(all_in_price, errors="coerce"), copy=False).astype(float)
    if fee_rate == 0:
        return target
    discriminant = (1.0 + fee_rate) ** 2 - 4.0 * fee_rate * target
    if (discriminant.dropna() < -1e-12).any():
        raise ValueError("The all-in price is incompatible with the configured fee curve")
    return ((1.0 + fee_rate) - np.sqrt(discriminant.clip(lower=0.0))) / (2.0 * fee_rate)


def prepare_execution_trades(
    trades: pd.DataFrame,
    *,
    config: ExecutionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct auditable opportunities under the frozen execution policy."""
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()
    output = trades.copy()
    required = [
        *MATCH_CACHE_KEY,
        *TRADE_IDENTITY_KEY,
        "is_taker_side",
        "maker_side",
        "taker_side",
        "football_data_release_utc",
        "kickoff_uk_time",
        "dune_trade_time_utc",
        "football_data_max_home_odds",
        "q_hat_fd",
        "dune_home_win_price",
        "dune_trade_amount_usd",
        "dune_trade_shares",
        "dune_recorded_fee_usd",
        "home_win_result",
    ]
    missing = [column for column in required if column not in output]
    if missing:
        raise KeyError(f"Eligible-trade cache is missing required columns: {missing}")
    for column in ["football_data_release_utc", "kickoff_uk_time", "dune_trade_time_utc"]:
        output[column] = pd.to_datetime(output[column], utc=True, errors="coerce")
    for column in [
        "dune_home_win_price",
        "dune_trade_amount_usd",
        "dune_trade_shares",
        "dune_recorded_fee_usd",
        "football_data_max_home_odds",
        "q_hat_fd",
        "home_win_result",
        "evt_index",
    ]:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    output["is_taker_side"] = parse_boolean(output["is_taker_side"])
    if not output["is_taker_side"].all():
        raise ValueError("Eligible cache contains a noncanonical Dune row")
    output["maker_side"] = output["maker_side"].astype(str).str.upper()
    output["taker_side"] = output["taker_side"].astype(str).str.upper()

    before_deduplication = len(output)
    output = (
        output.sort_values(["dune_trade_time_utc", *TRADE_IDENTITY_KEY], kind="stable")
        .drop_duplicates(subset=TRADE_IDENTITY_KEY, keep="first")
        .reset_index(drop=True)
    )
    output["observed_raw_price"] = output["dune_trade_amount_usd"] / output["dune_trade_shares"]
    output["price_reconstruction_error"] = (
        output["observed_raw_price"] - output["dune_home_win_price"]
    ).abs()
    output["minimum_required_odds"] = (
        output["football_data_max_home_odds"] * (1.0 + config.minimum_odds_premium)
    )
    output["predefined_max_effective_price"] = 1.0 / output["minimum_required_odds"]
    output["predefined_raw_limit_price"] = raw_price_for_all_in_limit(
        output["predefined_max_effective_price"], config.buyer_fee_rate
    ).to_numpy()

    seller_initiated = output["taker_side"].eq("SELL") & output["maker_side"].eq("BUY")
    buyer_initiated = output["taker_side"].eq("BUY") & output["maker_side"].eq("SELL")
    output["direction_pair_valid"] = seller_initiated | buyer_initiated
    if not output["direction_pair_valid"].all():
        raise ValueError("Eligible cache contains an inconsistent maker/taker side pair")
    output["historical_trade_type"] = np.select(
        [seller_initiated, buyer_initiated],
        ["taker_sell__maker_buy", "taker_buy__maker_sell"],
        default="unsupported_direction_pair",
    )
    output["execution_raw_price"] = np.where(
        seller_initiated,
        output["predefined_raw_limit_price"],
        output["observed_raw_price"],
    )
    output["execution_price_source"] = np.where(
        seller_initiated,
        "predefined_limit",
        "observed_trade_price",
    )
    output["hypothetical_fee_per_share"] = (
        config.buyer_fee_rate
        * output["execution_raw_price"]
        * (1.0 - output["execution_raw_price"])
    )
    output["hypothetical_buyer_fee_usd"] = (
        output["dune_trade_shares"] * output["hypothetical_fee_per_share"]
    )
    output["hypothetical_raw_cost_usd"] = (
        output["dune_trade_shares"] * output["execution_raw_price"]
    )
    output["hypothetical_total_cost_usd"] = (
        output["hypothetical_raw_cost_usd"] + output["hypothetical_buyer_fee_usd"]
    )
    output["hypothetical_effective_price"] = (
        output["hypothetical_total_cost_usd"] / output["dune_trade_shares"]
    )
    output["hypothetical_effective_odds"] = (
        output["dune_trade_shares"] / output["hypothetical_total_cost_usd"]
    )
    np.testing.assert_allclose(
        output["hypothetical_total_cost_usd"],
        output["hypothetical_raw_cost_usd"] + output["hypothetical_buyer_fee_usd"],
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    if seller_initiated.any():
        np.testing.assert_allclose(
            output.loc[seller_initiated, "hypothetical_effective_price"],
            output.loc[seller_initiated, "predefined_max_effective_price"],
            rtol=1e-10,
            atol=1e-12,
        )

    valid_window = (
        output["dune_trade_time_utc"].notna()
        & output["football_data_release_utc"].notna()
        & output["kickoff_uk_time"].notna()
        & output["dune_trade_time_utc"].ge(output["football_data_release_utc"])
        & output["dune_trade_time_utc"].lt(output["kickoff_uk_time"])
    )
    valid_trade = (
        output["observed_raw_price"].between(0.0, 1.0, inclusive="neither")
        & output["execution_raw_price"].between(0.0, 1.0, inclusive="neither")
        & output["dune_trade_amount_usd"].gt(0)
        & output["dune_trade_shares"].gt(0)
        & output["hypothetical_buyer_fee_usd"].ge(0)
        & output["hypothetical_total_cost_usd"].gt(0)
    )
    output["price_evidence_eligible"] = np.select(
        [seller_initiated, buyer_initiated],
        [
            output["observed_raw_price"].le(output["predefined_raw_limit_price"] + 1e-12),
            output["hypothetical_effective_price"].le(
                output["predefined_max_effective_price"] + 1e-12
            ),
        ],
        default=False,
    ).astype(bool)
    output["execution_eligible"] = (
        valid_window
        & valid_trade
        & output["direction_pair_valid"]
        & output["price_evidence_eligible"]
        & output["hypothetical_effective_odds"].ge(output["minimum_required_odds"] - 1e-12)
    )
    diagnostics = (
        output.groupby(
            [
                "match_date",
                "league",
                "home_team",
                "away_team",
                "football_data_max_home_odds",
                "condition_id",
            ],
            sort=True,
            dropna=False,
        )
        .agg(
            canonical_trade_rows=("tx_hash", "size"),
            taker_sell_maker_buy_rows=(
                "historical_trade_type",
                lambda values: int(pd.Series(values).eq("taker_sell__maker_buy").sum()),
            ),
            taker_buy_maker_sell_rows=(
                "historical_trade_type",
                lambda values: int(pd.Series(values).eq("taker_buy__maker_sell").sum()),
            ),
            price_evidence_eligible_rows=("price_evidence_eligible", "sum"),
            execution_eligible_rows=("execution_eligible", "sum"),
            maximum_price_reconstruction_error=("price_reconstruction_error", "max"),
        )
        .reset_index()
    )
    diagnostics["duplicate_event_rows_removed"] = before_deduplication - len(output)
    return output, diagnostics


def _match_identifier(row: pd.Series) -> tuple[Any, ...]:
    return tuple(row[column] for column in MATCH_CACHE_KEY)


def target_stake_for_strategy(
    order_row: pd.Series,
    *,
    kelly_fraction: float | None,
    current_equity: float,
    config: ExecutionConfig,
) -> tuple[float, float]:
    """Set stake once at order placement using only frozen-rule information."""
    if kelly_fraction is None:
        return config.flat_stake_usd, config.flat_stake_usd / current_equity
    full_kelly = float(
        full_kelly_fraction(
            np.array([float(order_row["q_hat_fd"])]),
            np.array([float(order_row["minimum_required_odds"])]),
        )[0]
    )
    stake_fraction = min(
        kelly_fraction * full_kelly,
        config.maximum_kelly_stake_fraction,
    )
    return current_equity * stake_fraction, stake_fraction


def simulate_strategy(
    trades: pd.DataFrame,
    *,
    strategy: str,
    kelly_fraction: float | None,
    config: ExecutionConfig,
) -> StrategySimulation:
    """Process orders, transaction evidence, fills, and settlements causally."""
    if trades.empty:
        return StrategySimulation(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    settled_trades = trades.loc[
        trades["home_win_result"].isin([0.0, 1.0]) & trades["execution_eligible"]
    ].copy()
    if settled_trades.empty:
        return StrategySimulation(pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    settled_trades["match_id"] = settled_trades.apply(_match_identifier, axis=1)
    metadata: dict[tuple[Any, ...], dict[str, Any]] = {}
    events: list[tuple[pd.Timestamp, int, str, tuple[Any, ...], Any]] = []
    for match_id, group in settled_trades.groupby("match_id", sort=False):
        group = group.sort_values(["dune_trade_time_utc", "evt_index"], kind="stable")
        first = group.iloc[0]
        release_time = pd.Timestamp(first["football_data_release_utc"])
        settlement_time = pd.Timestamp(first["kickoff_uk_time"]) + pd.Timedelta(
            hours=config.assumed_match_duration_hours
        )
        metadata[match_id] = first.to_dict()
        events.append((release_time, 0, "order", match_id, None))
        for row_index in group.index:
            trade_time = pd.Timestamp(group.loc[row_index, "dune_trade_time_utc"])
            events.append((trade_time, 1, "trade", match_id, row_index))
        events.append((settlement_time, 2, "settlement", match_id, None))
    events.sort(key=lambda value: (value[0], value[1]))

    cash = float(config.initial_capital_usd)
    orders: dict[tuple[Any, ...], dict[str, float]] = {}
    open_positions: dict[tuple[Any, ...], dict[str, Any]] = {}
    fill_rows: list[dict[str, Any]] = []
    settled_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = [
        {
            "strategy": strategy,
            "event_time_utc": events[0][0] - pd.Timedelta(nanoseconds=1),
            "event": "initial capital",
            "cash_usd": cash,
            "open_exposure_usd": 0.0,
            "bankroll_usd": cash,
        }
    ]

    for event_time, _, event_type, match_id, row_index in events:
        open_exposure = float(
            sum(position["filled_cost_usd"] for position in open_positions.values())
        )
        current_equity = float(cash + open_exposure)
        if event_type == "order":
            target, configured_fraction = target_stake_for_strategy(
                pd.Series(metadata[match_id]),
                kelly_fraction=kelly_fraction,
                current_equity=current_equity,
                config=config,
            )
            orders[match_id] = {
                "target_stake_usd": target,
                "configured_stake_fraction": configured_fraction,
            }
            continue
        if event_type == "trade":
            order = orders.get(match_id)
            if order is None or order["target_stake_usd"] <= 1e-12:
                continue
            trade = settled_trades.loc[row_index]
            if not bool(trade["execution_eligible"]):
                continue
            position = open_positions.setdefault(
                match_id,
                {
                    "filled_cost_usd": 0.0,
                    "shares": 0.0,
                    "fees_usd": 0.0,
                    "first_fill_time_utc": pd.NaT,
                    "last_fill_time_utc": pd.NaT,
                    "fills": 0,
                },
            )
            remaining_target = max(
                order["target_stake_usd"] - position["filled_cost_usd"], 0.0
            )
            exposure_limit = max(
                config.maximum_open_exposure_fraction * current_equity, 0.0
            )
            available_exposure = max(exposure_limit - open_exposure, 0.0)
            observed_shares = float(trade["dune_trade_shares"])
            unit_cost = float(trade["hypothetical_effective_price"])
            available_shares = config.maximum_trade_participation * observed_shares
            fee_per_share = float(trade["hypothetical_buyer_fee_usd"]) / observed_shares
            available_trade_capacity = available_shares * unit_cost
            fill_cost = min(
                remaining_target,
                available_trade_capacity,
                cash,
                available_exposure,
            )
            if fill_cost <= 1e-12 or unit_cost <= 0:
                if position["filled_cost_usd"] <= 1e-12:
                    open_positions.pop(match_id, None)
                continue
            fill_shares = fill_cost / unit_cost
            fill_fee = fill_shares * fee_per_share
            cash -= fill_cost
            position["filled_cost_usd"] += fill_cost
            position["shares"] += fill_shares
            position["fees_usd"] += fill_fee
            position["fills"] += 1
            if pd.isna(position["first_fill_time_utc"]):
                position["first_fill_time_utc"] = event_time
            position["last_fill_time_utc"] = event_time
            fill_rows.append(
                {
                    "strategy": strategy,
                    "match_id": match_id,
                    "fill_time_utc": event_time,
                    "historical_trade_type": trade["historical_trade_type"],
                    "execution_price_source": trade["execution_price_source"],
                    "fill_cost_usd": fill_cost,
                    "fill_raw_cost_usd": fill_shares * float(trade["execution_raw_price"]),
                    "fill_shares": fill_shares,
                    "fill_fee_usd": fill_fee,
                    "fill_effective_price": unit_cost,
                    "fill_effective_odds": 1.0 / unit_cost,
                    "minimum_required_odds": float(trade["minimum_required_odds"]),
                    "observed_trade_capacity_usd": available_trade_capacity,
                    "fee_source": f"modeled_taker_curve_{config.buyer_fee_rate:.3f}",
                    "tx_hash": trade["tx_hash"],
                    "evt_index": trade["evt_index"],
                }
            )
            continue

        position = open_positions.pop(match_id, None)
        if position is None or position["filled_cost_usd"] <= 1e-12:
            continue
        order = orders[match_id]
        match = metadata[match_id]
        filled_cost = float(position["filled_cost_usd"])
        acquired_shares = float(position["shares"])
        target_stake = float(order["target_stake_usd"])
        fill_fraction = filled_cost / target_stake if target_stake > 0 else 0.0
        won = int(float(match["home_win_result"]) == 1.0)
        payout = acquired_shares if won else 0.0
        profit = payout - filled_cost
        cash += payout
        remaining_exposure = float(
            sum(other["filled_cost_usd"] for other in open_positions.values())
        )
        equity_after = float(cash + remaining_exposure)
        included = fill_fraction + 1e-12 >= config.minimum_fill_fraction
        result = {
            "strategy": strategy,
            **{column: match[column] for column in MATCH_CACHE_KEY},
            "football_data_max_home_odds": float(match["football_data_max_home_odds"]),
            "minimum_required_odds": float(match["minimum_required_odds"]),
            "q_hat_fd": float(match["q_hat_fd"]),
            "home_goals": match.get("home_goals", np.nan),
            "away_goals": match.get("away_goals", np.nan),
            "home_win_result": float(match["home_win_result"]),
            "first_fill_time_utc": position["first_fill_time_utc"],
            "last_fill_time_utc": position["last_fill_time_utc"],
            "settlement_time_utc": event_time,
            "target_stake_usd": target_stake,
            "filled_stake_usd": filled_cost,
            "fill_fraction": fill_fraction,
            "fully_filled": fill_fraction >= 1.0 - 1e-9,
            "number_of_fills": int(position["fills"]),
            "fees_paid_usd": float(position["fees_usd"]),
            "weighted_effective_price": filled_cost / acquired_shares,
            "weighted_effective_odds": acquired_shares / filled_cost,
            "configured_stake_fraction": float(order["configured_stake_fraction"]),
            "profit_usd": profit,
            "bankroll_after_settlement_usd": equity_after,
            "included_by_min_fill_rule": included,
        }
        if included:
            settled_rows.append(result)
        path_rows.append(
            {
                "strategy": strategy,
                "event_time_utc": event_time,
                "event": "match settlement",
                "cash_usd": cash,
                "open_exposure_usd": remaining_exposure,
                "bankroll_usd": equity_after,
            }
        )
    return StrategySimulation(
        pd.DataFrame(settled_rows),
        pd.DataFrame(path_rows),
        pd.DataFrame(fill_rows),
    )


def run_strategy_set(
    trades: pd.DataFrame,
    *,
    config: ExecutionConfig,
) -> StrategySimulation:
    """Run flat stake and every configured fractional-Kelly strategy."""
    specifications = [("flat stake", None)] + [
        (f"{fraction:.0%} Kelly", fraction) for fraction in config.kelly_fractions
    ]
    simulations = [
        simulate_strategy(
            trades,
            strategy=strategy,
            kelly_fraction=fraction,
            config=config,
        )
        for strategy, fraction in specifications
    ]

    def combine(attribute: str) -> pd.DataFrame:
        frames = [getattr(item, attribute) for item in simulations]
        frames = [frame for frame in frames if not frame.empty]
        return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    return StrategySimulation(
        bets=combine("bets"),
        bankroll_paths=combine("bankroll_paths"),
        fills=combine("fills"),
    )


def _add_local_period(
    frame: pd.DataFrame,
    time_column: str,
    *,
    frequency: str,
    timezone: str,
) -> pd.Series:
    local = pd.to_datetime(frame[time_column], utc=True, errors="coerce").dt.tz_convert(timezone)
    return local.dt.tz_localize(None).dt.to_period(frequency)


def period_return_summary(
    paths: pd.DataFrame,
    bets: pd.DataFrame,
    *,
    frequency: str,
    period_column: str,
    initial_capital_usd: float,
    timezone: str,
) -> pd.DataFrame:
    """Calculate non-overlapping bankroll returns, Stake ROI, and bet counts."""
    if paths.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for strategy, strategy_path in paths.groupby("strategy", sort=True):
        strategy_path = strategy_path.sort_values("event_time_utc", kind="stable").copy()
        strategy_path["period"] = _add_local_period(
            strategy_path,
            "event_time_utc",
            frequency=frequency,
            timezone=timezone,
        )
        first_period = strategy_path["period"].min()
        last_period = strategy_path["period"].max()
        periods = pd.period_range(first_period, last_period, freq=frequency)
        ending_bankroll = (
            strategy_path.groupby("period", sort=True)["bankroll_usd"].last().reindex(periods).ffill()
        )
        starting_bankroll = ending_bankroll.shift(1).fillna(initial_capital_usd)
        returns = ending_bankroll / starting_bankroll - 1.0
        strategy_bets = bets.loc[bets["strategy"].eq(strategy)].copy()
        if strategy_bets.empty:
            bet_counts = pd.Series(0, index=periods, dtype=int)
            profits = pd.Series(0.0, index=periods)
            filled_stakes = pd.Series(0.0, index=periods)
        else:
            strategy_bets["period"] = _add_local_period(
                strategy_bets,
                "settlement_time_utc",
                frequency=frequency,
                timezone=timezone,
            )
            bet_counts = strategy_bets.groupby("period").size().reindex(periods, fill_value=0)
            profits = strategy_bets.groupby("period")["profit_usd"].sum().reindex(
                periods, fill_value=0.0
            )
            filled_stakes = strategy_bets.groupby("period")["filled_stake_usd"].sum().reindex(
                periods, fill_value=0.0
            )
        for period in periods:
            filled_stake = float(filled_stakes.loc[period])
            profit = float(profits.loc[period])
            rows.append(
                {
                    "strategy": strategy,
                    period_column: str(period),
                    "bets": int(bet_counts.loc[period]),
                    "filled_stake_usd": filled_stake,
                    "profit_usd": profit,
                    "stake_roi_pct": (
                        profit / filled_stake * 100.0 if filled_stake > 0 else np.nan
                    ),
                    "start_bankroll_usd": float(starting_bankroll.loc[period]),
                    "end_bankroll_usd": float(ending_bankroll.loc[period]),
                    "bankroll_return_pct": float(returns.loc[period] * 100.0),
                }
            )
    return pd.DataFrame(rows)


def maximum_drawdown_pct(bankroll: pd.Series) -> float:
    """Return positive peak-to-trough drawdown in percentage points."""
    values = bankroll.astype(float)
    if values.empty:
        return np.nan
    drawdown = values / values.cummax() - 1.0
    return float(-drawdown.min() * 100.0)


def summarize_strategy_metrics(
    paths: pd.DataFrame,
    bets: pd.DataFrame,
    weekly_returns: pd.DataFrame,
    *,
    config: ExecutionConfig,
) -> pd.DataFrame:
    """Summarize economics and weekly risk for each strategy."""
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
        filled_stake = float(strategy_bets["filled_stake_usd"].sum())
        profit = float(strategy_bets["profit_usd"].sum())
        rows.append(
            {
                "strategy": strategy,
                "bets": len(strategy_bets),
                "wins": int(strategy_bets["home_win_result"].sum()),
                "hit_rate_pct": strategy_bets["home_win_result"].mean() * 100.0,
                "filled_stake_usd": filled_stake,
                "profit_usd": profit,
                "stake_roi_pct": profit / filled_stake * 100.0 if filled_stake > 0 else np.nan,
                "final_bankroll_usd": final_bankroll,
                "cumulative_return_pct": (
                    final_bankroll / config.initial_capital_usd - 1.0
                ) * 100.0,
                "mean_weekly_return_pct": weekly_mean * 100.0,
                "annualized_weekly_volatility_pct": annualized_volatility * 100.0,
                "annualized_weekly_sharpe": annualized_sharpe,
                "max_drawdown_pct": maximum_drawdown_pct(strategy_path["bankroll_usd"]),
                "weeks_observed": len(weekly),
                "mean_fill_fraction": strategy_bets["fill_fraction"].mean(),
            }
        )
    return pd.DataFrame(rows)


def separate_common_strategy_statistics(
    metrics: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = (),
    common_columns: tuple[str, ...] = ("bets", "wins", "hit_rate_pct"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate genuinely shared sample statistics from strategy metrics.

    Common columns are removed from the strategy-level table only when every
    configured strategy has the same value within every requested group.
    """
    if metrics.empty:
        return pd.DataFrame(), metrics.copy()
    required = [*group_columns, "strategy", *common_columns]
    missing = [column for column in required if column not in metrics]
    if missing:
        raise KeyError(f"Strategy metrics are missing presentation columns: {missing}")

    grouped = metrics.groupby(list(group_columns), dropna=False, sort=True) if group_columns else [((), metrics)]
    group_frames = [group for _, group in grouped]
    values_are_common = all(
        all(group[column].nunique(dropna=False) == 1 for column in common_columns)
        for group in group_frames
    )
    if not values_are_common:
        return pd.DataFrame(), metrics.copy()

    common = metrics.loc[:, [*group_columns, *common_columns]].drop_duplicates().reset_index(drop=True)
    strategy_metrics = metrics.drop(columns=list(common_columns)).copy()
    return common, strategy_metrics


def evaluate_execution_trades(
    trades: pd.DataFrame,
    *,
    config: ExecutionConfig,
) -> ExecutionEvaluation:
    """Run all strategies and construct weekly, monthly, and total summaries."""
    simulation = run_strategy_set(trades, config=config)
    if not simulation.fills.empty:
        np.testing.assert_allclose(
            simulation.fills["fill_cost_usd"],
            simulation.fills["fill_raw_cost_usd"] + simulation.fills["fill_fee_usd"],
            rtol=1e-10,
            atol=1e-10,
        )
        if not (
            simulation.fills["fill_effective_odds"] + 1e-12
            >= simulation.fills["minimum_required_odds"]
        ).all():
            raise AssertionError("A fill did not satisfy the premium-adjusted odds threshold")
    weekly = period_return_summary(
        simulation.bankroll_paths,
        simulation.bets,
        frequency=config.weekly_frequency,
        period_column="week",
        initial_capital_usd=config.initial_capital_usd,
        timezone=config.reporting_timezone,
    )
    monthly = period_return_summary(
        simulation.bankroll_paths,
        simulation.bets,
        frequency=config.monthly_frequency,
        period_column="month",
        initial_capital_usd=config.initial_capital_usd,
        timezone=config.reporting_timezone,
    )
    metrics = summarize_strategy_metrics(
        simulation.bankroll_paths,
        simulation.bets,
        weekly,
        config=config,
    )
    return ExecutionEvaluation(simulation, weekly, monthly, metrics)


def evaluate_execution_by_league(
    trades: pd.DataFrame,
    *,
    config: ExecutionConfig,
) -> ExecutionEvaluation:
    """Evaluate each league as an independent full-capital sleeve."""
    if trades.empty or "league" not in trades:
        return ExecutionEvaluation(
            simulation=StrategySimulation(pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
            weekly_returns=pd.DataFrame(),
            monthly_returns=pd.DataFrame(),
            metrics=pd.DataFrame(),
        )
    evaluations: list[tuple[str, ExecutionEvaluation]] = []
    for league, league_trades in trades.groupby("league", sort=True):
        evaluations.append((str(league), evaluate_execution_trades(league_trades.copy(), config=config)))

    metric_frames = []
    bet_frames = []
    path_frames = []
    fill_frames = []
    weekly_frames = []
    monthly_frames = []
    for league, evaluation in evaluations:
        for frame, destination in [
            (evaluation.metrics, metric_frames),
            (evaluation.simulation.bets, bet_frames),
            (evaluation.simulation.bankroll_paths, path_frames),
            (evaluation.simulation.fills, fill_frames),
            (evaluation.weekly_returns, weekly_frames),
            (evaluation.monthly_returns, monthly_frames),
        ]:
            if not frame.empty:
                annotated = frame.copy()
                annotated.insert(0, "analysis_league", league)
                destination.append(annotated)

    def concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()

    return ExecutionEvaluation(
        simulation=StrategySimulation(concat(bet_frames), concat(path_frames), concat(fill_frames)),
        weekly_returns=concat(weekly_frames),
        monthly_returns=concat(monthly_frames),
        metrics=concat(metric_frames),
    )


def round_numeric_for_display(frame: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Round numeric presentation columns without touching timestamps."""
    numeric = frame.select_dtypes(include=[np.number]).columns
    rounded = frame.round({column: decimals for column in numeric})
    floating = rounded.select_dtypes(include=[np.floating]).columns
    for column in floating:
        rounded[column] = rounded[column].mask(rounded[column].eq(0), 0.0)
    return rounded


def order_strategy_rows_for_display(
    frame: pd.DataFrame,
    *,
    group_columns: tuple[str, ...] = (),
    strategy_order: tuple[str, ...] = ("flat stake", "10% Kelly", "25% Kelly"),
) -> pd.DataFrame:
    """Return a presentation-only copy ordered by benchmark then Kelly fraction."""
    if frame.empty or "strategy" not in frame:
        return frame.copy()
    missing_groups = [column for column in group_columns if column not in frame]
    if missing_groups:
        raise KeyError(f"Strategy display groups are missing: {missing_groups}")
    rank = {strategy: position for position, strategy in enumerate(strategy_order)}
    ordered = frame.copy()
    ordered["_strategy_display_order"] = (
        ordered["strategy"].map(rank).fillna(len(rank)).astype(int)
    )
    return (
        ordered.sort_values(
            [*group_columns, "_strategy_display_order"], kind="stable"
        )
        .drop(columns="_strategy_display_order")
        .reset_index(drop=True)
    )
