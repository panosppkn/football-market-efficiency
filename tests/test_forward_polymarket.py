from datetime import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from football_edge.forward_polymarket import (
    MonthlyDuneCollection,
    build_batched_eligible_trade_sql,
    ensure_month_is_not_finalized,
    football_data_release_utc_for_match,
    normalize_text,
    prepare_current_season_fixture_selection,
    prepare_eligible_trade_rows,
    prepare_monthly_fixture_selection,
    resolve_unique_home_win_market,
    save_monthly_outputs_by_league,
)


def test_name_normalization_handles_accents_and_provider_punctuation() -> None:
    assert normalize_text("Club Atlético de Madrid") == "club atletico de madrid"
    assert normalize_text("Vitória-SC") == "vitoria sc"


def test_release_times_use_uk_local_clock_and_tuesday_monday_windows() -> None:
    saturday = football_data_release_utc_for_match("2026-09-12")
    thursday = football_data_release_utc_for_match("2026-09-10")

    assert saturday == pd.Timestamp("2026-09-11 16:00:00+00:00")
    assert thursday == pd.Timestamp("2026-09-08 12:00:00+00:00")


def test_monthly_selection_uses_only_frozen_maxh_probability_range(
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "all-euro-data-2026-2027.xlsx"
    fixtures = pd.DataFrame(
        {
            "Div": ["SP1", "SP1", "SP1"],
            "Date": ["05/09/2026", "12/09/2026", "03/10/2026"],
            "Time": ["15:00", "15:00", "15:00"],
            "HomeTeam": ["A", "C", "E"],
            "AwayTeam": ["B", "D", "F"],
            "MaxH": [2.0, 3.0, 1.8],
            "FTR": ["A", "H", "H"],
            "FTHG": [0, 2, 3],
            "FTAG": [1, 0, 0],
        }
    )
    with pd.ExcelWriter(workbook) as writer:
        fixtures.to_excel(writer, sheet_name="SP1", index=False)
    parameters = tmp_path / "frozen.csv"
    pd.DataFrame(
        {
            "as_of_training_end_season": ["25_26"],
            "market_outcome": ["home_win"],
            "source": ["market_maximum"],
            "league": ["La Liga"],
            "alpha": [-0.01],
            "beta": [0.05],
            "accepted_probability_min": [0.48],
            "accepted_probability_max": [0.99],
        }
    ).to_csv(parameters, index=False)

    selection = prepare_monthly_fixture_selection(
        workbook,
        parameters,
        month="2026-09",
    )

    assert len(selection.observed_fixtures) == 2
    assert list(selection.selected_fixtures["home_team"]) == ["A"]
    assert selection.selected_fixtures.iloc[0]["home_win_result"] == 0.0
    expected_q = 1 / 2.0 - 0.01 + 0.05 * (1 / 2.0)
    assert selection.selected_fixtures.iloc[0]["q_hat_fd"] == pytest.approx(expected_q)


def test_current_season_and_monthly_selection_agree_for_valid_month_rows(
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "all-euro-data-2026-2027.xlsx"
    fixtures = pd.DataFrame(
        {
            "Div": ["SP1", "SP1", "SP1"],
            "Date": ["05/09/2026", "12/09/2026", "03/10/2026"],
            "Time": ["15:00", "15:00", "15:00"],
            "HomeTeam": ["A", "C", "E"],
            "AwayTeam": ["B", "D", "F"],
            "MaxH": [2.0, np.nan, 1.8],
            "FTR": ["H", "D", "H"],
            "FTHG": [2, 0, 3],
            "FTAG": [0, 0, 0],
        }
    )
    with pd.ExcelWriter(workbook) as writer:
        fixtures.to_excel(writer, sheet_name="SP1", index=False)
    parameters = tmp_path / "frozen.csv"
    pd.DataFrame(
        {
            "as_of_training_end_season": ["25_26"],
            "market_outcome": ["home_win"],
            "source": ["market_maximum"],
            "league": ["La Liga"],
            "alpha": [-0.01],
            "beta": [0.05],
            "accepted_probability_min": [0.48],
            "accepted_probability_max": [0.99],
        }
    ).to_csv(parameters, index=False)

    current = prepare_current_season_fixture_selection(workbook, parameters)
    monthly = prepare_monthly_fixture_selection(workbook, parameters, month="2026-09")

    assert len(current.observed_fixtures) == 3
    assert len(monthly.observed_fixtures) == 1
    assert current.observed_fixtures["valid_selection_input"].tolist() == [True, False, True]
    current_september = current.selected_fixtures.loc[
        current.selected_fixtures["date"].dt.to_period("M").eq(pd.Period("2026-09"))
    ]
    assert list(current_september["home_team"]) == list(
        monthly.selected_fixtures["home_team"]
    ) == ["A"]


def test_eligible_trade_sql_preserves_both_canonical_buy_directions() -> None:
    matches = pd.DataFrame(
        {
            "match_index": [0],
            "condition_id": ["0xabc123"],
            "football_data_release_utc": [pd.Timestamp("2026-09-04 16:00", tz="UTC")],
            "kickoff_utc": [pd.Timestamp("2026-09-05 14:00", tz="UTC")],
            "football_data_max_home_odds": [2.0],
        }
    )

    sql = build_batched_eligible_trade_sql(matches)

    assert "taker_side, '')) = 'SELL'" in sql
    assert "maker_side, '')) = 'BUY'" in sql
    assert "t.shares / t.amount >= f.max_h" in sql
    assert "taker_side, '')) = 'BUY'" in sql
    assert "maker_side, '')) = 'SELL'" in sql
    assert "t.shares / (t.amount + t.fee) >= f.max_h" in sql
    assert "t.is_taker_side = TRUE" in sql


def _match_lookup() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "match_index": [0],
            "match_date": ["2026-09-05"],
            "season": ["26_27"],
            "league": ["La Liga"],
            "home_team": ["A"],
            "away_team": ["B"],
            "football_data_release_utc": [pd.Timestamp("2026-09-04 16:00", tz="UTC")],
            "football_data_release_uk_time": [pd.Timestamp("2026-09-04 17:00", tz="Europe/London")],
            "kickoff_utc": [pd.Timestamp("2026-09-05 14:00", tz="UTC")],
            "kickoff_uk_time": [pd.Timestamp("2026-09-05 15:00", tz="Europe/London")],
            "football_data_max_home_odds": [2.0],
            "p_fd": [0.5],
            "g_hat_fd": [0.02],
            "q_hat_fd": [0.52],
            "alpha": [-0.01],
            "beta": [0.06],
            "accepted_probability_min": [0.48],
            "accepted_probability_max": [0.99],
            "home_goals": [1],
            "away_goals": [0],
            "home_win_result": [1.0],
        }
    )


def test_trade_preparation_reconstructs_direction_specific_effective_odds() -> None:
    common = {
        "match_index": 0,
        "price": 0.4,
        "amount": 40.0,
        "shares": 100.0,
        "condition_id": "0xabc123",
        "asset_id": "asset",
        "is_taker_side": True,
    }
    trades = pd.DataFrame(
        [
            {
                **common,
                "block_time": "2026-09-05T10:00:00Z",
                "fee": 0.0,
                "taker_side": "SELL",
                "maker_side": "BUY",
                "tx_hash": "0x1",
                "evt_index": 1,
            },
            {
                **common,
                "block_time": "2026-09-05T11:00:00Z",
                "fee": 2.0,
                "taker_side": "BUY",
                "maker_side": "SELL",
                "tx_hash": "0x2",
                "evt_index": 2,
            },
        ]
    )

    prepared = prepare_eligible_trade_rows(trades, _match_lookup())

    assert prepared.loc[0, "execution_proxy"] == "maker_buy_proxy"
    assert prepared.loc[0, "execution_effective_odds"] == pytest.approx(2.5)
    assert prepared.loc[1, "execution_proxy"] == "taker_buy_proxy"
    assert prepared.loc[1, "execution_effective_odds"] == pytest.approx(100 / 42)


def test_market_resolution_distinguishes_missing_ambiguous_and_unique() -> None:
    assert resolve_unique_home_win_market(pd.DataFrame())[1] == "not_discovered_in_trade_data"
    ambiguous = pd.DataFrame(
        {"condition_id": ["0x1", "0x2"], "asset_id": ["a", "b"]}
    )
    assert resolve_unique_home_win_market(ambiguous)[1] == "ambiguous_2_markets"
    unique = pd.DataFrame(
        {"condition_id": ["0x1", "0x1"], "asset_id": ["a", "a"]}
    )
    market, status = resolve_unique_home_win_market(unique)
    assert status == "resolved"
    assert market["condition_id"] == "0x1"


def test_finalized_month_cannot_be_silently_rewritten(tmp_path: Path) -> None:
    selected = _match_lookup()
    selected["date"] = pd.to_datetime(selected["match_date"])
    collection = MonthlyDuneCollection(
        resolution_table=pd.DataFrame(),
        eligible_trades=pd.DataFrame(),
        match_summary=selected.assign(
            market_resolution_status="not_discovered_in_trade_data",
            eligible_trade_status="market_not_resolved",
        ),
        generated_sql=(),
        discovery_queries_run=0,
        trade_queries_run=0,
    )
    first = save_monthly_outputs_by_league(
        collection,
        selected,
        output_dir=tmp_path,
        season="26_27",
        month="2026-09",
    )
    assert Path(first.iloc[0]["match_summary_file"]).exists()
    assert Path(first.iloc[0]["eligible_trades_file"]).exists()

    with pytest.raises(FileExistsError, match="already has finalized output"):
        ensure_month_is_not_finalized(
            selected,
            output_dir=tmp_path,
            season="26_27",
            month="2026-09",
        )
