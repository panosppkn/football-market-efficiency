"""Forward Polymarket data collection for frozen Football-Data signals.

This module deliberately separates match selection from execution-data collection:
Football-Data ``MaxH`` and published parameters decide which fixtures qualify,
while Dune supplies historical Polymarket transaction observations only.  No
model fitting, staking, or performance calculation belongs here.
"""

from __future__ import annotations

import json
import os
import re
import time as time_module
import unicodedata
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from football_edge.data import LEAGUE_CODE_TO_NAME, _standardize_football_data_columns


MATCH_CACHE_KEY = ["season", "league", "match_date", "home_team", "away_team"]
TRADE_IDENTITY_KEY = ["tx_hash", "evt_index", "condition_id", "asset_id"]

SQL_TRANSLATE_FROM = "áàâäãåéèêëíìîïóòôöõúùûüçñşığ"
SQL_TRANSLATE_TO = "aaaaaaeeeeiiiiooooouuuucnsgi"
if len(SQL_TRANSLATE_FROM) != len(SQL_TRANSLATE_TO):  # pragma: no cover
    raise AssertionError("SQL normalization alphabets must have equal length")
PYTHON_NAME_TRANSLATION = str.maketrans(SQL_TRANSLATE_FROM, SQL_TRANSLATE_TO)

DEFAULT_BAD_MARKET_KEYWORDS = (
    "champion",
    "title",
    "winner of la liga",
    "top 4",
    "top4",
    "finish",
    "handicap",
    "spread",
    "total",
    "over ",
    "under ",
    "goals",
    "exact score",
    "first half",
    "second half",
    "halftime",
)

# These aliases address provider naming differences. Short ambiguous fragments
# such as ``mad`` and ``atm`` are intentionally excluded.
DEFAULT_TEAM_ALIASES: dict[str, tuple[str, ...]] = {
    "AEK": ("aek athens", "aek athens fc"),
    "Academico Viseu": ("academico de viseu",),
    "Alaves": ("alaves", "deportivo alaves"),
    "Ath Bilbao": ("ath bilbao", "athletic bilbao", "athletic club", "athletic club bilbao"),
    "Ath Madrid": ("ath madrid", "atletico", "atletico madrid", "tico de madrid"),
    "Barcelona": ("barcelona", "fc barcelona", "barca"),
    "Betis": ("betis", "real betis"),
    "Celta": ("celta", "celta vigo", "rc celta", "rc celta de vigo"),
    "Elche": ("elche", "elche cf"),
    "Espanol": ("espanol", "espanyol", "rcd espanyol", "rcd espanyol de barcelona"),
    "Getafe": ("getafe", "getafe cf"),
    "Guimaraes": ("vitoria sc",),
    "La Coruna": ("la coruna", "coruna", "deportivo", "deportivo la coruna", "deportivo de la coruna"),
    "Levadeiakos": ("levadiakos",),
    "Levante": ("levante", "levante ud"),
    "Malaga": ("malaga", "malaga cf"),
    "Olympiakos": ("olympiacos",),
    "Osasuna": ("osasuna", "ca osasuna"),
    "Real Madrid": ("real madrid", "real madrid cf"),
    "Santander": ("real racing club", "racing santander", "racing de santander", "santander"),
    "Sevilla": ("sevilla", "sevilla fc"),
    "Sociedad": ("sociedad", "real sociedad"),
    "Sp Lisbon": ("sporting cp",),
    "Valencia": ("valencia", "valencia cf"),
    "Vallecano": ("vallecano", "rayo", "rayo vallecano"),
    "Villarreal": ("villarreal", "villarreal cf"),
}

ELIGIBLE_TRADE_CACHE_COLUMNS = [
    "match_date",
    "season",
    "league",
    "home_team",
    "away_team",
    "football_data_release_utc",
    "football_data_release_uk_time",
    "kickoff_utc",
    "kickoff_uk_time",
    "football_data_max_home_odds",
    "p_fd",
    "q_hat_fd",
    "dune_trade_time_utc",
    "dune_trade_time_uk",
    "execution_proxy",
    "execution_effective_odds",
    "dune_home_win_price",
    "dune_price_from_amount",
    "dune_raw_decimal_odds",
    "dune_recorded_taker_buy_effective_odds",
    "dune_trade_amount_usd",
    "dune_trade_shares",
    "dune_recorded_fee_usd",
    "is_taker_side",
    "maker_side",
    "taker_side",
    "meets_maxh_execution_threshold",
    "home_goals",
    "away_goals",
    "home_win_result",
    "question",
    "event_market_name",
    "condition_id",
    "asset_id",
    "polymarket_link",
    "tx_hash",
    "evt_index",
    "unique_key",
    "order_hash",
    "action",
    "contract_version",
    "maker",
    "taker",
]


@dataclass(frozen=True)
class MonthlyFixtureSelection:
    """Prepared fixtures for one frozen-rule calendar month."""

    season: str
    month: str
    frozen_parameters: pd.DataFrame
    observed_fixtures: pd.DataFrame
    selected_fixtures: pd.DataFrame
    skipped_sheets: pd.DataFrame


@dataclass(frozen=True)
class CurrentSeasonFixtureSelection:
    """Frozen-rule fixture preparation for a complete current-season workbook."""

    season: str
    frozen_parameters: pd.DataFrame
    observed_fixtures: pd.DataFrame
    selected_fixtures: pd.DataFrame
    skipped_sheets: pd.DataFrame


@dataclass(frozen=True)
class MonthlyDuneCollection:
    """Dune collection output before monthly files are finalized."""

    resolution_table: pd.DataFrame
    eligible_trades: pd.DataFrame
    match_summary: pd.DataFrame
    generated_sql: tuple[dict[str, str], ...]
    discovery_queries_run: int
    trade_queries_run: int


class DuneClient:
    """Minimal Dune SQL API client with credentials supplied by the caller."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.dune.com/api/v1",
        performance: str = "medium",
        poll_seconds: float = 5.0,
        timeout_seconds: float = 600.0,
        result_limit: int = 10_000,
        request_timeout_seconds: float = 60.0,
    ) -> None:
        if not api_key:
            raise ValueError("A non-empty Dune API key is required")
        self._api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.performance = performance
        self.poll_seconds = poll_seconds
        self.timeout_seconds = timeout_seconds
        self.result_limit = result_limit
        self.request_timeout_seconds = request_timeout_seconds

    @classmethod
    def from_environment(cls, variable: str = "DUNE_API_KEY", **kwargs: Any) -> "DuneClient":
        """Create a client without embedding credentials in code or notebooks."""
        api_key = os.getenv(variable, "")
        if not api_key:
            raise RuntimeError(f"{variable} is not configured in this process environment")
        return cls(api_key, **kwargs)

    def _request(
        self,
        method: str,
        endpoint: str,
        *,
        payload: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        if params:
            url = f"{url}?{urlencode(params)}"
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={"X-Dune-Api-Key": self._api_key, "Content-Type": "application/json"},
            method=method,
        )
        try:
            with urlopen(request, timeout=self.request_timeout_seconds) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Dune API HTTP {error.code}: {detail}") from error
        except URLError as error:
            raise RuntimeError(f"Dune API request failed: {error}") from error

    def run_sql(self, sql: str, *, label: str) -> pd.DataFrame:
        """Execute SQL, wait for completion, and retrieve every result page."""
        started = time_module.perf_counter()
        response = self._request(
            "POST",
            "/sql/execute",
            payload={"sql": sql, "performance": self.performance},
        )
        execution_id = response.get("execution_id")
        if not execution_id:
            raise RuntimeError(f"Dune did not return an execution_id: {response}")
        print(f"{label}: submitted Dune execution {execution_id}")

        while True:
            status = self._request("GET", f"/execution/{execution_id}/status")
            state = status.get("state") or status.get("execution_state")
            if state in {
                "QUERY_STATE_COMPLETED",
                "QUERY_STATE_FAILED",
                "QUERY_STATE_CANCELLED",
                "QUERY_STATE_EXPIRED",
            }:
                break
            if time_module.perf_counter() - started > self.timeout_seconds:
                raise TimeoutError(f"Dune execution timed out: {execution_id}; last status: {status}")
            time_module.sleep(self.poll_seconds)
        if state != "QUERY_STATE_COMPLETED":
            raise RuntimeError(f"{label} failed: {status}")

        rows: list[dict[str, Any]] = []
        offset = 0
        while True:
            payload = self._request(
                "GET",
                f"/execution/{execution_id}/results",
                params={"limit": self.result_limit, "offset": offset},
            )
            result = payload.get("result", payload)
            rows.extend(result.get("rows") or payload.get("rows") or [])
            next_offset = payload.get("next_offset") or result.get("next_offset")
            if next_offset is None:
                break
            offset = int(next_offset)
        elapsed = time_module.perf_counter() - started
        print(f"{label}: completed in {elapsed:.1f}s; {len(rows):,} rows returned")
        return pd.DataFrame(rows)


def infer_season_from_file(path: str | Path) -> str:
    """Infer ``YY_YY`` from a workbook name containing ``YYYY-YYYY``."""
    path = Path(path)
    match = re.search(r"(\d{4})[-_](\d{4})", path.stem)
    if match is None:
        raise ValueError("Could not infer the season from the workbook name")
    start_year, end_year = match.groups()
    return f"{start_year[-2:]}_{end_year[-2:]}"


def read_current_season_workbook(
    path: str | Path,
    *,
    season: str,
    require_max_odds: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read usable fixture sheets from a Football-Data all-Europe workbook."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Current-season workbook not found: {path}")
    workbook = pd.ExcelFile(path)
    frames: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []
    for sheet_name in workbook.sheet_names:
        raw = _standardize_football_data_columns(
            pd.read_excel(workbook, sheet_name=sheet_name)
        ).dropna(how="all")
        required = ["Date", "HomeTeam", "AwayTeam", "MaxH"]
        missing = [column for column in required if column not in raw]
        if missing:
            skipped.append({"source_sheet": str(sheet_name), "reason": f"missing {missing}"})
            continue
        required_values = required if require_max_odds else ["Date", "HomeTeam", "AwayTeam"]
        frame = raw.dropna(subset=required_values).copy()
        if frame.empty:
            requirement = "fixture/MaxH" if require_max_odds else "fixture"
            skipped.append(
                {"source_sheet": str(sheet_name), "reason": f"no complete {requirement} rows"}
            )
            continue
        division = (
            str(frame["Div"].dropna().astype(str).iloc[0])
            if "Div" in frame and frame["Div"].notna().any()
            else str(sheet_name)
        )
        frame["season"] = season
        frame["source_file"] = path.name
        frame["source_sheet"] = str(sheet_name)
        frame["division"] = division
        frame["league"] = LEAGUE_CODE_TO_NAME.get(division, division)
        frames.append(frame)
    if not frames:
        raise ValueError(f"No usable sheets found in {path}")
    return pd.concat(frames, ignore_index=True, sort=False), pd.DataFrame(skipped)


def load_frozen_parameters(
    path: str | Path,
    *,
    market_outcome: str = "home_win",
    source: str = "market_maximum",
) -> pd.DataFrame:
    """Load the published parameter rows for one fixed signal specification."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    parameters = pd.read_csv(path)
    required = {
        "as_of_training_end_season",
        "market_outcome",
        "source",
        "league",
        "alpha",
        "beta",
        "accepted_probability_min",
        "accepted_probability_max",
    }
    missing = sorted(required.difference(parameters.columns))
    if missing:
        raise ValueError(f"Frozen parameter file is missing columns: {missing}")
    parameters = parameters.loc[
        parameters["source"].eq(source)
        & parameters["market_outcome"].eq(market_outcome)
    ].copy()
    numeric = ["alpha", "beta", "accepted_probability_min", "accepted_probability_max"]
    for column in numeric:
        parameters[column] = pd.to_numeric(parameters[column], errors="coerce")
    parameters = parameters.dropna(subset=["league", *numeric]).reset_index(drop=True)
    if parameters.empty:
        raise ValueError(f"No frozen rows found for source={source!r}, outcome={market_outcome!r}")
    if parameters["league"].duplicated().any():
        raise ValueError("Frozen parameter file contains duplicate league rows")
    return parameters


def parse_fixture_kickoff_utc(
    row: pd.Series,
    *,
    fixture_timezone: str = "Europe/London",
    missing_time_default: time = time(15, 0),
) -> pd.Timestamp | pd.NaT:
    """Convert Football-Data's UK-local fixture date/time to UTC."""
    date = pd.to_datetime(row["Date"], dayfirst=True, errors="coerce")
    if pd.isna(date):
        return pd.NaT
    raw_time = row.get("Time")
    if raw_time in (None, "") or pd.isna(raw_time):
        kickoff_time = missing_time_default
    elif isinstance(raw_time, time):
        kickoff_time = raw_time
    elif isinstance(raw_time, datetime):
        kickoff_time = raw_time.time()
    elif isinstance(raw_time, (int, float)) and np.isfinite(raw_time):
        seconds = int(round(float(raw_time) * 24 * 60 * 60))
        kickoff_time = (datetime.min + timedelta(seconds=seconds)).time()
    else:
        parsed = pd.to_datetime(str(raw_time), errors="coerce")
        kickoff_time = parsed.time() if pd.notna(parsed) else missing_time_default
    local = datetime.combine(date.date(), kickoff_time, tzinfo=ZoneInfo(fixture_timezone))
    return pd.Timestamp(local.astimezone(timezone.utc))


def football_data_release_utc_for_match(
    match_date: Any,
    *,
    fixture_timezone: str = "Europe/London",
    weekend_release_time: time = time(17, 0),
    midweek_release_time: time = time(13, 0),
) -> pd.Timestamp | pd.NaT:
    """Return the documented latest Football-Data collection time.

    Friday--Monday fixtures use Friday 17:00 UK; Tuesday--Thursday fixtures
    use Tuesday 13:00 UK.  ``Europe/London`` handles daylight-saving time.
    """
    date = pd.to_datetime(match_date, errors="coerce")
    if pd.isna(date):
        return pd.NaT
    fixture_date = date.date()
    weekday = fixture_date.weekday()
    if weekday in {4, 5, 6, 0}:
        release_date = fixture_date - timedelta(days=(weekday - 4) % 7)
        release_time = weekend_release_time
    else:
        release_date = fixture_date - timedelta(days=weekday - 1)
        release_time = midweek_release_time
    local = datetime.combine(release_date, release_time, tzinfo=ZoneInfo(fixture_timezone))
    return pd.Timestamp(local.astimezone(timezone.utc))


def _home_win_result(frame: pd.DataFrame) -> pd.Series:
    if "FTR" not in frame:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    result = frame["FTR"]
    valid = result.isin(["H", "D", "A"])
    return result.eq("H").astype(float).where(valid)


def _prepare_frozen_fixture_frame(
    raw: pd.DataFrame,
    parameters: pd.DataFrame,
    *,
    fixture_timezone: str,
    weekend_release_time: time,
    midweek_release_time: time,
) -> pd.DataFrame:
    """Apply the shared frozen Football-Data calculations without period filtering."""
    fixtures = raw.merge(
        parameters[
            [
                "league",
                "alpha",
                "beta",
                "accepted_probability_min",
                "accepted_probability_max",
            ]
        ],
        on="league",
        how="inner",
        validate="many_to_one",
    ).rename(columns={"HomeTeam": "home_team", "AwayTeam": "away_team"})
    fixtures["date"] = pd.to_datetime(fixtures["Date"], dayfirst=True, errors="coerce")
    fixtures["kickoff_utc"] = fixtures.apply(
        parse_fixture_kickoff_utc,
        axis=1,
        fixture_timezone=fixture_timezone,
    )
    fixtures["kickoff_uk_time"] = pd.to_datetime(
        fixtures["kickoff_utc"], utc=True
    ).dt.tz_convert(fixture_timezone)
    fixtures["football_data_release_utc"] = fixtures["date"].apply(
        football_data_release_utc_for_match,
        fixture_timezone=fixture_timezone,
        weekend_release_time=weekend_release_time,
        midweek_release_time=midweek_release_time,
    )
    fixtures["football_data_release_uk_time"] = pd.to_datetime(
        fixtures["football_data_release_utc"], utc=True
    ).dt.tz_convert(fixture_timezone)
    fixtures["football_data_max_home_odds"] = pd.to_numeric(
        fixtures["MaxH"], errors="coerce"
    )
    fixtures["valid_selection_input"] = (
        fixtures["date"].notna()
        & fixtures["kickoff_utc"].notna()
        & fixtures["football_data_release_utc"].notna()
        & (fixtures["football_data_release_utc"] < fixtures["kickoff_utc"])
        & fixtures["football_data_max_home_odds"].gt(1)
    )
    fixtures["p_fd"] = np.where(
        fixtures["football_data_max_home_odds"].gt(1),
        1.0 / fixtures["football_data_max_home_odds"],
        np.nan,
    )
    fixtures["g_hat_fd"] = fixtures["alpha"] + fixtures["beta"] * fixtures["p_fd"]
    fixtures["q_hat_fd"] = (fixtures["p_fd"] + fixtures["g_hat_fd"]).clip(0.001, 0.999)
    valid = fixtures["valid_selection_input"]
    expected_q = (
        1.0 / fixtures.loc[valid, "football_data_max_home_odds"]
        + fixtures.loc[valid, "alpha"]
        + fixtures.loc[valid, "beta"]
        * (1.0 / fixtures.loc[valid, "football_data_max_home_odds"])
    ).clip(0.001, 0.999)
    np.testing.assert_allclose(
        fixtures.loc[valid, "q_hat_fd"], expected_q, rtol=1e-12, atol=1e-12
    )
    fixtures["accepted_fd"] = valid & fixtures["p_fd"].between(
        fixtures["accepted_probability_min"],
        fixtures["accepted_probability_max"],
        inclusive="both",
    )
    fixtures["home_goals"] = pd.to_numeric(fixtures.get("FTHG"), errors="coerce")
    fixtures["away_goals"] = pd.to_numeric(fixtures.get("FTAG"), errors="coerce")
    fixtures["home_win_result"] = _home_win_result(fixtures)
    fixtures["match_date"] = fixtures["date"].dt.strftime("%Y-%m-%d")
    return fixtures.sort_values(
        ["league", "date", "home_team", "away_team"], kind="stable"
    ).reset_index(drop=True)


def prepare_current_season_fixture_selection(
    workbook_path: str | Path,
    frozen_parameters_path: str | Path,
    *,
    season: str | None = None,
    target_league: str | None = None,
    divisions: Iterable[str] | None = None,
    fixture_timezone: str = "Europe/London",
    weekend_release_time: time = time(17, 0),
    midweek_release_time: time = time(13, 0),
    market_outcome: str = "home_win",
    source: str = "market_maximum",
) -> CurrentSeasonFixtureSelection:
    """Apply the published frozen rule to every fixture in a current-season workbook."""
    workbook_path = Path(workbook_path)
    season = season or infer_season_from_file(workbook_path)
    raw, skipped = read_current_season_workbook(
        workbook_path, season=season, require_max_odds=False
    )
    parameters = load_frozen_parameters(
        frozen_parameters_path,
        market_outcome=market_outcome,
        source=source,
    )
    if divisions is not None:
        raw = raw.loc[raw["division"].isin(list(divisions))].copy()
    if target_league is not None:
        raw = raw.loc[raw["league"].eq(target_league)].copy()
    fixtures = _prepare_frozen_fixture_frame(
        raw,
        parameters,
        fixture_timezone=fixture_timezone,
        weekend_release_time=weekend_release_time,
        midweek_release_time=midweek_release_time,
    )
    selected = fixtures.loc[fixtures["accepted_fd"]].copy().reset_index(drop=True)
    selected.insert(0, "match_index", np.arange(len(selected), dtype=int))
    return CurrentSeasonFixtureSelection(
        season=season,
        frozen_parameters=parameters,
        observed_fixtures=fixtures,
        selected_fixtures=selected,
        skipped_sheets=skipped,
    )


def prepare_monthly_fixture_selection(
    workbook_path: str | Path,
    frozen_parameters_path: str | Path,
    *,
    month: str,
    season: str | None = None,
    target_league: str | None = None,
    divisions: Iterable[str] | None = None,
    fixture_timezone: str = "Europe/London",
    weekend_release_time: time = time(17, 0),
    midweek_release_time: time = time(13, 0),
    market_outcome: str = "home_win",
    source: str = "market_maximum",
) -> MonthlyFixtureSelection:
    """Apply the published frozen Football-Data rule to one calendar month."""
    try:
        target_period = pd.Period(month, freq="M")
    except ValueError as error:
        raise ValueError("month must use YYYY-MM format, for example '2026-09'") from error
    workbook_path = Path(workbook_path)
    season = season or infer_season_from_file(workbook_path)
    raw, skipped = read_current_season_workbook(workbook_path, season=season)
    parameters = load_frozen_parameters(
        frozen_parameters_path,
        market_outcome=market_outcome,
        source=source,
    )
    if divisions is not None:
        raw = raw.loc[raw["division"].isin(list(divisions))].copy()
    if target_league is not None:
        raw = raw.loc[raw["league"].eq(target_league)].copy()
    fixture_month = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce").dt.to_period("M")
    raw = raw.loc[fixture_month.eq(target_period)].copy()

    fixtures = _prepare_frozen_fixture_frame(
        raw,
        parameters,
        fixture_timezone=fixture_timezone,
        weekend_release_time=weekend_release_time,
        midweek_release_time=midweek_release_time,
    )
    # Preserve the collection notebook's established behavior: only fixtures
    # with a usable MaxH and a valid release-to-kickoff interval are observed.
    fixtures = fixtures.loc[fixtures["valid_selection_input"]].copy().reset_index(drop=True)
    fixtures = fixtures.drop(columns="valid_selection_input")

    selected = fixtures.loc[fixtures["accepted_fd"]].copy().reset_index(drop=True)
    selected.insert(0, "match_index", np.arange(len(selected), dtype=int))
    return MonthlyFixtureSelection(
        season=season,
        month=str(target_period),
        frozen_parameters=parameters,
        observed_fixtures=fixtures,
        selected_fixtures=selected,
        skipped_sheets=skipped,
    )


def normalize_text(value: Any) -> str:
    """Normalize provider text consistently in Python."""
    text = str(value or "").casefold().translate(PYTHON_NAME_TRANSLATION)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sql_quote(value: Any) -> str:
    """Quote a scalar as a Dune SQL string literal."""
    return "'" + str(value).replace("'", "''") + "'"


def sql_normalized_text(expression: str) -> str:
    """Return the Dune SQL counterpart of :func:`normalize_text`."""
    return (
        f"translate(lower(coalesce({expression}, '')), "
        f"{sql_quote(SQL_TRANSLATE_FROM)}, {sql_quote(SQL_TRANSLATE_TO)})"
    )


def match_aliases(
    team: str,
    *,
    aliases: dict[str, Iterable[str]] = DEFAULT_TEAM_ALIASES,
) -> list[str]:
    """Return deterministic normalized aliases with ambiguous short forms removed."""
    candidates = [team, normalize_text(team), *aliases.get(str(team), ())]
    output: list[str] = []
    for alias in candidates:
        normalized = normalize_text(alias)
        if normalized and len(normalized) >= 4 and normalized not in output:
            output.append(normalized)
    return output


def _sql_regex_pattern(values: Iterable[str]) -> str:
    escaped = [re.escape(normalize_text(value)) for value in values if normalize_text(value)]
    if not escaped:
        raise ValueError("At least one non-empty alias is required")
    return "(" + "|".join(sorted(set(escaped), key=lambda value: (-len(value), value))) + ")"


def _condition_id_sql_literal(value: Any) -> str:
    text = str(value).strip().lower()
    if not re.fullmatch(r"0x[0-9a-f]+", text):
        raise ValueError(f"Invalid condition_id: {value!r}")
    return text


def build_batched_market_discovery_sql(
    matches: pd.DataFrame,
    *,
    aliases: dict[str, Iterable[str]] = DEFAULT_TEAM_ALIASES,
    bad_market_keywords: Iterable[str] = DEFAULT_BAD_MARKET_KEYWORDS,
) -> str:
    """Build one batched query that discovers exact home-win market identities."""
    if matches.empty:
        raise ValueError("At least one unresolved fixture is required for discovery")
    values: list[str] = []
    for match in matches.itertuples(index=False):
        home_pattern = _sql_regex_pattern(match_aliases(str(match.home_team), aliases=aliases))
        away_pattern = _sql_regex_pattern(match_aliases(str(match.away_team), aliases=aliases))
        release = pd.Timestamp(match.football_data_release_utc)
        kickoff = pd.Timestamp(match.kickoff_utc)
        values.append(
            "(" + ", ".join(
                [
                    str(int(match.match_index)),
                    f"TIMESTAMP {sql_quote(release.strftime('%Y-%m-%d %H:%M:%S'))}",
                    f"TIMESTAMP {sql_quote(kickoff.strftime('%Y-%m-%d %H:%M:%S'))}",
                    sql_quote(home_pattern),
                    sql_quote(away_pattern),
                ]
            ) + ")"
        )
    global_start = pd.to_datetime(matches["football_data_release_utc"], utc=True).min()
    global_end = pd.to_datetime(matches["kickoff_utc"], utc=True).max()
    event_text = sql_normalized_text("t.event_market_name")
    question_text = sql_normalized_text("t.question")
    combined_text = f"({event_text} || ' ' || {question_text})"
    bad_filters = "\n  ".join(
        f"AND {combined_text} NOT LIKE {sql_quote('%' + normalize_text(keyword) + '%')}"
        for keyword in bad_market_keywords
    )
    values_sql = ",\n        ".join(values)
    return f"""
WITH fixtures(match_index, release_utc, kickoff_utc, home_pattern, away_pattern) AS (
    VALUES
        {values_sql}
), candidate_trades AS (
    SELECT block_time, condition_id, asset_id, question, event_market_name, polymarket_link
    FROM polymarket_polygon.market_trades
    WHERE block_time >= TIMESTAMP {sql_quote(global_start.strftime('%Y-%m-%d %H:%M:%S'))}
      AND block_time < TIMESTAMP {sql_quote(global_end.strftime('%Y-%m-%d %H:%M:%S'))}
      AND lower(coalesce(token_outcome, '')) = 'yes'
      AND is_taker_side = TRUE
)
SELECT DISTINCT
    f.match_index,
    t.condition_id,
    t.asset_id,
    t.question,
    t.event_market_name,
    t.polymarket_link
FROM fixtures f
JOIN candidate_trades t
  ON t.block_time >= f.release_utc
 AND t.block_time < f.kickoff_utc
WHERE regexp_like({event_text}, f.home_pattern)
  AND regexp_like({event_text}, f.away_pattern)
  AND regexp_like({question_text}, f.home_pattern)
  AND {question_text} LIKE '%win%'
  {bad_filters}
""".strip()


def resolve_unique_home_win_market(
    candidates: pd.DataFrame,
) -> tuple[dict[str, Any] | None, str]:
    """Resolve exactly one ``condition_id``/YES ``asset_id`` pair."""
    if candidates.empty:
        return None, "not_discovered_in_trade_data"
    missing = [column for column in ["condition_id", "asset_id"] if column not in candidates]
    if missing:
        raise KeyError(f"Dune market discovery is missing columns: {missing}")
    unique = candidates.dropna(subset=["condition_id", "asset_id"]).drop_duplicates(
        subset=["condition_id", "asset_id"]
    )
    if len(unique) != 1:
        return None, f"ambiguous_{len(unique)}_markets"
    return unique.iloc[0].to_dict(), "resolved"


def build_batched_eligible_trade_sql(matches: pd.DataFrame) -> str:
    """Build the direction-aware query for eligible YES execution observations."""
    if matches.empty:
        raise ValueError("At least one resolved fixture is required for trade retrieval")
    values: list[str] = []
    for match in matches.itertuples(index=False):
        release = pd.Timestamp(match.football_data_release_utc)
        kickoff = pd.Timestamp(match.kickoff_utc)
        values.append(
            "(" + ", ".join(
                [
                    str(int(match.match_index)),
                    _condition_id_sql_literal(match.condition_id),
                    f"TIMESTAMP {sql_quote(release.strftime('%Y-%m-%d %H:%M:%S'))}",
                    f"TIMESTAMP {sql_quote(kickoff.strftime('%Y-%m-%d %H:%M:%S'))}",
                    repr(float(match.football_data_max_home_odds)),
                ]
            ) + ")"
        )
    global_start = pd.to_datetime(matches["football_data_release_utc"], utc=True).min()
    global_end = pd.to_datetime(matches["kickoff_utc"], utc=True).max()
    values_sql = ",\n        ".join(values)
    return f"""
WITH fixtures(match_index, condition_id, release_utc, kickoff_utc, max_h) AS (
    VALUES
        {values_sql}
)
SELECT
    f.match_index,
    t.block_time,
    t.evt_index,
    t.action,
    t.contract_version,
    t.event_market_name,
    t.question,
    t.polymarket_link,
    t.token_outcome,
    t.token_outcome_name,
    t.price,
    t.amount,
    t.shares,
    t.fee,
    t.condition_id,
    t.asset_id,
    t.tx_hash,
    t.unique_key,
    t.order_hash,
    t.is_taker_side,
    t.maker_side,
    t.taker_side,
    t.maker,
    t.taker
FROM fixtures f
JOIN polymarket_polygon.market_trades t
  ON t.condition_id = f.condition_id
 AND t.block_time >= f.release_utc
 AND t.block_time < f.kickoff_utc
WHERE t.block_time >= TIMESTAMP {sql_quote(global_start.strftime('%Y-%m-%d %H:%M:%S'))}
  AND t.block_time < TIMESTAMP {sql_quote(global_end.strftime('%Y-%m-%d %H:%M:%S'))}
  AND lower(coalesce(t.token_outcome, '')) = 'yes'
  AND t.is_taker_side = TRUE
  AND t.price BETWEEN 0.001 AND 0.999
  AND t.shares IS NOT NULL AND t.shares > 0
  AND t.amount IS NOT NULL AND t.amount > 0
  AND (
      (
          upper(coalesce(t.taker_side, '')) = 'SELL'
          AND upper(coalesce(t.maker_side, '')) = 'BUY'
          AND t.shares / t.amount >= f.max_h
      )
      OR
      (
          upper(coalesce(t.taker_side, '')) = 'BUY'
          AND upper(coalesce(t.maker_side, '')) = 'SELL'
          AND t.fee IS NOT NULL
          AND t.amount + t.fee > 0
          AND t.shares / (t.amount + t.fee) >= f.max_h
      )
  )
""".strip()


def _read_csv_if_present(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(
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


def _normalize_match_keys(frame: pd.DataFrame, *, season: str | None = None) -> pd.DataFrame:
    output = frame.copy()
    if "date" in output and "match_date" not in output:
        output["match_date"] = pd.to_datetime(output["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    elif "match_date" in output:
        output["match_date"] = pd.to_datetime(output["match_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "season" not in output and season is not None and not output.empty:
        output["season"] = season
    for column in ["season", "league", "home_team", "away_team"]:
        if column in output:
            output[column] = output[column].astype(str)
    return output


def _match_key_series(frame: pd.DataFrame, *, season: str | None = None) -> pd.Series:
    normalized = _normalize_match_keys(frame, season=season)
    if normalized.empty:
        return pd.Series(index=normalized.index, dtype="string")
    return normalized[MATCH_CACHE_KEY].astype(str).agg("|".join, axis=1)


def _upsert_rows(existing: pd.DataFrame, new: pd.DataFrame, *, keys: list[str]) -> pd.DataFrame:
    if existing.empty:
        return new.copy().reset_index(drop=True)
    if new.empty:
        return existing.copy().reset_index(drop=True)
    combined = pd.concat([existing, new], ignore_index=True, sort=False)
    return combined.drop_duplicates(subset=keys, keep="last").reset_index(drop=True)


def prepare_eligible_trade_rows(
    raw_trades: pd.DataFrame,
    match_lookup: pd.DataFrame,
    *,
    fixture_timezone: str = "Europe/London",
) -> pd.DataFrame:
    """Validate and annotate canonical, direction-correct eligible transactions."""
    if raw_trades.empty:
        return pd.DataFrame(columns=ELIGIBLE_TRADE_CACHE_COLUMNS)
    trades = raw_trades.copy()
    trades["match_index"] = pd.to_numeric(trades["match_index"], errors="coerce").astype("Int64")
    trades["block_time"] = pd.to_datetime(trades["block_time"], utc=True, errors="coerce")
    for column in ["price", "amount", "shares", "fee", "evt_index"]:
        trades[column] = pd.to_numeric(trades[column], errors="coerce")
    trades = trades.dropna(subset=["match_index", "block_time", "price", "amount", "shares"])
    trades = trades.loc[trades["amount"].gt(0) & trades["shares"].gt(0)].copy()
    metadata = [
        "match_index",
        "match_date",
        "season",
        "league",
        "home_team",
        "away_team",
        "football_data_release_utc",
        "football_data_release_uk_time",
        "kickoff_utc",
        "kickoff_uk_time",
        "football_data_max_home_odds",
        "p_fd",
        "g_hat_fd",
        "q_hat_fd",
        "alpha",
        "beta",
        "accepted_probability_min",
        "accepted_probability_max",
        "home_goals",
        "away_goals",
        "home_win_result",
    ]
    trades = trades.merge(match_lookup[metadata], on="match_index", how="left", validate="many_to_one")
    trades["taker_side"] = trades["taker_side"].astype(str).str.upper()
    trades["maker_side"] = trades["maker_side"].astype(str).str.upper()
    maker_buy = trades["taker_side"].eq("SELL") & trades["maker_side"].eq("BUY")
    taker_buy = trades["taker_side"].eq("BUY") & trades["maker_side"].eq("SELL")
    if not (maker_buy | taker_buy).all():
        raise AssertionError("Eligible Dune rows contain inconsistent maker/taker directions")
    if not trades["is_taker_side"].fillna(False).astype(bool).all():
        raise AssertionError("Eligible Dune rows contain a noncanonical representation")

    trades["dune_trade_time_utc"] = trades["block_time"]
    trades["dune_trade_time_uk"] = trades["block_time"].dt.tz_convert(fixture_timezone)
    trades["dune_home_win_price"] = trades["price"]
    trades["dune_trade_amount_usd"] = trades["amount"]
    trades["dune_trade_shares"] = trades["shares"]
    trades["dune_recorded_fee_usd"] = trades["fee"]
    trades["dune_price_from_amount"] = trades["amount"] / trades["shares"]
    trades["dune_raw_decimal_odds"] = trades["shares"] / trades["amount"]
    trades["dune_recorded_taker_buy_effective_odds"] = np.where(
        taker_buy & trades["fee"].notna() & (trades["amount"] + trades["fee"]).gt(0),
        trades["shares"] / (trades["amount"] + trades["fee"]),
        np.nan,
    )
    trades["execution_proxy"] = np.select(
        [maker_buy, taker_buy], ["maker_buy_proxy", "taker_buy_proxy"], default="unsupported_side"
    )
    trades["execution_effective_odds"] = np.select(
        [maker_buy, taker_buy],
        [trades["dune_raw_decimal_odds"], trades["dune_recorded_taker_buy_effective_odds"]],
        default=np.nan,
    )
    trades["meets_maxh_execution_threshold"] = (
        trades["execution_effective_odds"] + 1e-12 >= trades["football_data_max_home_odds"]
    )
    if not trades["meets_maxh_execution_threshold"].all():
        raise AssertionError("Dune rows include an effective price below Football-Data MaxH")
    release = pd.to_datetime(trades["football_data_release_utc"], utc=True)
    kickoff = pd.to_datetime(trades["kickoff_utc"], utc=True)
    trades["within_release_window"] = trades["block_time"].ge(release) & trades["block_time"].lt(kickoff)
    if not trades["within_release_window"].all():
        raise AssertionError("Dune rows include a transaction outside the release-to-kickoff window")
    return trades


def collect_monthly_dune_data(
    selected_fixtures: pd.DataFrame,
    *,
    client: DuneClient,
    identity_cache_file: str | Path,
    fixture_timezone: str = "Europe/London",
    force_refresh_market_discovery: bool = False,
    aliases: dict[str, Iterable[str]] = DEFAULT_TEAM_ALIASES,
) -> MonthlyDuneCollection:
    """Resolve markets and fetch eligible Dune transactions for selected fixtures."""
    matches = selected_fixtures.copy().reset_index(drop=True)
    if "match_index" not in matches:
        matches.insert(0, "match_index", np.arange(len(matches), dtype=int))
    identity_cache_file = Path(identity_cache_file)
    identity_cache_file.parent.mkdir(parents=True, exist_ok=True)
    required_identity = [
        *MATCH_CACHE_KEY,
        "condition_id",
        "asset_id",
        "question",
        "event_market_name",
        "polymarket_link",
    ]
    identity_cache = _normalize_match_keys(_read_csv_if_present(identity_cache_file))
    if identity_cache.empty:
        identity_cache = pd.DataFrame(columns=required_identity)
    else:
        missing = [column for column in required_identity if column not in identity_cache]
        if missing:
            raise KeyError(f"Market identity cache is missing columns: {missing}")

    resolution = matches[["match_index", *MATCH_CACHE_KEY]].merge(
        identity_cache[required_identity], on=MATCH_CACHE_KEY, how="left"
    )
    resolution["market_resolution_status"] = np.where(
        resolution["condition_id"].notna(), "resolved", "pending_discovery"
    )
    resolution["market_resolution_source"] = np.where(
        resolution["condition_id"].notna(), "identity_cache", None
    )
    resolution["market_candidates_found"] = np.where(resolution["condition_id"].notna(), 1, 0)
    if force_refresh_market_discovery:
        resolution[["condition_id", "asset_id", "question", "event_market_name", "polymarket_link"]] = None
    discovery = matches.loc[
        matches["match_index"].isin(
            resolution.loc[resolution["condition_id"].isna(), "match_index"].astype(int)
        )
    ].copy()

    generated_sql: list[dict[str, str]] = []
    new_identities: list[dict[str, Any]] = []
    discovery_queries = 0
    if not discovery.empty:
        sql = build_batched_market_discovery_sql(discovery, aliases=aliases)
        generated_sql.append({"query_type": "batched_market_discovery", "sql": sql})
        candidates = client.run_sql(sql, label=f"Market discovery ({len(discovery)} fixtures)")
        discovery_queries = 1
        if not candidates.empty:
            candidates["match_index"] = pd.to_numeric(candidates["match_index"], errors="coerce").astype("Int64")
        for match_index in discovery["match_index"].astype(int):
            fixture_candidates = (
                candidates.loc[candidates["match_index"].eq(match_index)].copy()
                if not candidates.empty
                else pd.DataFrame()
            )
            market, status = resolve_unique_home_win_market(fixture_candidates)
            mask = resolution["match_index"].eq(match_index)
            resolution.loc[mask, "market_resolution_status"] = status
            resolution.loc[mask, "market_resolution_source"] = "dune_trade_discovery"
            resolution.loc[mask, "market_candidates_found"] = len(fixture_candidates)
            if market is not None:
                for column in ["condition_id", "asset_id", "question", "event_market_name", "polymarket_link"]:
                    resolution.loc[mask, column] = market.get(column)
                row = resolution.loc[mask].iloc[0].to_dict()
                row["market_resolved_at_utc"] = datetime.now(timezone.utc).isoformat()
                new_identities.append(
                    {column: row.get(column) for column in [*required_identity, "market_resolved_at_utc"]}
                )
    if new_identities:
        identity_cache = _upsert_rows(
            identity_cache,
            _normalize_match_keys(pd.DataFrame(new_identities)),
            keys=MATCH_CACHE_KEY,
        )
        identity_cache.to_csv(identity_cache_file, index=False)

    resolved = matches.merge(
        resolution[
            [
                "match_index",
                "condition_id",
                "asset_id",
                "question",
                "event_market_name",
                "polymarket_link",
                "market_resolution_status",
                "market_resolution_source",
                "market_candidates_found",
            ]
        ],
        on="match_index",
        how="left",
    )
    resolved = resolved.loc[resolved["market_resolution_status"].eq("resolved")].copy()
    raw_trades = pd.DataFrame()
    trade_queries = 0
    if not resolved.empty:
        sql = build_batched_eligible_trade_sql(resolved)
        generated_sql.append({"query_type": "batched_eligible_trades", "sql": sql})
        raw_trades = client.run_sql(sql, label=f"Eligible trade retrieval ({len(resolved)} fixtures)")
        trade_queries = 1
    eligible = prepare_eligible_trade_rows(raw_trades, matches, fixture_timezone=fixture_timezone)
    if not eligible.empty:
        eligible = eligible.drop_duplicates(subset=TRADE_IDENTITY_KEY, keep="last")

    summary = matches[
        [
            "match_index",
            "date",
            "match_date",
            "season",
            "football_data_release_uk_time",
            "kickoff_uk_time",
            "league",
            "home_team",
            "away_team",
            "football_data_max_home_odds",
            "p_fd",
            "q_hat_fd",
            "home_goals",
            "away_goals",
            "home_win_result",
        ]
    ].merge(
        resolution[
            [
                "match_index",
                "market_resolution_status",
                "market_resolution_source",
                "market_candidates_found",
                "condition_id",
                "asset_id",
                "question",
                "event_market_name",
                "polymarket_link",
            ]
        ],
        on="match_index",
        how="left",
    )
    if not eligible.empty:
        eligible["is_maker_buy_proxy"] = eligible["execution_proxy"].eq("maker_buy_proxy")
        eligible["is_taker_buy_proxy"] = eligible["execution_proxy"].eq("taker_buy_proxy")
        trade_summary = (
            eligible.groupby(MATCH_CACHE_KEY, sort=False)
            .agg(
                eligible_trade_rows=("dune_trade_time_utc", "size"),
                maker_buy_proxy_rows=("is_maker_buy_proxy", "sum"),
                taker_buy_proxy_rows=("is_taker_buy_proxy", "sum"),
                first_eligible_trade_uk=("dune_trade_time_uk", "min"),
                last_eligible_trade_uk=("dune_trade_time_uk", "max"),
                best_eligible_effective_odds=("execution_effective_odds", "max"),
                total_eligible_amount_usd=("dune_trade_amount_usd", "sum"),
                total_eligible_shares=("dune_trade_shares", "sum"),
            )
            .reset_index()
        )
        summary = summary.merge(trade_summary, on=MATCH_CACHE_KEY, how="left")
    for column in ["eligible_trade_rows", "maker_buy_proxy_rows", "taker_buy_proxy_rows"]:
        if column not in summary:
            summary[column] = 0
        summary[column] = summary[column].fillna(0).astype(int)
    summary["eligible_trade_status"] = np.select(
        [
            ~summary["market_resolution_status"].eq("resolved"),
            summary["eligible_trade_rows"].gt(0),
        ],
        ["market_not_resolved", "eligible_trades_found"],
        default="resolved_no_eligible_trades",
    )
    return MonthlyDuneCollection(
        resolution_table=resolution,
        eligible_trades=eligible,
        match_summary=summary,
        generated_sql=tuple(generated_sql),
        discovery_queries_run=discovery_queries,
        trade_queries_run=trade_queries,
    )


def monthly_collection_diagnostics(
    selection: MonthlyFixtureSelection,
    collection: MonthlyDuneCollection,
) -> pd.DataFrame:
    """Return the six primary audit counts for a monthly collection run."""
    summary = collection.match_summary
    resolved = summary["market_resolution_status"].eq("resolved")
    eligible = summary["eligible_trade_status"].eq("eligible_trades_found")
    not_discovered = summary["market_resolution_status"].eq("not_discovered_in_trade_data")
    no_execution = summary["eligible_trade_status"].eq("resolved_no_eligible_trades")
    return pd.DataFrame(
        [
            {
                "month": selection.month,
                "football_data_matches_observed": len(selection.observed_fixtures),
                "frozen_rule_matches_selected": len(selection.selected_fixtures),
                "polymarket_markets_discovered": int(resolved.sum()),
                "matches_with_eligible_execution": int(eligible.sum()),
                "matches_not_discovered_in_trade_data": int(not_discovered.sum()),
                "discovered_without_eligible_execution": int(no_execution.sum()),
            }
        ]
    )


def cache_slug(value: str) -> str:
    """Create a stable lower-snake-case component for monthly file names."""
    return re.sub(r"[^a-z0-9]+", "_", normalize_text(value)).strip("_")


def planned_monthly_output_files(
    selected_fixtures: pd.DataFrame,
    *,
    output_dir: str | Path,
    season: str,
    month: str,
    source: str = "market_maximum",
    market_outcome: str = "home_win",
) -> pd.DataFrame:
    """List the two finalized output files expected for every selected league."""
    output_dir = Path(output_dir)
    month_key = pd.Period(month, freq="M").strftime("%Y_%m")
    rows = []
    for league in sorted(selected_fixtures["league"].dropna().astype(str).unique()):
        prefix = f"{cache_slug(league)}_{season}_{month_key}_{source}_{market_outcome}"
        rows.append(
            {
                "league": league,
                "month": month,
                "match_summary_file": output_dir / f"{prefix}_match_summary.csv",
                "eligible_trades_file": output_dir / f"{prefix}_eligible_trades.csv",
            }
        )
    return pd.DataFrame(rows)


def ensure_month_is_not_finalized(
    selected_fixtures: pd.DataFrame,
    *,
    output_dir: str | Path,
    season: str,
    month: str,
    allow_correction_overwrite: bool = False,
    source: str = "market_maximum",
    market_outcome: str = "home_win",
) -> pd.DataFrame:
    """Refuse accidental replacement of an already finalized league-month."""
    planned = planned_monthly_output_files(
        selected_fixtures,
        output_dir=output_dir,
        season=season,
        month=month,
        source=source,
        market_outcome=market_outcome,
    )
    existing: list[Path] = []
    for column in ["match_summary_file", "eligible_trades_file"]:
        if column in planned:
            existing.extend(path for path in planned[column] if Path(path).exists())
    if existing and not allow_correction_overwrite:
        formatted = "\n".join(f"- {path}" for path in existing)
        raise FileExistsError(
            "This league-month already has finalized output. Set "
            "ALLOW_CORRECTION_OVERWRITE=True only for an explicitly documented correction:\n"
            f"{formatted}"
        )
    return planned


def save_monthly_outputs_by_league(
    collection: MonthlyDuneCollection,
    selected_fixtures: pd.DataFrame,
    *,
    output_dir: str | Path,
    season: str,
    month: str,
    allow_correction_overwrite: bool = False,
    source: str = "market_maximum",
    market_outcome: str = "home_win",
) -> pd.DataFrame:
    """Finalize auditable match-summary and eligible-trade files per league-month."""
    planned = ensure_month_is_not_finalized(
        selected_fixtures,
        output_dir=output_dir,
        season=season,
        month=month,
        allow_correction_overwrite=allow_correction_overwrite,
        source=source,
        market_outcome=market_outcome,
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict[str, Any]] = []
    for file_row in planned.itertuples(index=False):
        league = file_row.league
        summary = collection.match_summary.loc[
            collection.match_summary["league"].eq(league)
        ].drop(columns=["match_index"], errors="ignore").copy()
        trades = collection.eligible_trades.loc[
            collection.eligible_trades["league"].eq(league)
        ].copy() if not collection.eligible_trades.empty else pd.DataFrame(
            columns=ELIGIBLE_TRADE_CACHE_COLUMNS
        )
        available = [column for column in ELIGIBLE_TRADE_CACHE_COLUMNS if column in trades]
        trades = trades.loc[:, available]
        if not trades.empty:
            trades = trades.drop_duplicates(subset=TRADE_IDENTITY_KEY, keep="last").sort_values(
                ["dune_trade_time_utc", "tx_hash", "evt_index"], kind="stable"
            )
        _write_csv_atomically(summary, Path(file_row.match_summary_file))
        _write_csv_atomically(trades, Path(file_row.eligible_trades_file))
        saved.append(
            {
                "league": league,
                "month": month,
                "selected_matches": len(summary),
                "eligible_trade_rows": len(trades),
                "match_summary_file": str(file_row.match_summary_file),
                "eligible_trades_file": str(file_row.eligible_trades_file),
                "correction_overwrite": bool(allow_correction_overwrite),
            }
        )
    return pd.DataFrame(saved)


def _write_csv_atomically(frame: pd.DataFrame, path: Path) -> None:
    """Replace one finalized CSV only after its complete temporary file exists."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()
