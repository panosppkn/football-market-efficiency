"""Dataset discovery, naming, and validation."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from football_edge.config import ALL_EURO_DATA_PATTERN, DATA_PATTERN, RAW_DATA_DIR, REQUIRED_COLUMNS


LEAGUE_CODE_TO_NAME = {
    "E0": "Premier League",
    "E1": "Championship",
    "E2": "League One",
    "E3": "League Two",
    "EC": "Conference",
    "SC0": "Scottish Premiership",
    "SC1": "Scottish Championship",
    "SC2": "Scottish League One",
    "SC3": "Scottish League Two",
    "D1": "Bundesliga",
    "D2": "Bundesliga 2",
    "SP1": "La Liga",
    "SP2": "Segunda Division",
    "I1": "Serie A",
    "I2": "Serie B",
    "F1": "Championnat",
    "F2": "Ligue 2",
    "N1": "Eredivisie",
    "B1": "Belgian Pro League",
    "P1": "Primeira Liga",
    "T1": "Super Lig",
    "G1": "Greek Super League",
}


@dataclass(frozen=True)
class Dataset:
    """Metadata inferred from a raw source file.

    One-league CSV files use only ``path``, ``league``, and ``season``.
    All-Europe season workbooks additionally use ``sheet_name`` and
    ``division`` so each workbook sheet behaves like one logical league-season
    dataset.
    """

    path: Path
    league: str
    season: str
    sheet_name: str | None = None
    division: str | None = None


def parse_dataset_name(path: str | Path) -> Dataset:
    """Parse ``<league>_<YY>_<YY>.csv`` into human-readable metadata."""
    path = Path(path)
    try:
        league_key, start_year, end_year = path.stem.rsplit("_", 2)
    except ValueError as error:
        raise ValueError(
            f"{path.name!r} must follow <league>_<YY>_<YY>.csv"
        ) from error

    if not league_key or not (
        len(start_year) == len(end_year) == 2
        and start_year.isdigit()
        and end_year.isdigit()
    ):
        raise ValueError(f"{path.name!r} must follow <league>_<YY>_<YY>.csv")

    league = league_key.replace("_", " ")
    if league == "Seria A":
        league = "Serie A"

    return Dataset(
        path=path,
        league=league,
        season=f"{start_year}_{end_year}",
    )


def _normalize_season_key(season: str) -> str:
    """Normalize season inputs such as 2006/07, 2006/2007, or 06_07."""
    parts = re.findall(r"\d+", str(season))
    if len(parts) >= 2:
        return f"{parts[0][-2:]}_{parts[1][-2:]}"
    return str(season).replace("/", "_").replace("-", "_")


def _normalize_season_filter(seasons: Iterable[str] | None) -> set[str] | None:
    if seasons is None:
        return None
    return {_normalize_season_key(season) for season in seasons}


def parse_all_euro_dataset_name(path: str | Path, sheet_name: str) -> Dataset:
    """Parse ``all-euro[-data]-YYYY-YYYY.xls`` plus a league sheet name."""
    path = Path(path)
    match = re.fullmatch(r"all-euro(?:-data)?-(\d{4})-(\d{4})", path.stem)
    if match is None:
        raise ValueError(
            f"{path.name!r} must follow all-euro[-data]-YYYY-YYYY.xls"
        )

    start_year, end_year = match.groups()
    division = str(sheet_name)
    league = LEAGUE_CODE_TO_NAME.get(division, division)
    return Dataset(
        path=path,
        league=league,
        season=f"{start_year[-2:]}_{end_year[-2:]}",
        sheet_name=division,
        division=division,
    )


def discover_all_euro_datasets(
    data_dir: str | Path = RAW_DATA_DIR,
    pattern: str = ALL_EURO_DATA_PATTERN,
    seasons: Iterable[str] | None = None,
) -> list[Dataset]:
    """Return one logical dataset per sheet in each selected all-Europe workbook."""
    data_dir = Path(data_dir)
    selected_seasons = _normalize_season_filter(seasons)
    datasets: list[Dataset] = []
    for path in sorted(data_dir.glob(pattern)):
        match = re.fullmatch(r"all-euro(?:-data)?-(\d{4})-(\d{4})", path.stem)
        if match is None:
            continue
        workbook_season = f"{match.group(1)[-2:]}_{match.group(2)[-2:]}"
        if selected_seasons is not None and workbook_season not in selected_seasons:
            continue
        try:
            workbook = pd.ExcelFile(path)
        except ImportError as error:
            raise ImportError(
                "Reading Football-Data all-Europe .xls files requires xlrd. "
                "Install project dependencies or run `pip install xlrd>=2.0.1`."
            ) from error
        for sheet_name in workbook.sheet_names:
            datasets.append(parse_all_euro_dataset_name(path, str(sheet_name)))
    return datasets


def discover_datasets(
    data_dir: str | Path = RAW_DATA_DIR,
    pattern: str = DATA_PATTERN,
    seasons: Iterable[str] | None = None,
) -> list[Dataset]:
    """Return all valid league-season datasets in deterministic order.

    The default discovery remains backward-compatible with one-league CSV files
    and also includes all-Europe season workbooks when present.
    """
    data_dir = Path(data_dir)
    selected_seasons = _normalize_season_filter(seasons)
    datasets = [parse_dataset_name(path) for path in sorted(data_dir.glob(pattern))]
    if selected_seasons is not None:
        datasets = [dataset for dataset in datasets if dataset.season in selected_seasons]
    if pattern == DATA_PATTERN:
        datasets.extend(discover_all_euro_datasets(data_dir, seasons=selected_seasons))
    if not datasets:
        raise FileNotFoundError(f"No datasets matching {pattern!r} in {data_dir}")
    return sorted(
        datasets,
        key=lambda item: (item.league, item.season, item.division or "", item.path.name),
    )


def _standardize_football_data_columns(matches: pd.DataFrame) -> pd.DataFrame:
    """Normalize historical Football-Data column names to the repo schema.

    Football-Data changed some aggregate odds names over time. When multiple
    seasons are concatenated, both the old and new columns can be present, with
    values populated in different seasons. In that case we coalesce old values
    into the modern repo column instead of leaving early seasons as missing.
    """
    rename_map = {
        "BbAv>2.5": "Avg>2.5",
        "BbAv<2.5": "Avg<2.5",
        "BbMx>2.5": "Max>2.5",
        "BbMx<2.5": "Max<2.5",
        "BbAvH": "AvgH",
        "BbAvD": "AvgD",
        "BbAvA": "AvgA",
        "BbMxH": "MaxH",
        "BbMxD": "MaxD",
        "BbMxA": "MaxA",
    }
    standardized = matches.copy()
    for old_column, new_column in rename_map.items():
        if old_column not in standardized.columns:
            continue
        if new_column in standardized.columns:
            standardized[new_column] = standardized[new_column].combine_first(
                standardized[old_column]
            )
        else:
            standardized = standardized.rename(columns={old_column: new_column})
    return standardized


def _read_raw_matches(source: str | Path | Dataset) -> pd.DataFrame:
    if isinstance(source, Dataset):
        path = source.path
        sheet_name = source.sheet_name
    else:
        path = Path(source)
        sheet_name = None

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xls", ".xlsx"}:
        try:
            return pd.read_excel(path, sheet_name=sheet_name or 0)
        except ImportError as error:
            raise ImportError(
                "Reading Football-Data Excel files requires xlrd for .xls "
                "workbooks. Install project dependencies or run "
                "`pip install xlrd>=2.0.1`."
            ) from error
    raise ValueError(f"Unsupported raw data file type: {path.suffix}")


def load_matches(source: str | Path | Dataset) -> pd.DataFrame:
    """Load and validate completed matches from one logical dataset."""
    dataset = source if isinstance(source, Dataset) else None
    path = dataset.path if dataset is not None else Path(source)
    matches = _standardize_football_data_columns(_read_raw_matches(source))

    if dataset is not None and dataset.division is not None and "Div" in matches.columns:
        matches = matches.loc[matches["Div"].astype(str).eq(dataset.division)].copy()

    missing_columns = REQUIRED_COLUMNS.difference(matches.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    matches = matches.dropna(
        subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    ).copy()
    duplicate_matches = matches.duplicated(
        subset=["Date", "HomeTeam", "AwayTeam"], keep=False
    )
    if duplicate_matches.any():
        raise ValueError(
            f"{path.name} contains {int(duplicate_matches.sum())} duplicate match rows"
        )
    if matches["HomeTeam"].eq(matches["AwayTeam"]).any():
        raise ValueError(f"{path.name} contains a match with identical teams")

    matches["FTHG"] = pd.to_numeric(matches["FTHG"], errors="raise")
    matches["FTAG"] = pd.to_numeric(matches["FTAG"], errors="raise")

    # Use the match date as the chronological key. Older Football-Data files
    # may not provide a kickoff time, and the public walk-forward analysis is
    # calibrated at the season level rather than from intraday information.
    matches["date"] = pd.to_datetime(
        matches["Date"], dayfirst=True, errors="raise"
    )

    return (
        matches.sort_values("date", kind="stable")
        .reset_index(drop=True)
        .reset_index(names="match_id")
    )
