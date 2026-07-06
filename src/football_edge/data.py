"""Dataset discovery, naming, and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from football_edge.config import DATA_PATTERN, RAW_DATA_DIR, REQUIRED_COLUMNS


@dataclass(frozen=True)
class Dataset:
    """Metadata inferred from a raw CSV filename."""

    path: Path
    league: str
    season: str


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


def discover_datasets(
    data_dir: str | Path = RAW_DATA_DIR,
    pattern: str = DATA_PATTERN,
) -> list[Dataset]:
    """Return all valid league-season datasets in deterministic order."""
    data_dir = Path(data_dir)
    datasets = [parse_dataset_name(path) for path in sorted(data_dir.glob(pattern))]
    if not datasets:
        raise FileNotFoundError(f"No datasets matching {pattern!r} in {data_dir}")
    return sorted(datasets, key=lambda item: (item.league, item.season))


def load_matches(path: str | Path) -> pd.DataFrame:
    """Load and validate completed matches from one raw dataset."""
    path = Path(path)
    matches = pd.read_csv(path)

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

    date_time = (
        matches["Date"].astype(str)
        + " "
        + matches["Time"].fillna("00:00").astype(str)
    )
    matches["date"] = pd.to_datetime(date_time, dayfirst=True, errors="raise")

    return (
        matches.sort_values("date", kind="stable")
        .reset_index(drop=True)
        .reset_index(names="match_id")
    )
