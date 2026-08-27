from pathlib import Path

import pandas as pd
import pytest

from football_edge.data import load_matches, parse_dataset_name


def test_parse_dataset_name() -> None:
    dataset = parse_dataset_name(Path("Premier_League_24_25.csv"))

    assert dataset.league == "Premier League"
    assert dataset.season == "24_25"


def test_parse_serie_a_canonical_filename() -> None:
    dataset = parse_dataset_name(Path("Serie_A_24_25.csv"))

    assert dataset.league == "Serie A"
    assert dataset.season == "24_25"


def test_corrects_legacy_seria_a_source_spelling() -> None:
    dataset = parse_dataset_name(Path("Seria_A_24_25.csv"))

    assert dataset.league == "Serie A"


def test_parse_dataset_name_rejects_invalid_name() -> None:
    with pytest.raises(ValueError):
        parse_dataset_name(Path("matches.csv"))


def test_load_matches_accepts_files_without_time(tmp_path: Path) -> None:
    path = tmp_path / "Premier_League_24_25.csv"
    pd.DataFrame(
        {
            "Date": ["16/08/2024", "17/08/2024"],
            "HomeTeam": ["A", "C"],
            "AwayTeam": ["B", "D"],
            "FTHG": [1, 2],
            "FTAG": [0, 2],
        }
    ).to_csv(path, index=False)

    matches = load_matches(path)

    assert list(matches["date"]) == list(
        pd.to_datetime(["2024-08-16", "2024-08-17"])
    )



def test_parse_all_euro_dataset_name() -> None:
    from football_edge.data import parse_all_euro_dataset_name

    dataset = parse_all_euro_dataset_name(Path("all-euro-data-2006-2007.xls"), "E0")

    assert dataset.league == "Premier League"
    assert dataset.season == "06_07"
    assert dataset.sheet_name == "E0"
    assert dataset.division == "E0"


def test_load_matches_accepts_dataset_metadata_for_csv(tmp_path: Path) -> None:
    from football_edge.data import Dataset

    path = tmp_path / "Premier_League_24_25.csv"
    pd.DataFrame(
        {
            "Date": ["16/08/2024"],
            "HomeTeam": ["A"],
            "AwayTeam": ["B"],
            "FTHG": [1],
            "FTAG": [0],
        }
    ).to_csv(path, index=False)

    matches = load_matches(Dataset(path=path, league="Premier League", season="24_25"))

    assert len(matches) == 1
    assert matches.loc[0, "HomeTeam"] == "A"



def test_discover_datasets_filters_all_euro_before_loading(tmp_path: Path, monkeypatch) -> None:
    from football_edge import data

    (tmp_path / "all-euro-data-2006-2007.xls").write_bytes(b"fake")
    (tmp_path / "all-euro-data-2007-2008.xls").write_bytes(b"fake")

    class FakeWorkbook:
        sheet_names = ["E0", "D1"]

    opened = []

    def fake_excel_file(path):
        opened.append(Path(path).name)
        return FakeWorkbook()

    monkeypatch.setattr(data.pd, "ExcelFile", fake_excel_file)

    datasets = data.discover_datasets(tmp_path, seasons=["2006/07"])

    assert opened == ["all-euro-data-2006-2007.xls"]
    assert {(dataset.season, dataset.division) for dataset in datasets} == {
        ("06_07", "E0"),
        ("06_07", "D1"),
    }



def test_load_matches_standardizes_historical_market_average_and_maximum_odds(tmp_path: Path) -> None:
    path = tmp_path / "Premier_League_06_07.csv"
    pd.DataFrame(
        {
            "Date": ["19/08/2006"],
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Aston Villa"],
            "FTHG": [1],
            "FTAG": [1],
            "FTR": ["D"],
            "BbAvH": [1.27],
            "BbAvD": [4.82],
            "BbAvA": [10.72],
            "BbMxH": [1.33],
            "BbMxD": [5.50],
            "BbMxA": [13.25],
            "BbAv>2.5": [1.75],
            "BbAv<2.5": [2.01],
            "BbMx>2.5": [1.83],
            "BbMx<2.5": [2.12],
        }
    ).to_csv(path, index=False)

    matches = load_matches(path)

    for column in ["AvgH", "AvgD", "AvgA", "MaxH", "MaxD", "MaxA", "Avg>2.5", "Avg<2.5", "Max>2.5", "Max<2.5"]:
        assert column in matches.columns
    assert matches.loc[0, "FTR"] == "D"
    assert matches.loc[0, "AvgH"] == 1.27
