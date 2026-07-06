from pathlib import Path

import pytest

from football_edge.data import parse_dataset_name


def test_parse_dataset_name() -> None:
    dataset = parse_dataset_name(Path("Premier_League_24_25.csv"))

    assert dataset.league == "Premier League"
    assert dataset.season == "24_25"


def test_corrects_serie_a_source_spelling() -> None:
    dataset = parse_dataset_name(Path("Seria_A_24_25.csv"))

    assert dataset.league == "Serie A"


def test_parse_dataset_name_rejects_invalid_name() -> None:
    with pytest.raises(ValueError):
        parse_dataset_name(Path("matches.csv"))
