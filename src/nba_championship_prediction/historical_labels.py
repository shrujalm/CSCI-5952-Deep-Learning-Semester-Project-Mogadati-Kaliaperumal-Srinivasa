"""Curated historical NBA playoff outcomes used for supervised labels."""

from __future__ import annotations

from typing import Dict

PLAYOFF_LABEL_NAMES = {
    0: "Missed Playoffs",
    1: "First Round Exit",
    2: "Second Round Exit",
    3: "Conference Finals",
    4: "Finals Loss",
    5: "Champion",
}

HISTORICAL_PLAYOFF_RESULTS: dict[str, dict[str, int]] = {
    "2003-04": {
        "DET": 5, "LAL": 4, "MIN": 3, "IND": 3,
        "SAS": 2, "SAC": 2, "NJN": 2, "MIL": 2,
        "MEM": 1, "HOU": 1, "DAL": 1, "MIA": 1,
        "DEN": 1, "BOS": 1, "NOH": 1, "NYK": 1,
    },
    "2004-05": {
        "SAS": 5, "DET": 4, "MIA": 3, "PHX": 3,
        "IND": 2, "WAS": 2, "DAL": 2, "SEA": 2,
        "CHI": 1, "BOS": 1, "NJN": 1, "PHI": 1,
        "HOU": 1, "MEM": 1, "DEN": 1, "SAC": 1,
    },
    "2005-06": {
        "MIA": 5, "DAL": 4, "DET": 3, "LAC": 3,
        "CLE": 2, "NJN": 2, "PHX": 2, "SAS": 2,
        "CHI": 1, "MIL": 1, "IND": 1, "WAS": 1,
        "MEM": 1, "SAC": 1, "LAL": 1, "DEN": 1,
    },
    "2006-07": {
        "SAS": 5, "CLE": 4, "DET": 3, "UTA": 3,
        "CHI": 2, "NJN": 2, "PHX": 2, "GSW": 2,
        "TOR": 1, "WAS": 1, "ORL": 1, "MIA": 1,
        "HOU": 1, "LAL": 1, "DAL": 1, "DEN": 1,
    },
    "2007-08": {
        "BOS": 5, "LAL": 4, "DET": 3, "SAS": 3,
        "CLE": 2, "ORL": 2, "UTA": 2, "NOH": 2,
        "ATL": 1, "PHI": 1, "TOR": 1, "WAS": 1,
        "PHX": 1, "DAL": 1, "HOU": 1, "DEN": 1,
    },
    "2008-09": {
        "LAL": 5, "ORL": 4, "CLE": 3, "DEN": 3,
        "BOS": 2, "ATL": 2, "HOU": 2, "DAL": 2,
        "CHI": 1, "MIA": 1, "DET": 1, "PHI": 1,
        "SAS": 1, "POR": 1, "NOH": 1, "UTA": 1,
    },
    "2009-10": {
        "LAL": 5, "BOS": 4, "ORL": 3, "PHX": 3,
        "CLE": 2, "ATL": 2, "UTA": 2, "SAS": 2,
        "MIA": 1, "MIL": 1, "CHI": 1, "CHA": 1,
        "DAL": 1, "POR": 1, "OKC": 1, "DEN": 1,
    },
    "2010-11": {
        "DAL": 5, "MIA": 4, "CHI": 3, "OKC": 3,
        "ATL": 2, "BOS": 2, "LAL": 2, "MEM": 2,
        "PHI": 1, "NYK": 1, "ORL": 1, "IND": 1,
        "POR": 1, "DEN": 1, "NOH": 1, "SAS": 1,
    },
    "2011-12": {
        "MIA": 5, "OKC": 4, "BOS": 3, "SAS": 3,
        "IND": 2, "PHI": 2, "LAC": 2, "LAL": 2,
        "NYK": 1, "ATL": 1, "ORL": 1, "CHI": 1,
        "DAL": 1, "MEM": 1, "DEN": 1, "UTA": 1,
    },
    "2012-13": {
        "MIA": 5, "SAS": 4, "IND": 3, "MEM": 3,
        "NYK": 2, "CHI": 2, "GSW": 2, "OKC": 2,
        "BOS": 1, "MIL": 1, "ATL": 1, "BKN": 1,
        "LAC": 1, "DEN": 1, "HOU": 1, "LAL": 1,
    },
    "2013-14": {
        "SAS": 5, "MIA": 4, "IND": 3, "OKC": 3,
        "BKN": 2, "WAS": 2, "POR": 2, "LAC": 2,
        "TOR": 1, "CHI": 1, "CHA": 1, "ATL": 1,
        "HOU": 1, "MEM": 1, "DAL": 1, "GSW": 1,
    },
    "2014-15": {
        "GSW": 5, "CLE": 4, "ATL": 3, "HOU": 3,
        "CHI": 2, "WAS": 2, "LAC": 2, "MEM": 2,
        "TOR": 1, "MIL": 1, "BOS": 1, "BKN": 1,
        "SAS": 1, "DAL": 1, "POR": 1, "NOH": 1,
    },
    "2015-16": {
        "CLE": 5, "GSW": 4, "OKC": 3, "TOR": 3,
        "MIA": 2, "ATL": 2, "SAS": 2, "POR": 2,
        "CHA": 1, "IND": 1, "BOS": 1, "DET": 1,
        "LAC": 1, "MEM": 1, "DAL": 1, "HOU": 1,
    },
    "2016-17": {
        "GSW": 5, "CLE": 4, "BOS": 3, "SAS": 3,
        "WAS": 2, "TOR": 2, "HOU": 2, "UTA": 2,
        "MIL": 1, "ATL": 1, "CHI": 1, "IND": 1,
        "LAC": 1, "MEM": 1, "OKC": 1, "POR": 1,
    },
    "2017-18": {
        "GSW": 5, "CLE": 4, "BOS": 3, "HOU": 3,
        "TOR": 2, "PHI": 2, "NOP": 2, "UTA": 2,
        "IND": 1, "MIL": 1, "MIA": 1, "WAS": 1,
        "SAS": 1, "MIN": 1, "OKC": 1, "POR": 1,
    },
    "2018-19": {
        "TOR": 5, "GSW": 4, "MIL": 3, "POR": 3,
        "PHI": 2, "BOS": 2, "DEN": 2, "HOU": 2,
        "BKN": 1, "ORL": 1, "IND": 1, "DET": 1,
        "OKC": 1, "SAS": 1, "UTA": 1, "LAC": 1,
    },
    "2019-20": {
        "LAL": 5, "MIA": 4, "BOS": 3, "DEN": 3,
        "TOR": 2, "MIL": 2, "LAC": 2, "HOU": 2,
        "BKN": 1, "PHI": 1, "IND": 1, "ORL": 1,
        "OKC": 1, "DAL": 1, "POR": 1, "UTA": 1,
    },
    "2020-21": {
        "MIL": 5, "PHX": 4, "ATL": 3, "LAC": 3,
        "PHI": 2, "BKN": 2, "UTA": 2, "DEN": 2,
        "NYK": 1, "BOS": 1, "MIA": 1, "WAS": 1,
        "LAL": 1, "POR": 1, "DAL": 1, "MEM": 1,
    },
    "2021-22": {
        "GSW": 5, "BOS": 4, "MIA": 3, "DAL": 3,
        "PHI": 2, "MIL": 2, "MEM": 2, "PHX": 2,
        "TOR": 1, "CHI": 1, "BKN": 1, "ATL": 1,
        "NOP": 1, "MIN": 1, "UTA": 1, "DEN": 1,
    },
    "2022-23": {
        "DEN": 5, "MIA": 4, "BOS": 3, "LAL": 3,
        "PHI": 2, "NYK": 2, "PHX": 2, "GSW": 2,
        "CLE": 1, "ATL": 1, "MIL": 1, "BKN": 1,
        "MEM": 1, "SAC": 1, "LAC": 1, "MIN": 1,
    },
    "2023-24": {
        "BOS": 5, "DAL": 4, "IND": 3, "MIN": 3,
        "NYK": 2, "CLE": 2, "DEN": 2, "OKC": 2,
        "PHI": 1, "MIL": 1, "ORL": 1, "MIA": 1,
        "LAL": 1, "NOP": 1, "LAC": 1, "PHX": 1,
    },
}

SEASONS = list(HISTORICAL_PLAYOFF_RESULTS.keys())

ABBREV_FIXES = {
    "NJN": "BKN",
    "NOH": "NOP",
    "NOK": "NOP",
    "SEA": "OKC",
    "CHO": "CHA",
    "VAN": "MEM",
    "GOS": "GSW",
}

HISTORICAL_VARIANTS = {
    "BKN": ["NJN", "BKN"],
    "NOP": ["NOH", "NOK", "NOP"],
    "OKC": ["SEA", "OKC"],
    "CHA": ["CHA", "CHO", "CHH"],
    "MEM": ["VAN", "MEM"],
}


def fix_abbrev(abbrev: str) -> str:
    """Normalize historical team abbreviations to a consistent representation."""
    if abbrev == "CHO":
        return "CHA"
    return ABBREV_FIXES.get(abbrev, abbrev)


def get_playoff_label(team_abbrev: str, season: str) -> int:
    """Return the curated postseason label for a team-season."""
    results = HISTORICAL_PLAYOFF_RESULTS.get(season, {})
    if team_abbrev in results:
        return results[team_abbrev]

    clean_abbrev = fix_abbrev(team_abbrev)
    if clean_abbrev in results:
        return results[clean_abbrev]

    for normalized, variants in HISTORICAL_VARIANTS.items():
        if team_abbrev == normalized or team_abbrev in variants or clean_abbrev in variants:
            for variant in variants:
                if variant in results:
                    return results[variant]

    return 0


def get_playoff_results(season: str) -> Dict[str, int]:
    """Return the curated playoff mapping for one season."""
    return HISTORICAL_PLAYOFF_RESULTS.get(season, {})
