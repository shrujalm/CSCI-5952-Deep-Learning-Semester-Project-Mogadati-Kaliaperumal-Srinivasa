"""Dataset construction and feature engineering for the NBA study."""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .historical_labels import PLAYOFF_LABEL_NAMES, SEASONS, fix_abbrev, get_playoff_label

try:
    from nba_api.stats.endpoints import LeagueDashPlayerStats, LeagueDashTeamStats, TeamEstimatedMetrics
    from nba_api.stats.static import teams as nba_teams

    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False

NUM_ROTATION_PLAYERS = 8
TEAM_FEATURE_BASE_COLUMNS = [
    "W_PCT",
    "PTS",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "FG_PCT",
    "FG3_PCT",
    "FT_PCT",
    "PLUS_MINUS",
    "W",
    "L",
    "E_OFF_RATING",
    "E_DEF_RATING",
    "E_NET_RATING",
    "E_PACE",
]
CONTEXTUAL_FEATURE_COLUMNS = [
    "CONF_STRENGTH",
    "ROSTER_CONTINUITY",
]
PLAYER_FEATURE_BASE_COLUMNS = [
    "PTS",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "FG_PCT",
    "FG3_PCT",
    "FT_PCT",
    "MIN",
    "PER_APPROX",
]

TEAM_FEATURE_COLUMNS = [f"TEAM_{column}" for column in TEAM_FEATURE_BASE_COLUMNS] + CONTEXTUAL_FEATURE_COLUMNS

EASTERN_CONFERENCE_TEAMS = [
    "BOS", "BKN", "NJN", "NYK", "PHI", "TOR",
    "CHI", "CLE", "DET", "IND", "MIL",
    "ATL", "CHA", "CHO", "MIA", "ORL", "WAS",
]
WESTERN_CONFERENCE_TEAMS = [
    "DAL", "HOU", "MEM", "NOP", "NOH", "NOK", "SAS",
    "DEN", "MIN", "OKC", "SEA", "POR", "UTA",
    "GSW", "LAC", "LAL", "PHX", "SAC",
]

DATA_ROOT = Path("data")
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"
RAW_TEAM_STATS_FILE = RAW_DATA_DIR / "team_stats_all_seasons.csv"
RAW_ADVANCED_TEAM_STATS_FILE = RAW_DATA_DIR / "team_advanced_stats_all_seasons.csv"
RAW_PLAYER_STATS_FILE = RAW_DATA_DIR / "player_stats_all_seasons.csv"
RAW_MIDSEASON_TEAM_STATS_FILE = RAW_DATA_DIR / "midseason_team_stats_all_seasons.csv"
RAW_MIDSEASON_ADVANCED_TEAM_STATS_FILE = RAW_DATA_DIR / "midseason_team_advanced_stats_all_seasons.csv"
RAW_MIDSEASON_PLAYER_STATS_FILE = RAW_DATA_DIR / "midseason_player_stats_all_seasons.csv"
PROCESSED_FEATURES_FILE = PROCESSED_DATA_DIR / "features_research.csv"
PROCESSED_MIDSEASON_FEATURES_FILE = PROCESSED_DATA_DIR / "features_midseason.csv"

# Approximate All-Star Sunday cutoffs used as a mid-season information boundary.
# The 2020-21 season used a March All-Star Game because of the shortened COVID calendar.
MIDSEASON_CUTOFF_DATES = {
    "2003-04": "02/15/2004",
    "2004-05": "02/20/2005",
    "2005-06": "02/19/2006",
    "2006-07": "02/18/2007",
    "2007-08": "02/17/2008",
    "2008-09": "02/15/2009",
    "2009-10": "02/14/2010",
    "2010-11": "02/20/2011",
    "2011-12": "02/26/2012",
    "2012-13": "02/17/2013",
    "2013-14": "02/16/2014",
    "2014-15": "02/15/2015",
    "2015-16": "02/14/2016",
    "2016-17": "02/19/2017",
    "2017-18": "02/18/2018",
    "2018-19": "02/17/2019",
    "2019-20": "02/16/2020",
    "2020-21": "03/07/2021",
    "2021-22": "02/20/2022",
    "2022-23": "02/19/2023",
    "2023-24": "02/18/2024",
}


def _season_seed(season: str, offset: int = 0) -> int:
    return (sum(ord(char) for char in season) + offset) % (2**32)


def _attach_team_abbreviations(team_df: pd.DataFrame) -> pd.DataFrame:
    if "TEAM_ABBREVIATION" in team_df.columns:
        return team_df

    team_id_to_abbrev = {team["id"]: team["abbreviation"] for team in nba_teams.get_teams()}
    team_df = team_df.copy()
    team_df["TEAM_ABBREVIATION"] = team_df["TEAM_ID"].map(team_id_to_abbrev)
    return team_df


def _safe_numeric(value: object, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    return float(value)


def _load_cached_csv(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    return None


def _cached_subset(
    path: Path,
    seasons: List[str],
    force_refresh: bool = False,
) -> pd.DataFrame | None:
    if force_refresh:
        return None

    cached = _load_cached_csv(path)
    if cached is None:
        return None
    if "SEASON" not in cached.columns:
        return None
    if not set(seasons).issubset(set(cached["SEASON"].astype(str).unique())):
        return None
    return cached[cached["SEASON"].isin(seasons)].copy()


def _expected_feature_columns() -> list[str]:
    player_columns = [
        f"P{player_index}_{column}"
        for player_index in range(1, NUM_ROTATION_PLAYERS + 1)
        for column in PLAYER_FEATURE_BASE_COLUMNS
    ]
    return [
        "SEASON",
        "TEAM",
        "TEAM_ID",
        "PLAYOFF_RESULT",
        "PLAYOFF_LABEL",
        *TEAM_FEATURE_COLUMNS,
        *player_columns,
    ]


def _normalize_advanced_team_stats(advanced_stats: pd.DataFrame) -> pd.DataFrame:
    """Normalize advanced endpoint names to the estimated-metrics schema."""
    advanced_stats = advanced_stats.copy()
    rename_map = {
        "OFF_RATING": "E_OFF_RATING",
        "DEF_RATING": "E_DEF_RATING",
        "NET_RATING": "E_NET_RATING",
        "PACE": "E_PACE",
    }
    for source_column, target_column in rename_map.items():
        if target_column not in advanced_stats.columns and source_column in advanced_stats.columns:
            advanced_stats[target_column] = advanced_stats[source_column]

    for column in ["E_OFF_RATING", "E_DEF_RATING", "E_NET_RATING", "E_PACE"]:
        if column not in advanced_stats.columns:
            advanced_stats[column] = 0.0

    return advanced_stats


def _validate_midseason_cutoffs(seasons: List[str]) -> None:
    missing = [season for season in seasons if season not in MIDSEASON_CUTOFF_DATES]
    if missing:
        raise ValueError(
            "Missing mid-season cutoff date(s) for: "
            + ", ".join(missing)
        )


def download_basic_team_stats(
    seasons: List[str],
    raw_file: Path = RAW_TEAM_STATS_FILE,
    timeout: int = 60,
    sleep_seconds: float = 0.7,
) -> pd.DataFrame:
    """Download or load cached regular-season team box score statistics."""
    cached = _load_cached_csv(raw_file)
    if cached is not None:
        return cached[cached["SEASON"].isin(seasons)].copy()

    if not NBA_API_AVAILABLE:
        raise RuntimeError("nba_api is not installed.")

    raw_file.parent.mkdir(parents=True, exist_ok=True)
    all_team_stats = []

    for season in seasons:
        print(f"Fetching basic team stats for {season}...")
        stats = LeagueDashTeamStats(
            season=season,
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
            timeout=timeout,
        )
        season_df = stats.get_data_frames()[0]
        season_df["SEASON"] = season
        all_team_stats.append(season_df)
        time.sleep(sleep_seconds)

    team_stats = pd.concat(all_team_stats, ignore_index=True)
    team_stats = _attach_team_abbreviations(team_stats)
    team_stats.to_csv(raw_file, index=False)
    return team_stats


def download_advanced_team_stats(
    seasons: List[str],
    raw_file: Path = RAW_ADVANCED_TEAM_STATS_FILE,
    timeout: int = 60,
    sleep_seconds: float = 0.7,
) -> pd.DataFrame:
    """Download or load cached advanced team efficiency statistics."""
    cached = _load_cached_csv(raw_file)
    if cached is not None:
        return cached[cached["SEASON"].isin(seasons)].copy()

    if not NBA_API_AVAILABLE:
        raise RuntimeError("nba_api is not installed.")

    raw_file.parent.mkdir(parents=True, exist_ok=True)
    all_advanced_stats = []

    for season in seasons:
        print(f"Fetching advanced team stats for {season}...")
        stats = TeamEstimatedMetrics(
            season=season,
            season_type="Regular Season",
            timeout=timeout,
        )
        season_df = stats.get_data_frames()[0]
        season_df["SEASON"] = season
        all_advanced_stats.append(season_df)
        time.sleep(sleep_seconds)

    advanced_stats = pd.concat(all_advanced_stats, ignore_index=True)
    advanced_stats.to_csv(raw_file, index=False)
    return advanced_stats


def download_player_stats(
    seasons: List[str],
    raw_file: Path = RAW_PLAYER_STATS_FILE,
    timeout: int = 60,
    sleep_seconds: float = 0.7,
) -> pd.DataFrame:
    """Download or load cached regular-season player statistics."""
    cached = _load_cached_csv(raw_file)
    if cached is not None:
        return cached[cached["SEASON"].isin(seasons)].copy()

    if not NBA_API_AVAILABLE:
        raise RuntimeError("nba_api is not installed.")

    raw_file.parent.mkdir(parents=True, exist_ok=True)
    all_player_stats = []

    for season in seasons:
        print(f"Fetching player stats for {season}...")
        stats = LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
            timeout=timeout,
        )
        season_df = stats.get_data_frames()[0]
        season_df["SEASON"] = season
        all_player_stats.append(season_df)
        time.sleep(sleep_seconds)

    player_stats = pd.concat(all_player_stats, ignore_index=True)
    player_stats.to_csv(raw_file, index=False)
    return player_stats


def download_midseason_team_stats(
    seasons: List[str],
    raw_file: Path = RAW_MIDSEASON_TEAM_STATS_FILE,
    timeout: int = 60,
    sleep_seconds: float = 0.7,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download or load cached team stats through each season's mid-season cutoff."""
    cached = _cached_subset(raw_file, seasons, force_refresh=force_refresh)
    if cached is not None:
        return cached

    if not NBA_API_AVAILABLE:
        raise RuntimeError("nba_api is not installed and no cached mid-season team stats were found.")

    _validate_midseason_cutoffs(seasons)
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    all_team_stats = []

    for season in seasons:
        cutoff = MIDSEASON_CUTOFF_DATES[season]
        print(f"Fetching mid-season team stats for {season} through {cutoff}...")
        stats = LeagueDashTeamStats(
            season=season,
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
            date_to_nullable=cutoff,
            timeout=timeout,
        )
        season_df = stats.get_data_frames()[0]
        season_df["SEASON"] = season
        season_df["MIDSEASON_CUTOFF"] = cutoff
        all_team_stats.append(season_df)
        time.sleep(sleep_seconds)

    team_stats = pd.concat(all_team_stats, ignore_index=True)
    team_stats = _attach_team_abbreviations(team_stats)
    team_stats.to_csv(raw_file, index=False)
    return team_stats


def download_midseason_advanced_team_stats(
    seasons: List[str],
    raw_file: Path = RAW_MIDSEASON_ADVANCED_TEAM_STATS_FILE,
    timeout: int = 60,
    sleep_seconds: float = 0.7,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download or load cached advanced team stats through each mid-season cutoff."""
    cached = _cached_subset(raw_file, seasons, force_refresh=force_refresh)
    if cached is not None:
        return _normalize_advanced_team_stats(cached)

    if not NBA_API_AVAILABLE:
        raise RuntimeError("nba_api is not installed and no cached mid-season advanced stats were found.")

    _validate_midseason_cutoffs(seasons)
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    all_advanced_stats = []

    for season in seasons:
        cutoff = MIDSEASON_CUTOFF_DATES[season]
        print(f"Fetching mid-season advanced team stats for {season} through {cutoff}...")
        stats = LeagueDashTeamStats(
            season=season,
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
            measure_type_detailed_defense="Advanced",
            date_to_nullable=cutoff,
            timeout=timeout,
        )
        season_df = stats.get_data_frames()[0]
        season_df["SEASON"] = season
        season_df["MIDSEASON_CUTOFF"] = cutoff
        all_advanced_stats.append(season_df)
        time.sleep(sleep_seconds)

    advanced_stats = pd.concat(all_advanced_stats, ignore_index=True)
    advanced_stats = _normalize_advanced_team_stats(advanced_stats)
    advanced_stats.to_csv(raw_file, index=False)
    return advanced_stats


def download_midseason_player_stats(
    seasons: List[str],
    raw_file: Path = RAW_MIDSEASON_PLAYER_STATS_FILE,
    timeout: int = 60,
    sleep_seconds: float = 0.7,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Download or load cached player stats through each season's mid-season cutoff."""
    cached = _cached_subset(raw_file, seasons, force_refresh=force_refresh)
    if cached is not None:
        return cached

    if not NBA_API_AVAILABLE:
        raise RuntimeError("nba_api is not installed and no cached mid-season player stats were found.")

    _validate_midseason_cutoffs(seasons)
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    all_player_stats = []

    for season in seasons:
        cutoff = MIDSEASON_CUTOFF_DATES[season]
        print(f"Fetching mid-season player stats for {season} through {cutoff}...")
        stats = LeagueDashPlayerStats(
            season=season,
            per_mode_detailed="PerGame",
            season_type_all_star="Regular Season",
            date_to_nullable=cutoff,
            timeout=timeout,
        )
        season_df = stats.get_data_frames()[0]
        season_df["SEASON"] = season
        season_df["MIDSEASON_CUTOFF"] = cutoff
        all_player_stats.append(season_df)
        time.sleep(sleep_seconds)

    player_stats = pd.concat(all_player_stats, ignore_index=True)
    player_stats.to_csv(raw_file, index=False)
    return player_stats


def compute_approximate_per(player_row: pd.Series) -> float:
    """Approximate PER from box-score production per minute."""
    minutes = player_row.get("MIN", 0)
    if not minutes or pd.isna(minutes):
        return 0.0

    positive = (
        player_row.get("PTS", 0)
        + player_row.get("REB", 0)
        + player_row.get("AST", 0)
        + player_row.get("STL", 0)
        + player_row.get("BLK", 0)
    )
    negative = player_row.get("TOV", 0)
    return float((positive - negative) / minutes)


def get_conference(team_abbrev: str) -> str:
    normalized = fix_abbrev(team_abbrev)
    east = {fix_abbrev(team) for team in EASTERN_CONFERENCE_TEAMS}
    return "East" if normalized in east else "West"


def compute_roster_continuity(
    team_id: int,
    season: str,
    previous_season: str | None,
    player_data: pd.DataFrame,
    num_players: int = NUM_ROTATION_PLAYERS,
) -> float:
    """Fraction of a team's top rotation retained from the prior season."""
    current_players = player_data[
        (player_data["TEAM_ID"] == team_id) & (player_data["SEASON"] == season)
    ].sort_values("MIN", ascending=False).head(num_players)

    if len(current_players) == 0:
        return 0.5

    if previous_season is None:
        return 0.5

    previous_players = player_data[
        (player_data["TEAM_ID"] == team_id) & (player_data["SEASON"] == previous_season)
    ]

    if len(previous_players) == 0:
        return 0.5

    previous_player_ids = set(previous_players["PLAYER_ID"].tolist())
    returning = sum(1 for player_id in current_players["PLAYER_ID"].tolist() if player_id in previous_player_ids)
    return float(returning / len(current_players))


def _build_feature_matrix_from_stats(
    seasons: List[str],
    basic_team_stats: pd.DataFrame,
    advanced_team_stats: pd.DataFrame,
    player_stats: pd.DataFrame,
    processed_file: Path,
) -> pd.DataFrame:
    advanced_team_stats = _normalize_advanced_team_stats(advanced_team_stats)
    advanced_columns = ["TEAM_ID", "SEASON"] + [
        column for column in ["E_OFF_RATING", "E_DEF_RATING", "E_NET_RATING", "E_PACE"]
        if column in advanced_team_stats.columns
    ]
    all_team_stats = basic_team_stats.merge(
        advanced_team_stats[advanced_columns],
        on=["TEAM_ID", "SEASON"],
        how="left",
    )
    all_team_stats["TEAM_ABBREVIATION"] = all_team_stats["TEAM_ABBREVIATION"].astype(str)
    all_team_stats["TEAM_ABBREVIATION_CLEAN"] = all_team_stats["TEAM_ABBREVIATION"].apply(fix_abbrev)
    all_team_stats["PLAYOFF_RESULT"] = all_team_stats.apply(
        lambda row: get_playoff_label(row["TEAM_ABBREVIATION_CLEAN"], row["SEASON"]),
        axis=1,
    )
    all_team_stats["PLAYOFF_LABEL"] = all_team_stats["PLAYOFF_RESULT"].map(PLAYOFF_LABEL_NAMES)
    all_team_stats["CONFERENCE"] = all_team_stats["TEAM_ABBREVIATION_CLEAN"].apply(get_conference)

    all_player_stats = player_stats.copy()
    all_player_stats["TEAM_ABBREVIATION"] = all_player_stats["TEAM_ABBREVIATION"].astype(str)
    all_player_stats["TEAM_ABBREVIATION_CLEAN"] = all_player_stats["TEAM_ABBREVIATION"].apply(fix_abbrev)
    all_player_stats["PER_APPROX"] = all_player_stats.apply(compute_approximate_per, axis=1)

    season_pairs = {SEASONS[index]: SEASONS[index - 1] for index in range(1, len(SEASONS))}
    rows = []

    for season in seasons:
        print(f"Building engineered features for {season}...")
        season_teams = all_team_stats[all_team_stats["SEASON"] == season].copy()
        season_players = all_player_stats[all_player_stats["SEASON"] == season].copy()

        conf_strength = {}
        for conference in ("East", "West"):
            conference_teams = season_teams[season_teams["CONFERENCE"] == conference]
            conf_strength[conference] = (
                float(conference_teams["W_PCT"].mean()) if len(conference_teams) > 0 else 0.5
            )

        for _, team_row in season_teams.iterrows():
            team_abbrev = fix_abbrev(str(team_row["TEAM_ABBREVIATION"]))
            team_id = int(team_row["TEAM_ID"])
            team_conf = str(team_row["CONFERENCE"])
            team_players = season_players[season_players["TEAM_ID"] == team_id].sort_values(
                "MIN",
                ascending=False,
            )
            top_players = team_players.head(NUM_ROTATION_PLAYERS)

            row: dict[str, float | int | str] = {
                "SEASON": season,
                "TEAM": team_abbrev,
                "TEAM_ID": team_id,
                "PLAYOFF_RESULT": int(get_playoff_label(team_abbrev, season)),
                "PLAYOFF_LABEL": PLAYOFF_LABEL_NAMES[get_playoff_label(team_abbrev, season)],
            }

            for column in TEAM_FEATURE_BASE_COLUMNS:
                row[f"TEAM_{column}"] = _safe_numeric(team_row.get(column, 0.0))

            row["CONF_STRENGTH"] = conf_strength.get(team_conf, 0.5)
            row["ROSTER_CONTINUITY"] = compute_roster_continuity(
                team_id=team_id,
                season=season,
                previous_season=season_pairs.get(season),
                player_data=all_player_stats,
                num_players=NUM_ROTATION_PLAYERS,
            )

            for player_index in range(NUM_ROTATION_PLAYERS):
                if player_index < len(top_players):
                    player = top_players.iloc[player_index]
                    for column in PLAYER_FEATURE_BASE_COLUMNS:
                        row[f"P{player_index + 1}_{column}"] = _safe_numeric(player.get(column, 0.0))
                else:
                    for column in PLAYER_FEATURE_BASE_COLUMNS:
                        row[f"P{player_index + 1}_{column}"] = 0.0

            rows.append(row)

    features_df = pd.DataFrame(rows).fillna(0)
    features_df = features_df.reindex(columns=_expected_feature_columns(), fill_value=0)
    processed_file.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(processed_file, index=False)
    return features_df


def build_real_dataset(
    seasons: List[str],
    top_n_players: int = NUM_ROTATION_PLAYERS,
    processed_file: Path = PROCESSED_FEATURES_FILE,
) -> pd.DataFrame:
    """Build the real feature matrix from cached or downloaded NBA API data."""
    if top_n_players != NUM_ROTATION_PLAYERS:
        raise ValueError(
            f"This research pipeline expects top_n_players={NUM_ROTATION_PLAYERS}; "
            f"received {top_n_players}."
        )

    if processed_file.exists():
        cached_features = pd.read_csv(processed_file)
        if set(seasons).issubset(set(cached_features["SEASON"].unique())):
            return cached_features[cached_features["SEASON"].isin(seasons)].copy()

    basic_team_stats = download_basic_team_stats(SEASONS)
    advanced_team_stats = download_advanced_team_stats(SEASONS)
    player_stats = download_player_stats(SEASONS)

    if basic_team_stats.empty or advanced_team_stats.empty or player_stats.empty:
        raise RuntimeError("Unable to build the real dataset because one or more raw tables are empty.")

    return _build_feature_matrix_from_stats(
        seasons=seasons,
        basic_team_stats=basic_team_stats,
        advanced_team_stats=advanced_team_stats,
        player_stats=player_stats,
        processed_file=processed_file,
    )


def build_midseason_dataset(
    seasons: List[str],
    top_n_players: int = NUM_ROTATION_PLAYERS,
    processed_file: Path = PROCESSED_MIDSEASON_FEATURES_FILE,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Build the All-Star-break proxy feature matrix with final playoff labels."""
    if top_n_players != NUM_ROTATION_PLAYERS:
        raise ValueError(
            f"This research pipeline expects top_n_players={NUM_ROTATION_PLAYERS}; "
            f"received {top_n_players}."
        )

    if processed_file.exists() and not force_refresh:
        cached_features = pd.read_csv(processed_file)
        if set(seasons).issubset(set(cached_features["SEASON"].unique())):
            return cached_features[cached_features["SEASON"].isin(seasons)].copy()

    _validate_midseason_cutoffs(seasons)
    basic_team_stats = download_midseason_team_stats(
        seasons=seasons,
        force_refresh=force_refresh,
    )
    advanced_team_stats = download_midseason_advanced_team_stats(
        seasons=seasons,
        force_refresh=force_refresh,
    )
    player_stats = download_midseason_player_stats(
        seasons=seasons,
        force_refresh=force_refresh,
    )

    if basic_team_stats.empty or advanced_team_stats.empty or player_stats.empty:
        raise RuntimeError("Unable to build the mid-season dataset because one or more raw tables are empty.")

    return _build_feature_matrix_from_stats(
        seasons=seasons,
        basic_team_stats=basic_team_stats,
        advanced_team_stats=advanced_team_stats,
        player_stats=player_stats,
        processed_file=processed_file,
    )


def generate_mock_team_stats(season: str, n_teams: int = 30) -> pd.DataFrame:
    """Generate deterministic team-level mock data for local testing."""
    np.random.seed(_season_seed(season))
    team_names = [f"Team_{index + 1}" for index in range(n_teams)]

    data = {
        "TEAM_ID": np.arange(1000, 1000 + n_teams),
        "TEAM_NAME": team_names,
        "TEAM_ABBREVIATION": [f"T{index + 1:02d}" for index in range(n_teams)],
        "W_PCT": np.random.uniform(0.2, 0.8, n_teams),
        "PTS": np.random.uniform(95, 120, n_teams),
        "REB": np.random.uniform(38, 50, n_teams),
        "AST": np.random.uniform(18, 32, n_teams),
        "STL": np.random.uniform(5, 10, n_teams),
        "BLK": np.random.uniform(3, 8, n_teams),
        "TOV": np.random.uniform(10, 17, n_teams),
        "FG_PCT": np.random.uniform(0.42, 0.52, n_teams),
        "FG3_PCT": np.random.uniform(0.31, 0.40, n_teams),
        "FT_PCT": np.random.uniform(0.70, 0.84, n_teams),
        "PLUS_MINUS": np.random.uniform(-8, 8, n_teams),
        "W": np.random.randint(20, 62, n_teams),
        "L": np.random.randint(20, 62, n_teams),
        "E_OFF_RATING": np.random.uniform(105, 120, n_teams),
        "E_DEF_RATING": np.random.uniform(105, 120, n_teams),
        "E_NET_RATING": np.random.uniform(-10, 10, n_teams),
        "E_PACE": np.random.uniform(95, 105, n_teams),
    }
    return pd.DataFrame(data)


def generate_mock_player_stats(
    season: str,
    top_n: int = NUM_ROTATION_PLAYERS,
    n_teams: int = 30,
) -> pd.DataFrame:
    """Generate deterministic player-level mock data for local testing."""
    np.random.seed(_season_seed(season, offset=1))

    rows = []
    for team_index in range(n_teams):
        team_id = 1000 + team_index
        team_abbrev = f"T{team_index + 1:02d}"
        for player_index in range(top_n):
            rows.append(
                {
                    "PLAYER_ID": team_id * 100 + player_index,
                    "TEAM_ID": team_id,
                    "TEAM_ABBREVIATION": team_abbrev,
                    "PTS": np.random.uniform(4, 28),
                    "REB": np.random.uniform(1, 12),
                    "AST": np.random.uniform(0.5, 9),
                    "STL": np.random.uniform(0.2, 2.5),
                    "BLK": np.random.uniform(0.1, 2.2),
                    "TOV": np.random.uniform(0.3, 4.5),
                    "FG_PCT": np.random.uniform(0.38, 0.65),
                    "FG3_PCT": np.random.uniform(0.25, 0.45),
                    "FT_PCT": np.random.uniform(0.60, 0.95),
                    "MIN": np.random.uniform(8, 38),
                }
            )
    mock_player_stats = pd.DataFrame(rows)
    mock_player_stats["PER_APPROX"] = mock_player_stats.apply(compute_approximate_per, axis=1)
    return mock_player_stats


def build_mock_dataset(
    seasons: List[str],
    top_n_players: int = NUM_ROTATION_PLAYERS,
) -> pd.DataFrame:
    """Build a schema-compatible mock dataset when nba_api data is unavailable."""
    rows = []
    for season in seasons:
        team_stats = generate_mock_team_stats(season)
        player_stats = generate_mock_player_stats(season, top_n=top_n_players)
        player_stats["SEASON"] = season
        team_stats["SEASON"] = season

        for _, team_row in team_stats.iterrows():
            team_id = int(team_row["TEAM_ID"])
            team_abbrev = str(team_row["TEAM_ABBREVIATION"])
            top_players = player_stats[player_stats["TEAM_ID"] == team_id].sort_values("MIN", ascending=False)

            row: dict[str, float | int | str] = {
                "SEASON": season,
                "TEAM": team_abbrev,
                "TEAM_ID": team_id,
                "PLAYOFF_RESULT": int(np.random.choice(6, p=[0.45, 0.20, 0.15, 0.10, 0.07, 0.03])),
            }
            row["PLAYOFF_LABEL"] = PLAYOFF_LABEL_NAMES[row["PLAYOFF_RESULT"]]

            for column in TEAM_FEATURE_BASE_COLUMNS:
                row[f"TEAM_{column}"] = _safe_numeric(team_row.get(column, 0.0))

            row["CONF_STRENGTH"] = 0.5
            row["ROSTER_CONTINUITY"] = 0.5

            for player_index in range(NUM_ROTATION_PLAYERS):
                player = top_players.iloc[player_index]
                for column in PLAYER_FEATURE_BASE_COLUMNS:
                    row[f"P{player_index + 1}_{column}"] = _safe_numeric(player.get(column, 0.0))

            rows.append(row)

    return pd.DataFrame(rows).fillna(0)


def build_dataset(
    seasons: Optional[List[str]] = None,
    top_n_players: int = NUM_ROTATION_PLAYERS,
    save_path: Optional[str] = None,
    allow_mock_fallback: bool = True,
) -> pd.DataFrame:
    """Build the dataset used by the package.

    When the NBA API and cached downloads are available, this path builds the
    real historical feature matrix used for the research study. Otherwise the
    function falls back to a schema-compatible mock dataset.
    """
    seasons = seasons or SEASONS
    processed_file = Path(save_path) if save_path else PROCESSED_FEATURES_FILE

    try:
        dataset = build_real_dataset(
            seasons=seasons,
            top_n_players=top_n_players,
            processed_file=processed_file,
        )
    except Exception as exc:
        if not allow_mock_fallback:
            raise RuntimeError(
                "Failed to build the real dataset and mock fallback is disabled."
            ) from exc
        print(f"Warning: failed to build real dataset ({exc}). Falling back to mock data.")
        dataset = build_mock_dataset(seasons=seasons, top_n_players=top_n_players)

    print("\nDataset Summary:")
    print(f"  Total team-seasons: {len(dataset)}")
    print(f"  Seasons: {dataset['SEASON'].nunique()}")
    print(f"  Feature columns: {len(dataset.columns) - 5}")
    print("\nPlayoff Outcome Distribution:")
    outcome_counts = dataset["PLAYOFF_RESULT"].value_counts().sort_index()
    for label, count in outcome_counts.items():
        share = count / len(dataset) * 100
        print(f"  Class {label}: {count} ({share:.1f}%)")

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        dataset.to_csv(save_path, index=False)
        print(f"\nSaved dataset to {save_path}")

    return dataset


def _is_player_feature_column(column_name: str) -> bool:
    return bool(re.fullmatch(r"P\d+_.+", column_name))


def get_feature_column_groups(dataset: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return team/context and player feature columns."""
    team_columns = [column for column in TEAM_FEATURE_COLUMNS if column in dataset.columns]
    player_columns = [column for column in dataset.columns if _is_player_feature_column(column)]
    return team_columns, player_columns


def prepare_model_inputs(dataset: pd.DataFrame) -> Dict[str, np.ndarray | list[str] | int]:
    """Prepare a flat feature matrix plus metadata for experiments."""
    team_columns, player_columns = get_feature_column_groups(dataset)
    X = dataset[team_columns + player_columns].to_numpy(dtype=np.float32)
    y = dataset["PLAYOFF_RESULT"].to_numpy(dtype=np.int64)
    seasons_array = dataset["SEASON"].to_numpy(dtype=str)

    return {
        "X": X,
        "y": y,
        "seasons": seasons_array,
        "teams": dataset["TEAM"].astype(str).to_numpy(dtype=str),
        "team_columns": team_columns,
        "player_columns": player_columns,
        "num_team_features": len(team_columns),
        "num_player_stats": len(player_columns) // NUM_ROTATION_PLAYERS,
    }


def prepare_tensors(dataset: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Convert the engineered dataframe into tensors used by the package CLI."""
    prepared = prepare_model_inputs(dataset)
    team_columns = prepared["team_columns"]
    player_columns = prepared["player_columns"]

    return {
        "team_features": dataset[team_columns].to_numpy(dtype=np.float32),
        "player_features": dataset[player_columns].to_numpy(dtype=np.float32),
        "labels": dataset["PLAYOFF_RESULT"].to_numpy(dtype=np.int64),
        "seasons": dataset["SEASON"].astype(str).str.slice(0, 4).astype(np.int64).to_numpy(),
        "team_ids": dataset["TEAM"].astype(str).to_numpy(dtype=str),
    }
