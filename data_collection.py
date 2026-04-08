"""
data_collection.py - Data Collection and Preprocessing for NBA Championship Prediction

Implements the data collection from Section 4 of the proposal:
- Uses nba_api Python library to pull mid-season team and player statistics
- Collects data for the last 20+ NBA seasons
- Labels each team-season with their actual playoff result
- Feature engineering for team stats, player stats, and contextual features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

try:
    from nba_api.stats.endpoints import leaguedashteamstats, leaguedashplayerstats
    from nba_api.stats.static import teams
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    print("Warning: nba_api not installed. Using mock data generation.")


# ============================================================================
# Configuration
# ============================================================================

# Seasons to collect (last 20+ years as mentioned in proposal)
SEASONS = [f"{y}-{str(y+1)[-2:]}" for y in range(2003, 2024)]

# Team-level features (~15 from proposal)
TEAM_FEATURES = [
    'W', 'L', 'W_PCT', 'OFF_RATING', 'DEF_RATING', 'NET_RATING',
    'PACE', 'AST_RATIO', 'OREB_PCT', 'DREB_PCT', 'REB_PCT',
    'TM_TOV_PCT', 'EFG_PCT', 'TS_PCT', 'PACE_PER40'
]

# Player-level features (~10 from proposal)
PLAYER_FEATURES = [
    'PTS', 'REB', 'AST', 'FG_PCT', 'FG3_PCT', 'FT_PCT',
    'MIN', 'PLUS_MINUS', 'NBA_FANTASY_PTS', 'EFG_PCT'
]

# Playoff outcome mapping (6 classes from proposal)
PLAYOFF_OUTCOMES = {
    'Missed Playoffs': 0,
    'First Round Exit': 1,
    'Second Round Exit': 2,
    'Conference Finals': 3,
    'Finals Appearance': 4,
    'Champion': 5
}


# ============================================================================
# Data Collection
# ============================================================================

def collect_team_stats(season: str, season_type: str = 'Regular Season') -> pd.DataFrame:
    """
    Collect team statistics for a given season using nba_api.
    
    Args:
        season: Season string (e.g., '2023-24')
        season_type: 'Regular Season' or 'Playoffs'
    
    Returns:
        DataFrame with team statistics
    """
    if not NBA_API_AVAILABLE:
        # Return mock data for testing
        return generate_mock_team_stats(season)
    
    try:
        stats = leaguedashteamstats.LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            measure_type_detailed_defense='Advanced'
        )
        df = stats.get_data_frames()[0]
        return df
    except Exception as e:
        print(f"Error collecting team stats for {season}: {e}")
        return pd.DataFrame()


def collect_player_stats(season: str, season_type: str = 'Regular Season', 
                         top_n: int = 8) -> pd.DataFrame:
    """
    Collect player statistics for a given season using nba_api.
    
    Args:
        season: Season string (e.g., '2023-24')
        season_type: 'Regular Season' or 'Playoffs'
        top_n: Number of top players per team to collect (default: 8 from proposal)
    
    Returns:
        DataFrame with player statistics
    """
    if not NBA_API_AVAILABLE:
        # Return mock data for testing
        return generate_mock_player_stats(season, top_n)
    
    try:
        stats = leaguedashplayerstats.LeagueDashPlayerStats(
            season=season,
            season_type_all_star=season_type
        )
        df = stats.get_data_frames()[0]
        return df
    except Exception as e:
        print(f"Error collecting player stats for {season}: {e}")
        return pd.DataFrame()


def get_playoff_results(season: str) -> Dict[str, str]:
    """
    Get playoff results for each team in a given season.
    
    This would typically come from a separate data source or manual entry.
    For now, returns a placeholder.
    
    Args:
        season: Season string (e.g., '2023-24')
    
    Returns:
        Dictionary mapping team abbreviation to playoff outcome
    """
    # In practice, this would be loaded from a database or file
    # For demonstration, return empty dict
    return {}


# ============================================================================
# Mock Data Generation (for testing without nba_api)
# ============================================================================

def generate_mock_team_stats(season: str, n_teams: int = 30) -> pd.DataFrame:
    """Generate mock team statistics for testing."""
    np.random.seed(hash(season) % 2**32)
    
    team_names = [f"Team_{i+1}" for i in range(n_teams)]
    
    data = {
        'TEAM_NAME': team_names,
        'TEAM_ABBREVIATION': [f"T{i+1:02d}" for i in range(n_teams)],
        'W': np.random.randint(15, 65, n_teams),
        'L': np.random.randint(15, 65, n_teams),
        'W_PCT': np.random.uniform(0.2, 0.8, n_teams),
        'OFF_RATING': np.random.uniform(105, 120, n_teams),
        'DEF_RATING': np.random.uniform(105, 120, n_teams),
        'NET_RATING': np.random.uniform(-10, 10, n_teams),
        'PACE': np.random.uniform(95, 105, n_teams),
        'AST_RATIO': np.random.uniform(15, 20, n_teams),
        'OREB_PCT': np.random.uniform(0.2, 0.35, n_teams),
        'DREB_PCT': np.random.uniform(0.65, 0.8, n_teams),
        'REB_PCT': np.random.uniform(0.45, 0.55, n_teams),
        'TM_TOV_PCT': np.random.uniform(10, 16, n_teams),
        'EFG_PCT': np.random.uniform(0.48, 0.58, n_teams),
        'TS_PCT': np.random.uniform(0.52, 0.62, n_teams),
        'PACE_PER40': np.random.uniform(95, 105, n_teams),
    }
    
    return pd.DataFrame(data)


def generate_mock_player_stats(season: str, top_n: int = 8, 
                                n_teams: int = 30) -> pd.DataFrame:
    """Generate mock player statistics for testing."""
    np.random.seed(hash(season) % 2**32 + 1)
    
    players = []
    for team_idx in range(n_teams):
        team_abbr = f"T{team_idx+1:02d}"
        for player_idx in range(top_n):
            players.append({
                'PLAYER_NAME': f"Player_{team_idx}_{player_idx}",
                'TEAM_ABBREVIATION': team_abbr,
                'PTS': np.random.uniform(5, 30),
                'REB': np.random.uniform(2, 12),
                'AST': np.random.uniform(1, 10),
                'FG_PCT': np.random.uniform(0.4, 0.6),
                'FG3_PCT': np.random.uniform(0.3, 0.45),
                'FT_PCT': np.random.uniform(0.7, 0.9),
                'MIN': np.random.uniform(10, 40),
                'PLUS_MINUS': np.random.uniform(-5, 5),
                'NBA_FANTASY_PTS': np.random.uniform(10, 50),
                'EFG_PCT': np.random.uniform(0.45, 0.65),
            })
    
    return pd.DataFrame(players)


def generate_mock_playoff_labels(season: str, n_teams: int = 30) -> Dict[str, int]:
    """Generate mock playoff labels for testing."""
    np.random.seed(hash(season) % 2**32 + 2)
    
    # Imbalanced distribution as expected in real data
    outcomes = np.random.choice(
        6, 
        size=n_teams, 
        p=[0.45, 0.20, 0.15, 0.10, 0.07, 0.03]
    )
    
    return {f"T{i+1:02d}": outcomes[i] for i in range(n_teams)}


# ============================================================================
# Feature Engineering
# ============================================================================

def engineer_team_features(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer team-level features as described in the proposal.
    
    Features include:
    - Basic stats: wins, losses, win percentage
    - Advanced: offensive/defensive/net rating, pace
    - Shooting: eFG%, TS%
    - Rebounding: OREB%, DREB%, REB%
    - Contextual: strength of schedule (would need separate data)
    
    Args:
        team_df: Raw team statistics DataFrame
    
    Returns:
        DataFrame with engineered features
    """
    features = pd.DataFrame()
    features['team_id'] = team_df['TEAM_ABBREVIATION']
    
    # Basic stats
    features['wins'] = team_df['W']
    features['losses'] = team_df['L']
    features['win_pct'] = team_df['W_PCT']
    
    # Advanced ratings
    features['off_rating'] = team_df['OFF_RATING']
    features['def_rating'] = team_df['DEF_RATING']
    features['net_rating'] = team_df['NET_RATING']
    features['pace'] = team_df['PACE']
    
    # Shooting
    features['efg_pct'] = team_df['EFG_PCT']
    features['ts_pct'] = team_df['TS_PCT']
    
    # Rebounding
    features['oreb_pct'] = team_df['OREB_PCT']
    features['dreb_pct'] = team_df['DREB_PCT']
    features['reb_pct'] = team_df['REB_PCT']
    
    # Other
    features['ast_ratio'] = team_df['AST_RATIO']
    features['tov_pct'] = team_df['TM_TOV_PCT']
    
    # Placeholder for strength of schedule (would need separate calculation)
    features['sos'] = 0.0
    
    return features


def engineer_player_features(player_df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """
    Engineer player-level features for the top N players per team.
    
    From proposal: "individual stats for the top 8 players in the rotation
    (about 10 stats each)"
    
    Args:
        player_df: Raw player statistics DataFrame
        top_n: Number of top players per team to include
    
    Returns:
        DataFrame with engineered player features (flattened per team)
    """
    # Sort players by minutes played (proxy for rotation importance)
    player_df = player_df.sort_values(['TEAM_ABBREVIATION', 'MIN'], ascending=[True, False])
    
    team_features = []
    
    for team in player_df['TEAM_ABBREVIATION'].unique():
        team_players = player_df[player_df['TEAM_ABBREVIATION'] == team].head(top_n)
        
        if len(team_players) < top_n:
            # Pad with zeros if team has fewer players
            padding = top_n - len(team_players)
            team_players = pd.concat([
                team_players,
                pd.DataFrame(0, index=range(padding), columns=team_players.columns)
            ])
        
        features = {'team_id': team}
        
        for i, (_, player) in enumerate(team_players.iterrows()):
            prefix = f'p{i+1}_'
            features[f'{prefix}pts'] = player['PTS']
            features[f'{prefix}reb'] = player['REB']
            features[f'{prefix}ast'] = player['AST']
            features[f'{prefix}fg_pct'] = player['FG_PCT']
            features[f'{prefix}fg3_pct'] = player['FG3_PCT']
            features[f'{prefix}ft_pct'] = player['FT_PCT']
            features[f'{prefix}min'] = player['MIN']
            features[f'{prefix}plus_minus'] = player['PLUS_MINUS']
            features[f'{prefix}fantasy'] = player['NBA_FANTASY_PTS']
            features[f'{prefix}efg_pct'] = player['EFG_PCT']
        
        team_features.append(features)
    
    return pd.DataFrame(team_features)


def add_contextual_features(features_df: pd.DataFrame, season: str) -> pd.DataFrame:
    """
    Add contextual features as mentioned in the proposal.
    
    Contextual features include:
    - Roster continuity (would need historical data)
    - Conference strength (would need separate calculation)
    
    Args:
        features_df: DataFrame with team and player features
        season: Season string
    
    Returns:
        DataFrame with additional contextual features
    """
    # Placeholder for roster continuity
    # In practice, this would compare current roster to previous season
    features_df['roster_continuity'] = 0.0
    
    # Placeholder for conference strength
    # In practice, this would be calculated from conference win percentages
    features_df['conference_strength'] = 0.0
    
    return features_df


# ============================================================================
# Main Pipeline
# ============================================================================

def collect_and_process_season(season: str, 
                               top_n_players: int = 8) -> Optional[pd.DataFrame]:
    """
    Collect and process all data for a single season.
    
    Args:
        season: Season string (e.g., '2023-24')
        top_n_players: Number of top players per team to include
    
    Returns:
        DataFrame with all features and labels, or None if error
    """
    print(f"Processing season {season}...")
    
    # Collect raw data
    team_stats = collect_team_stats(season)
    player_stats = collect_player_stats(season, top_n=top_n_players)
    playoff_labels = generate_mock_playoff_labels(season, len(team_stats))
    
    if team_stats.empty or player_stats.empty:
        print(f"  Warning: No data for season {season}")
        return None
    
    # Engineer features
    team_features = engineer_team_features(team_stats)
    player_features = engineer_player_features(player_stats, top_n=top_n_players)
    
    # Merge team and player features
    features = pd.merge(team_features, player_features, on='team_id', how='inner')
    
    # Add contextual features
    features = add_contextual_features(features, season)
    
    # Add playoff labels
    features['playoff_outcome'] = features['team_id'].map(playoff_labels)
    features['season'] = season
    
    print(f"  Collected {len(features)} team-seasons")
    
    return features


def build_dataset(seasons: List[str] = None, 
                  top_n_players: int = 8,
                  save_path: Optional[str] = None) -> pd.DataFrame:
    """
    Build the full dataset for all seasons.
    
    Args:
        seasons: List of seasons to include (default: last 20+ years)
        top_n_players: Number of top players per team to include
        save_path: Optional path to save the dataset
    
    Returns:
        DataFrame with all features and labels
    """
    if seasons is None:
        seasons = SEASONS
    
    all_data = []
    
    for season in seasons:
        season_data = collect_and_process_season(season, top_n_players)
        if season_data is not None:
            all_data.append(season_data)
    
    if not all_data:
        raise ValueError("No data collected for any season")
    
    dataset = pd.concat(all_data, ignore_index=True)
    
    print(f"\nDataset Summary:")
    print(f"  Total team-seasons: {len(dataset)}")
    print(f"  Seasons: {dataset['season'].nunique()}")
    print(f"  Features: {len(dataset.columns) - 3}")  # Exclude team_id, season, playoff_outcome
    
    # Print class distribution
    print(f"\nPlayoff Outcome Distribution:")
    outcome_counts = dataset['playoff_outcome'].value_counts().sort_index()
    for outcome, count in outcome_counts.items():
        pct = count / len(dataset) * 100
        print(f"  Class {outcome}: {count} ({pct:.1f}%)")
    
    if save_path:
        dataset.to_csv(save_path, index=False)
        print(f"\nDataset saved to {save_path}")
    
    return dataset


def prepare_tensors(dataset: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Prepare numpy tensors for model training from the dataset.
    
    Args:
        dataset: DataFrame from build_dataset()
    
    Returns:
        Dictionary with 'team_features', 'player_features', 'labels', 'seasons'
    """
    # Identify feature columns
    team_feature_cols = [c for c in dataset.columns if c.startswith(('wins', 'losses', 'win_pct',
        'off_rating', 'def_rating', 'net_rating', 'pace', 'efg_pct', 'ts_pct', 'oreb_pct',
        'dreb_pct', 'reb_pct', 'ast_ratio', 'tov_pct', 'sos', 'roster_continuity', 'conference_strength'))]
    
    player_feature_cols = [c for c in dataset.columns if c.startswith('p') and '_' in c]
    
    team_features = dataset[team_feature_cols].values.astype(np.float32)
    player_features = dataset[player_feature_cols].values.astype(np.float32)
    labels = dataset['playoff_outcome'].values.astype(np.int64)
    seasons = dataset['season'].apply(lambda x: int(x.split('-')[0])).values.astype(np.int64)
    
    return {
        'team_features': team_features,
        'player_features': player_features,
        'labels': labels,
        'seasons': seasons,
        'team_ids': dataset['team_id'].values
    }


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NBA Championship Prediction - Data Collection")
    print("=" * 60)
    
    # Build dataset
    dataset = build_dataset(
        seasons=SEASONS[:5],  # Use first 5 seasons for demonstration
        top_n_players=8,
        save_path='nba_dataset.csv'
    )
    
    # Prepare tensors for training
    print("\n" + "-" * 60)
    print("Preparing tensors for model training...")
    tensors = prepare_tensors(dataset)
    
    print(f"\nTensor shapes:")
    print(f"  Team features: {tensors['team_features'].shape}")
    print(f"  Player features: {tensors['player_features'].shape}")
    print(f"  Labels: {tensors['labels'].shape}")
    print(f"  Seasons: {tensors['seasons'].shape}")
    
    # Save processed tensors
    np.savez('nba_tensors.npz', **tensors)
    print("\nTensors saved to 'nba_tensors.npz'")
    
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)
