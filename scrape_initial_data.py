# scrape_initial_data.py
# Initial data pull: fetch season stats from FanGraphs and pitch-level data from Statcast

import os
import pandas as pd
import unicodedata
from datetime import datetime
from pybaseball import pitching_stats, statcast, playerid_lookup, team_batting

# Helper to remove accents/diacritics
def strip_accents(s: str) -> str:
    if not isinstance(s, str):
        return s
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# File paths
data_dir = os.path.join(os.path.dirname(__file__), 'data')
csv_out  = os.path.join(data_dir, 'pitching_data.csv')

# 1. Pull season-level stats from FanGraphs
print("Fetching season stats from FanGraphs...")
fg = pitching_stats(2025, 2025)

# Ensure 'GS' column present for games started
if 'GS' not in fg.columns:
    raise KeyError("Expected 'GS' column for games started in FanGraphs data, got: {}".format(fg.columns.tolist()))
# Filter to starting pitchers (e.g., at least 5 starts)
fg = fg[fg['GS'] >= 5]

# Identify batters faced column dynamically
bf_candidates = ['BFP', 'BF', 'TBF']
bf_field = next((c for c in bf_candidates if c in fg.columns), None)
if bf_field is None:
    raise KeyError(f"None of {bf_candidates} found in FanGraphs data columns: {fg.columns.tolist()}")

# Select relevant fields and rename 'batters faced' to 'BF'
select_cols = ['Name', 'Team', 'IP', bf_field, 'SO', 'G']
fg = fg[select_cols]
fg = fg.rename(columns={
    'Name': 'Name',
    'Team': 'Team',
    'IP': 'IP',
    bf_field: 'BF',    # Batters Faced
    'SO': 'SO',        # Total strikeouts
    'G': 'G'           # Games pitched
})

# Compute SO per game
fg['SO'] = fg['SO'] / fg['G']

# Lookup BBRef IDs
print("Looking up BBRef IDs...")
def get_bbref_id(name):
    try:
        parts = name.split()
        lookup = playerid_lookup(parts[-1], parts[0])
        return lookup.loc[0, 'key_bbref']
    except Exception:
        return None

fg['Name-additional'] = fg['Name'].apply(get_bbref_id)

# 2. Pull pitch-level data from Statcast for the full season
today    = datetime.today().strftime('%Y-%m-%d')
start_dt = '2025-03-01'
print(f"Fetching Statcast data from {start_dt} to {today}...")
sc = statcast(start_dt, today)

# Filter to needed columns
enum = sc[['player_name', 'release_speed', 'release_spin_rate', 'release_extension', 'description']]

# 3. Aggregate Statcast features per pitcher
print("Aggregating pitch-level features...")
def aggregate_features(group):
    total    = len(group)
    sw_str   = group['description'].str.contains('swinging_strike', case=False).sum()
    call_str = group['description'].str.contains('called_strike', case=False).sum()
    foul     = group['description'].str.contains('foul', case=False).sum()
    return pd.Series({
        'Str_pct':   (sw_str + call_str + foul) / total if total else 0,
        'SwStr_pct': sw_str / total if total else 0,
        'CS_pct':    call_str / total if total else 0,
        'FB_vel':    group['release_speed'].mean(),
        'FB_spin':   group['release_spin_rate'].mean(),
        'Rel_ext':   group['release_extension'].mean()
    })

agg = (
    enum
      .groupby('player_name')
      .apply(aggregate_features)
      .reset_index()
      .rename(columns={'player_name': 'Name'})
)

# 4. Standardize names for merge ("Last, First" → "First Last") and strip accents
def standardize_name(name: str) -> str:
    if ',' in name:
        last, first = name.split(', ')
        return f"{first} {last}"
    return name

fg['Name_std'] = (
    fg['Name']
      .apply(standardize_name)
      .str.normalize('NFC')
      .apply(strip_accents)
)
agg['Name_std'] = (
    agg['Name']
       .apply(standardize_name)
       .str.normalize('NFC')
       .apply(strip_accents)
)

# 5. Merge
print("Merging datasets on Name_std with left join...")
df = pd.merge(
    fg,
    agg,
    on='Name_std',
    how='left',
    suffixes=('', '_stat')
)

# DEBUG: verify that 'BF' is present
print("Columns I actually have:", df.columns.tolist())

# 6. Fill missing pitch-level stats
agg_cols = ['Str_pct', 'SwStr_pct', 'CS_pct', 'FB_vel', 'FB_spin', 'Rel_ext']
df[agg_cols] = df[agg_cols].fillna(0)

# 7. Compute derived ratios
print("Computing IP_G and BF_IP...")
df['IP_G']  = df['IP'] / df['G']
df['BF_IP'] = df['BF'] / df['IP']

# 8. Save to CSV
print(f"Saving initial dataset to {csv_out}...")
os.makedirs(data_dir, exist_ok=True)
df.to_csv(csv_out, index=False)
print("Done With Pitching Data.")

# --- Additional Feature Engineering: Rolling Averages and Home/Away Splits ---
print("Computing rolling averages and home/away splits from Statcast data...")

# Ensure Statcast has game_date, home_team, away_team
if not all(col in sc.columns for col in ['game_date', 'home_team', 'away_team']):
    print("Statcast data missing game_date, home_team, or away_team. Skipping advanced features.")
else:
    # Prepare per-game aggregates for each pitcher
    sc['game_date'] = pd.to_datetime(sc['game_date'])
    game_groups = sc.groupby(['player_name', 'game_date', 'home_team', 'away_team'])
    def per_game_agg(group):
        sw_str   = group['description'].str.contains('swinging_strike', case=False).sum()
        call_str = group['description'].str.contains('called_strike', case=False).sum()
        foul     = group['description'].str.contains('foul', case=False).sum()
        so       = group['description'].str.contains('strikeout', case=False).sum()
        total    = len(group)
        return pd.Series({
            'Str_pct':   (sw_str + call_str + foul) / total if total else 0,
            'SwStr_pct': sw_str / total if total else 0,
            'CS_pct':    call_str / total if total else 0,
            'FB_vel':    group['release_speed'].mean(),
            'FB_spin':   group['release_spin_rate'].mean(),
            'Rel_ext':   group['release_extension'].mean(),
            'SO':        so,
            'Pitches':   total
        })
    per_game = game_groups.apply(per_game_agg).reset_index()

    # Rolling averages (last 3 games) for each pitcher
    rolling_features = ['SO', 'Str_pct', 'SwStr_pct', 'CS_pct', 'FB_vel', 'FB_spin', 'Rel_ext']
    rolling_stats = []
    for name, group in per_game.groupby('player_name'):
        group = group.sort_values('game_date')
        roll = group[rolling_features].rolling(window=3, min_periods=1).mean()
        roll['player_name'] = name
        roll['game_date'] = group['game_date'].values
        rolling_stats.append(roll)
    rolling_df = pd.concat(rolling_stats, ignore_index=True)
    rolling_df = rolling_df[['player_name', 'game_date'] + rolling_features]
    rolling_df.to_csv(os.path.join(data_dir, 'pitcher_rolling_stats.csv'), index=False)
    print("Saved rolling averages to data/pitcher_rolling_stats.csv")

    # Home/Away splits for each pitcher
    def home_away(row):
        # If pitcher's team is home_team, it's home; else away
        # Try to infer from player_name and home_team/away_team
        # This is a best effort; for perfect accuracy, use player_id
        return 'home' if row['home_team'] in row['player_name'] else 'away'
    # Instead, use home_team == team for each appearance
    # First, get mapping from player_name to team from main df
    name_to_team = dict(zip(df['Name'], df['Team']))
    per_game['pitcher_team'] = per_game['player_name'].map(name_to_team)
    per_game['home_or_away'] = per_game.apply(lambda r: 'home' if r['pitcher_team'] == r['home_team'] else 'away', axis=1)
    home_away_agg = per_game.groupby(['player_name', 'home_or_away'])[rolling_features].mean().reset_index()
    home_away_agg.to_csv(os.path.join(data_dir, 'pitcher_home_away_stats.csv'), index=False)
    print("Saved home/away splits to data/pitcher_home_away_stats.csv")

# --- Scrape opponent (team) K% and save to CSV ---
print("Fetching team batting stats for opponent K%...")
try:
    team_bat = team_batting(2025)  # Use 2024 if 2025 is not available
except Exception as e:
    print(f"Failed to fetch 2025 team batting stats, error: {e}. Trying 2024...")
    team_bat = team_batting(2024)

# Inspect columns to find K% (may be 'K%' or 'SO%')
# print("Team batting columns:", team_bat.columns.tolist())

# Try to find the right column for strikeout rate
k_col = None
for col in ['K%', 'SO%', 'SO/P']:
    if col in team_bat.columns:
        k_col = col
        break
if k_col is None:
    raise KeyError("Could not find a K% column in team batting stats.")

# Build DataFrame with team name and K%
opp_k = team_bat[['Team', k_col]].copy()
opp_k = opp_k.rename(columns={'Team': 'team_name', k_col: 'opp_K_pct'})

# Convert K% to decimal if needed (e.g., '23.5%' -> 0.235)
if opp_k['opp_K_pct'].dtype == object:
    opp_k['opp_K_pct'] = opp_k['opp_K_pct'].str.rstrip('%').astype(float) / 100

# Save to CSV
opp_k.to_csv(os.path.join(data_dir, 'opponent_k_pct.csv'), index=False)
print("Saved opponent K% to data/opponent_k_pct.csv")
