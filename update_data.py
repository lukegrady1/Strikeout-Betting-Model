# update_data.py
# This script fetches new game logs for each pitcher and updates the master CSV

import os
import pandas as pd
from pybaseball import statcast_pitcher, playerid_reverse_lookup
from datetime import datetime, timedelta

# Constants
data_dir = os.path.join(os.path.dirname(__file__), 'data')
master_csv = os.path.join(data_dir, 'pitching_data.csv')

# Helper: get MLBAM player ID from BBRef ID
def bbref_to_mlbam(bbref_id):
    try:
        player_info = playerid_reverse_lookup(bbref_id)
        return int(player_info.loc[0, 'key_mlbam'])
    except Exception:
        return None

# Fetch new Statcast logs for a pitcher since last_date
def fetch_new_game_logs(bbref_id, last_date):
    mlbam_id = bbref_to_mlbam(bbref_id)
    if mlbam_id is None:
        return pd.DataFrame()

    start = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
    end = datetime.today().strftime('%Y-%m-%d')
    if start > end:
        return pd.DataFrame()

    try:
        df = statcast_pitcher(start_dt=start, end_dt=end, player_id=mlbam_id)
        return df
    except Exception:
        return pd.DataFrame()

# Compute aggregated features from daily Statcast logs
def compute_features(logs: pd.DataFrame) -> pd.Series:
    if logs.empty:
        return pd.Series()

    # Example aggregates; adapt column names as needed
    games = logs['game_pk'].nunique()
    so = logs['events'].eq('strikeout').sum()
    bf = logs.shape[0]
    ip = logs['outs'].sum() / 3
    sw_str = logs['description'].str.contains('swinging strike', case=False).sum()
    total_p = logs.shape[0]

    return pd.Series({
        'SO': so / games if games else 0,
        'BF_IP': bf / ip if ip else 0,
        'IP_G': ip / games if games else 0,
        'SwStr %': sw_str / total_p if total_p else 0,
        'LastUpdate': datetime.today().strftime('%Y-%m-%d')
    })

# Main update routine
def main():
    # Load master CSV; index by bbref_id
    master = pd.read_csv(master_csv, parse_dates=['LastUpdate'], index_col='Name-additional')
    updated = {}

    for bbref_id, row in master.iterrows():
        last_date = row['LastUpdate'] if 'LastUpdate' in row and not pd.isna(row['LastUpdate']) else datetime(2025,3,1)
        logs = fetch_new_game_logs(bbref_id, last_date)
        feats = compute_features(logs)
        if not feats.empty:
            updated[bbref_id] = feats

    # Apply updates
    for bbref_id, feats in updated.items():
        for col, val in feats.items():
            master.at[bbref_id, col] = val

    # Write back to CSV
    master.reset_index().to_csv(master_csv, index=False)
    print(f"Updated {len(updated)} pitchers and saved to {master_csv}")

if __name__ == '__main__':
    main()
