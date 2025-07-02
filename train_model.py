# mlb_strikeout_app/train_model.py

import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# 1. Load data from CSVs
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
pitcher_file        = os.path.join(DATA_DIR, 'MLB_2025_Pitching_Data.csv')
opponent_stats_file = os.path.join(DATA_DIR, 'opponent_k_pct.csv')  # team_name, opp_K_pct

pitcher_df = pd.read_csv(pitcher_file)
opp_df     = pd.read_csv(opponent_stats_file)

# 2. Rename and select features in pitcher data
pitcher_df = pitcher_df.rename(columns={
    'Name': 'PlayerName',
    'Name-additional': 'bbref_id',
    'Team': 'Tm',
    'K/9': 'K9',
    'SwStr %': 'SwStr_pct',
    'CS %': 'CS_pct',
    'Fastball Spin': 'FB_spin',
    'Fastball Velocity': 'FB_vel',
    'FB_pct': 'FB_mix',
    'SL_pct': 'SL_mix',
    'CH_pct': 'CH_mix',
    'Release Extension': 'Rel_ext'
})

# 3. Merge in opponent context
#    (park factors and rest days are omitted)
df = pitcher_df.merge(
    opp_df,
    left_on='Tm', right_on='team_name', how='left'
)

# 4. Prepare target and feature matrix
target = 'SO'
features = [
    'K9', 'SwStr_pct', 'CS_pct',
    'FB_spin', 'FB_vel',
    'FB_mix', 'SL_mix', 'CH_mix', 'Rel_ext',
    'opp_K_pct'
]

X = df[features]
y = df[target]

# 5. Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 6. Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
val_data   = lgb.Dataset(X_val,   label=y_val, reference=train_data)

# 7. Train LightGBM model
params = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42
}

model = lgb.train(
    params,
    train_data,
    num_boost_round       = 1000,
    valid_sets            = [train_data, val_data],
    early_stopping_rounds = 50,
    verbose_eval          = 50
)

# 8. Evaluate performance
y_pred = model.predict(X_val, num_iteration=model.best_iteration)
rmse   = mean_squared_error(y_val, y_pred, squared=False)
print(f"Validation RMSE: {rmse:.3f}")

# 9. Save trained model
model_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'lgb_so_model.pkl')
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
