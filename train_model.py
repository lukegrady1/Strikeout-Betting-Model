# mlb_strikeout_app/train_model.py

import os
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
from math import sqrt

# 1. Load data from CSVs
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
pitcher_file        = os.path.join(DATA_DIR, 'pitching_data.csv')
opponent_stats_file = os.path.join(DATA_DIR, 'opponent_k_pct.csv')  # team_name, opp_K_pct

pitcher_df = pd.read_csv(pitcher_file)
opp_df     = pd.read_csv(opponent_stats_file)

# 2. Merge in opponent context
#    (park factors and rest days are omitted)
df = pitcher_df.merge(
    opp_df,
    left_on='Team', right_on='team_name', how='left'
)

# 3. Feature engineering and selection
#    Use all relevant features, drop those with too many missing values or low variance
features = [
    'Str_pct', 'SwStr_pct', 'CS_pct',
    'FB_vel', 'FB_spin', 'Rel_ext',
    'IP_G', 'BF_IP',
    'opp_K_pct'
]
target = 'SO'

# 4. Handle missing values (fill with median or drop rows with missing features)
for col in features:
    if df[col].isnull().any():
        median = df[col].median()
        df[col] = df[col].fillna(median)

# 5. Prepare feature matrix and target
X = df[features]
y = df[target]

# 6. Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Create LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
val_data   = lgb.Dataset(X_val,   label=y_val, reference=train_data)

# 8. Train LightGBM model
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
    num_boost_round = 1000,
    valid_sets      = [train_data, val_data],
    callbacks      = [lgb.early_stopping(50), lgb.log_evaluation(50)]
)

# 9. Evaluate performance
y_pred = model.predict(X_val, num_iteration=model.best_iteration)
rmse   = sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.3f}")

# 10. Output feature importances
importances = model.feature_importance()
feat_imp = pd.DataFrame({'feature': features, 'importance': importances})
feat_imp = feat_imp.sort_values('importance', ascending=False)
print("\nFeature Importances:")
print(feat_imp)

# 11. Save trained model and feature list
model_dir = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'lgb_so_model.pkl')
features_path = os.path.join(model_dir, 'lgb_so_model_features.txt')
joblib.dump(model, model_path)
with open(features_path, 'w') as f:
    for feat in features:
        f.write(f"{feat}\n")
print(f"Model saved to {model_path}")
print(f"Feature list saved to {features_path}")
