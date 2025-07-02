import pandas as pd
import numpy as np
from scipy.stats import nbinom


def american_to_decimal(american_odds):
    return (american_odds / 100 + 1) if american_odds > 0 else (100 / abs(american_odds) + 1)


def decimal_to_american(decimal_odds):
    return (decimal_odds - 1) * 100 if decimal_odds >= 2 else -100 / (decimal_odds - 1)


def calculate_mean_strikeouts(pitcher_data, player_name, opponent_team, opponent_adjusted):
    adj_row = opponent_adjusted[opponent_adjusted['team_name'] == opponent_team]
    if adj_row.empty:
        return np.nan
    opponent_adj = adj_row['opponent_adj'].values[0]

    player_row = pitcher_data[pitcher_data['PlayerName'] == player_name]
    if player_row.empty:
        return np.nan

    adjusted_k_pct = player_row['xK_pct'] * opponent_adj
    mean_so = (adjusted_k_pct * player_row['BF_IP'] * player_row['IP_G']).mean()
    return mean_so


def calculate_variance_so(pitcher_data, player_name):
    player_rows = pitcher_data[pitcher_data['PlayerName'] == player_name]
    return player_rows['SO'].var()


def calculate_B(mean_so, variance_so):
    return (variance_so / mean_so) - 1 if mean_so and variance_so else np.nan


def calculate_R(mean_so, variance_so):
    return (mean_so**2) / (variance_so - mean_so) if mean_so and variance_so and variance_so > mean_so else np.nan


def calculate_all_statistics(pitcher_data, opponent_adjusted, player_name, opponent_team):
    mean_so = calculate_mean_strikeouts(pitcher_data, player_name, opponent_team, opponent_adjusted)
    variance_so = calculate_variance_so(pitcher_data, player_name)
    B = calculate_B(mean_so, variance_so)
    R = calculate_R(mean_so, variance_so)
    return {'mean_SO': mean_so, 'variance_SO': variance_so, 'B': B, 'R': R}


def calculate_strikeout_probabilities(R, B):
    probabilities = [nbinom.pmf(k, R, 1 / (1 + B)) for k in range(16)]
    cumulative = np.cumsum(probabilities)
    return pd.DataFrame({
        'strikeout_count': range(16),
        'probability': probabilities,
        'cumulative_probability': cumulative
    })


def calculate_bet_values(prob_df, prop_line, price_over, price_under):
    under_idx = int(np.floor(prop_line - 0.5))
    under_percent = prob_df.loc[prob_df['strikeout_count'] == under_idx, 'cumulative_probability'].values[0]
    over_percent = 1 - under_percent
    price_over_decimal = american_to_decimal(price_over)
    price_under_decimal = american_to_decimal(price_under)
    value_over = over_percent * price_over_decimal - 1
    value_under = under_percent * price_under_decimal - 1
    return {
        'Under %': under_percent,
        'Over %': over_percent,
        'Value Over': value_over,
        'Value Under': value_under,
        'Price Over (Dec)': price_over_decimal,
        'Price Under (Dec)': price_under_decimal
    }
