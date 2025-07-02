import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import sys, os

# Make project root importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.model import (
    calculate_all_statistics,
    calculate_strikeout_probabilities,
    calculate_bet_values
)

# Load real pitching data from Excel
data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'MLB_2025_Pitching_Data.xlsx')
pitcher_data = pd.read_excel(data_path)

# Rename columns to match model expectations
pitcher_data = pitcher_data.rename(columns={
    "Name": "PlayerName",
    "Name-additional": "bbref_id",
    "Team": "Tm",
    "Str %": "Str_pct",
    "CS %": "L_str_pct",
    "SwStr %": "S_str_pct"
})

# Compute derived columns if missing: IP_G and BF_IP
if 'IP_G' not in pitcher_data.columns:
    if all(col in pitcher_data.columns for col in ['IP', 'G']):
        pitcher_data['IP_G'] = pitcher_data['IP'] / pitcher_data['G']
    else:
        st.error("Missing 'IP_G' or ['IP','G'] columns in your data. Please include either 'IP_G' or both 'IP' and 'G'.")

if 'BF_IP' not in pitcher_data.columns:
    if all(col in pitcher_data.columns for col in ['BF', 'IP']):
        pitcher_data['BF_IP'] = pitcher_data['BF'] / pitcher_data['IP']
    else:
        st.error("Missing 'BF_IP' or ['BF','IP'] columns. Please include either 'BF_IP' or both 'BF' and 'IP'.")

# Compute xK_pct if not provided
if 'xK_pct' not in pitcher_data.columns:
    pitcher_data["xK_pct"] = -0.8432 + (
        pitcher_data["Str_pct"] * 0.2916 +
        pitcher_data["L_str_pct"] * 1.2689 +
        pitcher_data["S_str_pct"] * 1.5334
    )

# Load or mock opponent adjustment data
# TODO: replace with real opponent adjustments when available
opponent_data = pd.DataFrame({
    'team_name': ['TEX', 'NYY', 'BOS'],
    'opponent_adj': [1.00, 0.98, 1.02]
})

st.title("MLB Pitcher Strikeout Prediction Model (2025 Data)")

# Sidebar inputs
player_name = st.sidebar.selectbox("Select Player Name", pitcher_data["PlayerName"].unique())
opponent_team = st.sidebar.selectbox("Select Opponent Team", opponent_data["team_name"].unique())
prop_line = st.sidebar.number_input("Prop Line (Strikeouts)", min_value=0.0, max_value=15.0, value=6.5, step=0.1)
price_over = st.sidebar.number_input("Price Over (American Odds)", value=120, step=1)
price_under = st.sidebar.number_input("Price Under (American Odds)", value=-155, step=1)

# Calculate statistics
stats = calculate_all_statistics(pitcher_data, opponent_data, player_name, opponent_team)
mean_so = stats["mean_SO"]
B = stats["B"]
R = stats["R"]

# If any core stat is NaN, inform the user
if np.isnan(mean_so) or np.isnan(B) or np.isnan(R):
    st.error("Unable to calculate stats. Please check your data for missing values.")
else:
    prob_df = calculate_strikeout_probabilities(R, B)
    values = calculate_bet_values(prob_df, prop_line, price_over, price_under)

    # Display results
    st.subheader("Expected Strikeouts")
    st.metric(label="Expected K's", value=f"{mean_so:.1f}")

    # Betting info table
    betting_table = pd.DataFrame({
        "Prop Line": [prop_line],
        "Under %": [values["Under %"]],
        "Over %": [values["Over %"]],
        "Price Over (Dec)": [f"{values['Price Over (Dec)']:.2f}"],
        "Price Under (Dec)": [f"{values['Price Under (Dec)']:.2f}"],
        "Value Over": [values["Value Over"]],
        "Value Under": [values["Value Under"]],
    })
    st.subheader("Betting Information")
    st.table(betting_table)

    # Strikeout distribution histogram
    st.subheader("Strikeout Probability Distribution")
    fig, ax = plt.subplots()
    ax.bar(prob_df["strikeout_count"], prob_df["probability"])
    ax.set_xlabel("Strikeouts")
    ax.set_ylabel("Probability")
    ax.set_title("Strikeout Distribution")
    st.pyplot(fig)
