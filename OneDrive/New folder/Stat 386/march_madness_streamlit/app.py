# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
@st.cache_data
def load_data():
    advancement_df = pd.read_csv("advancement_probabilities.csv")
    winners_df = pd.read_csv("bracket_winners.csv")
    return advancement_df, winners_df

advancement_df, winners_df = load_data()

# Rename columns for easier use
advancement_df.rename(columns={"Unnamed: 0": "Team"}, inplace=True)
winners_df.rename(columns={"Unnamed: 0": "Index", "Game": "Game", "Most Likely Winner": "Winner"}, inplace=True)

# App Layout
st.title("🏀 March Madness 2025: Bracket Model Explorer")
st.markdown("""
Welcome to the **March Madness Prediction App**!  
Explore simulated team advancement probabilities and first-round winners from my statistical model.

*Created by Aaron Clemons*
""")

# Section 1 – First Round Predictions
st.header("🎯 First Round: Predicted Winners")

st.write("Below are the model's predicted winners for the first-round matchups:")

st.dataframe(winners_df[["Game", "Winner"]])

# Section 2 – Advancement Probabilities
st.header("📈 Team Advancement Probabilities")

selected_team = st.selectbox("Select a team to view their round-by-round advancement probabilities:", advancement_df["Team"].unique())

team_data = advancement_df[advancement_df["Team"] == selected_team].drop(columns="Team").T
team_data.columns = [selected_team]
team_data.index.name = "Round"

st.bar_chart(team_data)

# Section 3 – Heatmap
st.header("🔥 Full Tournament Heatmap")

st.markdown("This heatmap shows the probability of each team advancing to each round.")

heatmap_data = advancement_df.set_index("Team")
fig, ax = plt.subplots(figsize=(12, 8))
sns.heatmap(heatmap_data, cmap="YlGnBu", annot=True, fmt=".2f", linewidths=.5, ax=ax)
ax.set_title("Team Advancement Probability Heatmap")
st.pyplot(fig)

# Section 4 – Team Comparison
st.header("📊 Compare Two Teams")
team1 = st.selectbox("Team 1", advancement_df["Team"].unique(), index=0)
team2 = st.selectbox("Team 2", advancement_df["Team"].unique(), index=1)

team1_data = advancement_df[advancement_df["Team"] == team1].drop(columns=["Team"]).T
team2_data = advancement_df[advancement_df["Team"] == team2].drop(columns=["Team"]).T

compare_df = pd.concat([team1_data, team2_data], axis=1)
compare_df.columns = [team1, team2]
st.bar_chart(compare_df)

# Section 5 – Round selector
st.header("🗺️ Advancement by Round")

selected_round = st.selectbox("Choose a Round", advancement_df.columns[1:])
round_data = advancement_df[["Team", selected_round]].set_index("Team")

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(round_data, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=.5, ax=ax)
ax.set_title(f"Probability to Advance to {selected_round}")
st.pyplot(fig)

# Team selection for round advancement
st.header("🔍 Search & Download Team Stats")
search_team = st.selectbox("Search a Team", advancement_df["Team"].unique())
search_result = advancement_df[advancement_df["Team"] == search_team]

st.write(f"### {search_team} Round Probabilities")
st.dataframe(search_result.T)

csv = search_result.T.to_csv().encode('utf-8')
st.download_button("📥 Download This Team's Stats", data=csv, file_name=f'{search_team}_advancement.csv', mime='text/csv')
