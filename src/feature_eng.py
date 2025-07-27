import pandas as pd


# Function that computes rolling averages for each teams given which stats to compute rolling
# averages for, as well as the window. Returns
def rolling_averages(team, cols, new_cols, window=10):
    team = team.sort_values("Date")    # Getting team data organized chronologically
    rolling = team[cols].rolling(window, closed='left').mean()   # closed=left to ignore current row in sliding window
    team[new_cols] = rolling
    team = team.dropna(subset=new_cols) # dropping first rows because not enough data
    return team


# Function to compute elo features.
### Returns a pandas dataframe with team elo, opp elo, and elo differential
def compute_elo(data, k=30, base_elo=1500):
    teams = data['Team'].unique()
    elo_ratings = {team: base_elo for team in teams}

    elo_features = []

    # Loop through each game and update ratings
    for idx, row in data.iterrows():
        team = row['Team']
        opponent = row['Opponent']
        result = row['Result']

        # Optional: home-ice advantage
        team_elo = elo_ratings[team]
        opponent_elo = elo_ratings[opponent]

        # Store Elo features BEFORE the game
        elo_features.append({
            'team_elo': team_elo,
            'opponent_elo': opponent_elo,
            'elo_diff': team_elo - opponent_elo
        })

        # Calculate expected outcome
        expected_win = 1 / (1 + 10 ** ((opponent_elo - team_elo) / 400))

        change = k * (result - expected_win)
        elo_ratings[team] += change
        elo_ratings[opponent] -= change

    return pd.DataFrame(elo_features)




def get_opponent_diff(df, stat_cols):
    diff_col = []
    for stat in stat_cols:
        df[f'{stat}_diff'] = df[stat] - df[f'Opponent_{stat}']
        diff_col.append(f'{stat}_diff')
    return df, diff_col

