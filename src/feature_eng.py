import pandas as pd


# Function that computes rolling averages for each teams given which stats to compute rolling
# averages for, as well as the window. Returns
def rolling_averages(team, cols, new_cols, window=10):
    team = team.sort_values("Date")    # Getting team data organized chronologically
    rolling = team[cols].rolling(window, closed='left').mean()   # closed=left to ignore current row in sliding window
    team[new_cols] = rolling
    team = team.dropna(subset=new_cols) # dropping first rows because not enough data
    return team


def compute_elo(data, k=30, base_elo=1500):
    teams = data['Team'].unique()
    elo_ratings = {team: base_elo for team in teams}

    elo_features = []

    # Loop through each game and update ratings
    for idx, row in data.iterrows():
        team = row['Team']
        opponent = row['Opponent']
        result = row['Result']  # 1 if win, 0 if loss

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
    for stat in stat_cols:
        df[f'{stat}_diff'] = df[stat] - df[f'Opponent_{stat}']
    return df


"""
def prepare_model_data(df, cutoff_date, team_column='Team'):
    cols = ['G', 'S', 'SV%', 'Result']
    new_cols = [f"{c}_rolling" for c in cols]
    df['Date'] = pd.to_datetime(df['Date'])

    result = df.groupby(team_column).apply(lambda x: rolling_averages(x, cols, new_cols))
    result = result.droplevel(0).reset_index(drop=True)
    result['venue'] = result['venue'].map({'Home': 1, 'Away': 0})
    result['opponent'] = result['Opponent'].astype('category').cat.codes
    predict = ['opponent', 'venue']

    return result, new_cols + predict

"""