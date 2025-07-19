import pandas as pd


# Function that computes rolling averages for each teams given which stats to compute rolling
# averages for, as well as the window. Returns
def rolling_averages(team, cols, new_cols, window=3):
    team = team.sort_values("Date")    # Getting team data organized chronologically
    rolling = team[cols].rolling(window, closed='left').mean()   # closed=left to ignore current row in sliding window
    team[new_cols] = rolling
    team = team.dropna(subset=new_cols) # dropping first rows because not enough data
    return team

def prepare_model_data(df, cutoff_date, team_column='Team'):
    cols = ['G', 'GA', 'S', 'S%', 'SV%', 'PIM', 'Result']
    new_cols = [f"{c}_rolling" for c in cols]
    df['Date'] = pd.to_datetime(df['Date'])

    result = df.groupby(team_column).apply(lambda x: rolling_averages(x, cols, new_cols))
    result = result.droplevel(0).reset_index(drop=True)
    result['venue'] = result['venue'].map({'Home': 1, 'Away': 0})
    result['opponent'] = result['Opponent'].astype('category').cat.codes
    predict = ['opponent', 'venue']

    return result, new_cols + predict
