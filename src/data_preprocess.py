import pandas as pd
from io import StringIO

# Returns the table as a dataframe
def parse_stats_table(html):
    table = pd.read_html(StringIO(html), attrs={'id' : 'teams'})[0]
    return table

def format_game_data(df):
    df = df.copy()
    df['Opponent'] = df.groupby('Game')['Team'].transform(lambda x: x[::-1].values)
    df['Date'] = df['Game'].str.extract(r'(^\d{4}-\d{2}-\d{2})')  # extracting data from the Game column
    df['Date'] = pd.to_datetime(df['Date'])
    df['Result'] = (df['GF'] > df['GA']).astype(int)
    df.drop(columns=['Game', 'Unnamed: 2'], inplace=True)
    return df


# Function to have Date and Result column visible in the first 3 columns
def format_for_vis(df):
    columns = list(df.columns)

    columns.remove('Date')
    columns.remove('Result')

    columns.insert(1, 'Date')
    columns.insert(2, 'Result')

    df = df[columns]
    return df

# Transform non-numeric/object columns to numeric
def to_numeric(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
