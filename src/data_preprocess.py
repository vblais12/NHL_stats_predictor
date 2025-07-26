import pandas as pd
from io import StringIO

def parse_stats_table(html):
    table = pd.read_html(StringIO(html), attrs={'id' : 'teams'})[0]
    return table

def format_opp_data(df):
    df = df.copy()
    df['Opponent'] = df.groupby('Game')['Team'].transform(lambda x: x[::-1].values)
    df['Date'] = df['Game'].str.extract(r'(^\d{4}-\d{2}-\d{2})')  # extracting data from the Game column
    df['Date'] = pd.to_datetime(df['Date'])
    df['Result'] = (df['GF'] > df['GA']).astype(int)
    df.drop(columns=['Game', 'Unnamed: 2'], inplace=True)
    return df

