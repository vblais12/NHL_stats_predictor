import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup

# This function parses the schedule page, extracts the schedule & results table, as well as the links to
# the box score for each game. Returns the table with all boxscores links
def parse_schedule_page(html):
    soup = BeautifulSoup(html, 'html.parser')
    html = StringIO(html)
    table = pd.read_html(html, match='NHL Regular Season Schedule')[0]
    box_scores_table = soup.find(id='games')  # Get table for box scores links
    box_scores_links = [l.get("href") for l in box_scores_table.find_all("a")]  # Get all links
    box_scores_links = [l for l in box_scores_links if l and "boxscores/" in l] # Only keep boxscores links
    return table, box_scores_links


# Function to extract box score stats from box scores table. Returns a pandas dataframe containing stats for home team
# and 'opponent' / visitors team
def parse_box_score(html, home_abbrev, visitors_abbrev):
    html = StringIO(html)

    v_table = pd.read_html(html, attrs={'id': f'{visitors_abbrev}_skaters'}, header=1)[0]
    h_table = pd.read_html(html, attrs={'id': f'{home_abbrev}_skaters'}, header=1)[0]

    # Find row where the player name/index contains "TOTAL" and get the last one
    v_stats_table = v_table[v_table.iloc[:, 1].str.contains('TOTAL', na=False)].iloc[[-1]]
    h_stats_table = h_table[h_table.iloc[:, 1].str.contains('TOTAL', na=False)].iloc[[-1]]

    # Only keeping relevant columns
    v_stats_table = v_stats_table[['PIM', 'S', 'S%']]
    h_stats_table = h_stats_table[['PIM', 'S', 'S%']]

    # Renaming visitor columns for processing later
    v_stats_table.rename(columns={'PIM' : 'Opponent PIM', 'S' : 'SA', 'S%' : 'SA%'}, inplace=True)

    return pd.concat([v_stats_table.reset_index(drop=True), h_stats_table.reset_index(drop=True)], axis=1)



# This function extracts goalie stats from the box score page. Returns a pandas dataframe
# containing goalie stats for both the home and visitors team
def parse_goalie_stats(html):
    html = StringIO(html)

    goalie_stats = pd.read_html(html, match='Goalies Table', header=1)  # List of tables (df)
    v_goalie_stats = goalie_stats[0]  # First table is visitors
    h_goalie_stats = goalie_stats[1]  # Second table is home

    # Getting team's primary goalie
    v_goalie_stats = v_goalie_stats[v_goalie_stats['Rk'] == 1]
    h_goalie_stats = h_goalie_stats[h_goalie_stats['Rk'] == 1]

    # Only keeping relevant columns
    v_goalie_stats = v_goalie_stats[['SV%']]
    h_goalie_stats = h_goalie_stats[['SV%']]

    v_goalie_stats.rename(columns={'SV%': 'Opponent SV%'}, inplace=True)

    return pd.concat([v_goalie_stats.reset_index(drop=True), h_goalie_stats.reset_index(drop=True)], axis=1)











