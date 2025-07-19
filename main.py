import logging
from src.config import SEASONS, BASE_URL, HEADERS
from src.data_scraper import get_page
from src.data_organizer import parse_schedule_page, parse_box_score, parse_goalie_stats
from src.feature_eng import prepare_model_data
from src.model import tune_model, make_predictions
from Data.team_abbreviations import team_map
import pandas as pd
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    all_games, all_links, all_scores, goalie_data = [], [], [], []

    for season in SEASONS:
        url = f"{BASE_URL}/leagues/NHL_{season}_games.html"
        html = get_page(url, headers=HEADERS)
        games, links = parse_schedule_page(html)
        all_games.append(games)
        all_links.extend([f"{BASE_URL}{l}" for l in links])

    full_games = pd.concat(all_games)
    full_games['Home Team Win'] = (full_games['G.1'] > full_games['G']).astype(int)
    full_games.rename(columns={'G': 'GA', 'G.1': 'G'}, inplace=True)

    for link in all_links:
        html = get_page(link, headers=HEADERS)
        soup = BeautifulSoup(html, 'html.parser')
        teams = soup.find('div', class_='scorebox').find_all('strong')
        away = team_map.get(teams[0].text.strip())
        home = team_map.get(teams[1].text.strip())
        all_scores.append(parse_box_score(html, home, away))
        goalie_data.append(parse_goalie_stats(html))

    final_scores = pd.concat(all_scores).reset_index(drop=True)
    goalie_stats = pd.concat(goalie_data).reset_index(drop=True)
    combined = pd.concat([full_games.reset_index(drop=True), final_scores, goalie_stats], axis=1)

    # Transform to team-level rows (home + away)
    # Transform to team-level rows
    logger.info("Transforming to team-level rows...")
    combined.rename(columns={'Home Team Win' : 'Win/Loss'}, inplace=True)

    home = combined.copy()
    home.rename(columns={
        'Home': 'Team',
        'Visitor' : 'Opponent',
        'Win/Loss': 'Result'
    }, inplace=True)
    home['venue'] = 'Home'

    away = combined.copy()
    away.rename(columns={
        'Visitor': 'Team',
        'Home': 'Opponent',
        'Win/Loss': 'Result'
    }, inplace=True)
    away['venue'] = 'Away'

    # Flip home team/away stats to be from the away team's perspective
    away['G'], away['GA'] = away['GA'], away['G']
    away['S'], away['SA'] = away['SA'], away['S']
    away['S%'], away['SA%'] = away['SA%'], away['S%']
    away['SV%'], away['Opponent SV%'] = away['Opponent SV%'], away['SV%']
    away['PIM'], away['Opponent PIM'] = away['Opponent PIM'], away['PIM']
    away['Result'] = away['Result'].apply(lambda x: 1 - x)

    # Combine home and away rows
    combined_team_view = pd.concat([home, away], ignore_index=True)

    # Reorder columns for readability
    columns = combined_team_view.columns.tolist()
    columns.remove('venue')
    columns.insert(columns.index('Opponent') + 1, 'venue')
    combined_team_view = combined_team_view[columns]

    # Sort by team and date
    combined_team_view['Date'] = pd.to_datetime(combined_team_view['Date'])
    combined_team_view = combined_team_view.sort_values(by=['Team', 'Date']).reset_index(drop=True)

    combined_team_view.to_csv("games.csv", index=False)
    logger.info("Saved games.csv")

    # Load and run model
    data = pd.read_csv("games.csv")
    data, predictors = prepare_model_data(data, '2024-04-19')
    model = tune_model(data[data["Date"] < "2024-04-19"], predictors)
    results, precision = make_predictions(data, predictors, model)
    logger.info(f"Final model precision: {precision:.2f}, {(precision * 100):.2f}% accuracy")

    # Further precision results
    results = results.merge(combined_team_view[['Date', 'Team', 'Opponent', 'Result']], on=['Date', 'Team'], how='left')

    final = results.merge(results, left_on=['Date', 'Team'], right_on=['Date', 'Opponent'], suffixes=('_team', '_opponent'))  # few games will drop due to rolling windows

    # Getting results for when the prediction of home team and opponent match
    paired = final[(final['prediction_x'] == 1) & (final['prediction_y'] == 0)]['actual_x'].value_counts()

    # Filters for games when team A is predicted to win and team B is predicted to lose
    agree = paired[paired['prediction_team'] == 1]
    agree = agree[agree['prediction_opponent'] == 0]

    # Getting number of 1's
    correct = agree['Result_team'].value_counts().get(1, 0)

    # Compute accuracy
    total = len(paired)
    accuracy = correct / total

    logger.info(f"Accuracy when model predicts one team will win and opponent won't: {accuracy:.2%}, {(precision * 100):.2f}% accuracy")


if __name__ == "__main__":
    main()