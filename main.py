# Needed libraries
import logging
from src.data_scraper import get_page, url_to_file
from src.data_preprocess import parse_stats_table, format_game_data, format_for_vis, to_numeric
from src.config import FEATURES, FEATURES_DIFF, ALL_FEATURES, TRAIN_TEST_SPLIT_DATE
from src.feature_eng import get_opponent_diff, rolling_averages, compute_elo
from src.model import gridsearch, evaluate_model
from src.search_grids import XGB_grid, RF_grid, LOGREG_grid
from sklearn.preprocessing import StandardScaler
import pandas as pd


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Links to page
url1 = 'https://www.naturalstattrick.com/games.php?fromseason=20212022&thruseason=20232024&stype=2&sit=all&loc=B&team=All&rate=n'
url2 = 'https://www.naturalstattrick.com/games.php?fromseason=20242025&thruseason=20242025&stype=2&sit=all&loc=B&team=All&rate=n'

# Getting html data
train_data = get_page(url1)
test_data = get_page(url2)

# Getting stats tables
train_df = parse_stats_table(train_data)
test_df = parse_stats_table(test_data)

# Concat tables and format properly
all_data = pd.concat([train_df, test_df], ignore_index=True)
all_data = format_game_data(all_data)

# Merge each team with its opponent
df_opp = all_data.drop(columns=['Opponent']).rename(columns={
    col : f"Opponent_{col}" for col in FEATURES
})
df_opp.rename(columns={'Team' : 'Opponent'}, inplace=True)
df_opp = df_opp.drop(columns=['Result'])

merged = all_data.merge(
    df_opp,
    left_on=['Date', 'Opponent'],
    right_on=['Date', 'Opponent'],
    how='inner',
    suffixes=('', '_y')
)

# Now computing differential stats
df, diff_cols = get_opponent_diff(merged, FEATURES_DIFF)
predictors = FEATURES + diff_cols

# Get rid of Arizona Coyotes (not a team anymore) + re-format to have Date + Result visible
df = df[df['Team'] != 'Arizona Coyotes']
df = df[df['Opponent'] != 'Arizona Coyotes']

df = format_for_vis(df)

# Transform object column to numeric dtypes
object_col = ['GF%', 'xGF%']
df = to_numeric(df, object_col)


# Compute rolling features
rolling_cols = [f'{col}_rolling' for col in predictors]
data = df.groupby('Team').apply(lambda x: rolling_averages(x, predictors, rolling_cols, window=3))
data = data.droplevel('Team')
data.index = range(data.shape[0])

print(data[rolling_cols].dtypes)

# Scale features
scaler = StandardScaler().fit(data[rolling_cols])
data[rolling_cols] = scaler.transform(data[rolling_cols])

# Add Elo ratings & features
elo_df = compute_elo(data, 30, 1500)
dataset = pd.concat([data.reset_index(drop=True), elo_df], axis=1)

# Split into train/test + setup predictors
dataset['Date'] = pd.to_datetime(dataset['Date'])
train = dataset[dataset['Date'] < TRAIN_TEST_SPLIT_DATE]
test = dataset[dataset['Date'] > TRAIN_TEST_SPLIT_DATE]

predictors = rolling_cols + ['elo_diff', 'team_elo', 'opponent_elo']
target = 'Result'

# Get # of pos/neg class for scale_pos_weight
class_counts = dataset['Result'].value_counts()

count_class_0 = class_counts[0]
count_class_1 = class_counts[1]

print(f"Losses (0): {count_class_0}")
print(f"Wins   (1): {count_class_1}")

scale = count_class_0 / count_class_1

# Machine Learning

# Machine Learning models
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression

# Models used
XGB = XGBClassifier(scale_pos_weight = scale, random_state=10)
RF = RandomForestClassifier(random_state=10)
LOGREG = BaggingClassifier(LogisticRegression(random_state=10, solver='liblinear', penalty='l2', max_iter=1000))

models = [XGB, RF, LOGREG]

# TimeSeriesSplit
TSS = TimeSeriesSplit(n_splits=10)


# GridSearch + training

model = gridsearch(train, predictors, target, XGB, XGB_grid, 5)

# Evaluate model

preds, probs, report = evaluate_model(model, test, predictors, target)

# Step 10: Output evaluation results
print("Evaluation Report:")
for k, v in report.items():
    print(f"{k}: {v}")


