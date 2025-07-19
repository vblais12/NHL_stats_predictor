from sklearn.metrics import precision_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def make_predictions(data, predictors, model, cutoff='2024-04-19'):
    train = data[data['Date'] < cutoff]
    test = data[data['Date'] > cutoff]
    model.fit(train[predictors], train['Result'])
    preds = model.predict(test[predictors])
    combined = test.copy()
    combined['prediction'] = preds
    precision = precision_score(test['Result'], preds)
    return combined, precision

def tune_model(data, predictors):
    model = XGBClassifier(random_state=10)
    search_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 6],
        'learning_rate': [0.001, 0.01, 0.1],
        'reg_alpha': [0, 1, 5, 10],
        'reg_lambda': [0, 1, 5, 10]
    }
    GS = GridSearchCV(model, search_grid, scoring="accuracy", cv=5, verbose=2)
    GS.fit(data[predictors], data['Result'])
    return GS.best_estimator_