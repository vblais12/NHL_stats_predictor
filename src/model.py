from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit



def train_model(train_data, predictors, target, model, cv=5):
    TSS = TimeSeriesSplit(n_splits=cv)
    model.fit(train_data[predictors], train_data[target])
    return model




def evaluate_model(model, test_data, predictors, target):
    preds = model.predict(test_data[predictors])
    probs = model.predict_proba(test_data[predictors])[:, 1]
    report = classification_report(test_data[target], preds, output_dict=True)
    return preds, probs, report




def gridsearch(train_data, predictors, target, model, search_grid, cv=5):
    TSS = TimeSeriesSplit(n_splits=cv)
    GS = GridSearchCV(
        estimator=model,
        param_grid=search_grid,
        scoring='accuracy',
        cv=TSS,
        verbose=4
    )
    GS.fit(train_data[predictors], train_data[target])
    return GS.best_estimator_






