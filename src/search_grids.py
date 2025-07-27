# Defining search space for GridSearchCV

# XGBoost
XGB_grid = {
    'n_estimators': [50, 75, 100, 150],
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.03, 0.05],
    'reg_alpha': [1, 5, 10],
    'reg_lambda': [1, 5, 10]

}

# Random Forest
RF_grid = {
    'n_estimators' : [50, 100, 200, 500],
    'max_depth' : [3, 6, 9],
    'min_samples_split': [3, 5, 10]
}

# Logistic Regression
LOGREG_grid = {
    # Logistic Regression hyperparameters (base_estimator__)
    'estimator__C' : [0.5, 0.8, 1.0],
    'n_estimators': [3, 5, 10, 50, 100],
}