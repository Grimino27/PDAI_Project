import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

def logistic_model_selection(train_X, train_y):

    param_grid = [
        {
            'penalty': ['l2'],
            'C': np.logspace(-3, 2, 6),
            'class_weight': [None, 'balanced'],
            'solver': ['lbfgs']
        },
        {
            'penalty': ['l1'],
            'C': np.logspace(-3, 2, 6),
            'class_weight': [None, 'balanced'],
            'solver': ['saga']
        }
    ]

    grid = GridSearchCV(
        estimator=LogisticRegression(random_state=42, max_iter=2000),
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )

    grid.fit(train_X, train_y)

    return grid, grid.best_params_, grid.best_score_

def random_forest_model_selection(train_X, train_y):
    param_grid = []