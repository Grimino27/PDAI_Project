import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

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
    
    param_grid = [
        {
            "n_estimators": [50, 100, 150, 200],
            "max_depth": [3, 5, 7, 10],  # più basso!,
            "min_samples_split": [10],
            "min_samples_leaf": [5, 10, 20],
            "class_weight": [None, 'balanced', 'balanced_subsample'],
        },        
    ]

    grid = GridSearchCV(
            estimator=RandomForestClassifier(random_state = 42),
            param_grid=param_grid
            cv=StratifiedKFold(n_splits=5),
            scoring=f1, #uso come scoring l'f1
            n_jobs=-1,
        )
    
    grid.fit(train_X, train_y)

    return grid, grid.best_params_, grid.best_score_


