import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=5),
            scoring='f1_macro',
            n_jobs=-1,
        )
    
    grid.fit(train_X, train_y)

    return grid, grid.best_params_, grid.best_score_

def prepare_train_test(X, y):

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,           # 20% per il test
        random_state=42,         # Per reproducibilità
        stratify=y               # Mantiene le proporzioni delle classi
    )

    # Scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return scaler, X_train,  y_train, X_test, y_test


def plot_confusion_matrix(cm, class_labels, title="Confusion Matrix"):

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=ax,
        xticklabels=class_labels,
        yticklabels=class_labels
    )

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt
import pandas as pd

def plot_logistic_coefficients(coefs, feature_names, classes, figsize_per_class=4):
    fig, axes = plt.subplots(
        len(classes), 1,
        figsize=(10, figsize_per_class * len(classes))
    )

    # se c'è solo una classe, matplotlib non restituisce lista
    if len(classes) == 1:
        axes = [axes]

    for i, cls in enumerate(classes):

        coef_df = pd.DataFrame({
            "feature": feature_names,
            "coef": coefs[i]
        }).sort_values("coef")

        colors = coef_df["coef"].apply(lambda x: "green" if x > 0 else "red")

        axes[i].barh(
            coef_df["feature"],
            coef_df["coef"],
            color=colors
        )

        axes[i].axvline(0, color="black", linewidth=1)
        axes[i].set_title(
            f"Logistic Regression Coefficients - Class {cls}",
            fontweight="bold"
        )
        axes[i].set_xlabel("Coefficient value")

    plt.tight_layout()
    plt.show()


