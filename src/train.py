"""Entraînement et comparaison des modèles de churn (Partie 6 du TP).

Usage (depuis la racine du projet) :
    python src/train.py
"""
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

df = pd.read_csv("data/raw/churn.csv")

X = df.drop("churn", axis=1)
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),

    "decision_tree": DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    ),

    "random_forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
}

results = {}

for name, model in models.items():

    print(f"\nEntraînement : {name}")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    results[name] = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions)
    }

    print(results[name])

# Sélection du Random Forest pour la suite du TP
model = models["random_forest"]

joblib.dump(model, "models/model.pkl")

print("\nModèle sauvegardé dans models/model.pkl")
