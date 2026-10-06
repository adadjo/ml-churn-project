"""Entraînement et comparaison des modèles de churn (Partie 6 du TP).

Usage (depuis la racine du projet) :
    python src/train.py

Sorties :
    models/model.pkl      modèle sérialisé (Random Forest, utilisé par l'API)
    models/metadata.json  métadonnées de traçabilité : version, date, empreinte du
                          dataset, versions des bibliothèques, métriques de chaque modèle
"""
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import joblib
import sklearn

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
    f1_score,
    roc_auc_score
)

MODEL_VERSION = "1.0.0"
DATASET_PATH = Path("data/raw/churn.csv")

df = pd.read_csv(DATASET_PATH)

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
    probabilities = model.predict_proba(X_test)[:, 1]

    results[name] = {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities)
    }

    print(results[name])

# Sélection du Random Forest pour la suite du TP
selected = "random_forest"
model = models[selected]

joblib.dump(model, "models/model.pkl")

# Traçabilité : de quoi retrouver comment ce modèle a été produit
metadata = {
    "model_name": "Random Forest",
    "model_type": type(model).__name__,
    "model_version": MODEL_VERSION,
    "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "features": list(X.columns),
    "target": "churn",
    "train_size": int(len(X_train)),
    "test_size": int(len(X_test)),
    "dataset_sha256": hashlib.sha256(DATASET_PATH.read_bytes()).hexdigest(),
    "sklearn_version": sklearn.__version__,
    "python_version": platform.python_version(),
    "selected_model": selected,
    "metrics": results[selected],
    "all_models_metrics": results
}

with open("models/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

print("\nModèle sauvegardé dans models/model.pkl")
print("Métadonnées sauvegardées dans models/metadata.json")
