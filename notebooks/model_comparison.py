"""Comparaison des modèles par validation croisée (analyse complémentaire, hors énoncé).

Pourquoi ? Avec un seul découpage train/test de 400 lignes, l'écart-type de l'accuracy
est d'environ 2,4 points : les écarts entre modèles ne sont pas significatifs.
La validation croisée stratifiée à 5 plis donne une comparaison plus fiable.

On calcule aussi la performance de l'« oracle » : un classifieur qui connaîtrait la
formule génératrice des données (src/generate_data.py). Il donne le plafond théorique
atteignable sur ce dataset, dont le label est tiré au hasard (bruit irréductible).

Usage : python notebooks/model_comparison.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "data" / "raw" / "churn.csv")
X = df.drop("churn", axis=1)
y = df["churn"]

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ]),
    "decision_tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"]

rows = []
for name, model in models.items():
    scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
    row = {"modele": name}
    for metric in scoring:
        values = scores[f"test_{metric}"]
        row[metric] = f"{values.mean():.3f} ± {values.std():.3f}"
    rows.append(row)

print("Validation croisée stratifiée (5 plis) : moyenne ± écart-type\n")
print(pd.DataFrame(rows).to_string(index=False))

# Oracle : probabilité vraie utilisée pour générer le label
score = (
    0.04 * df["nb_reclamations"]
    + 0.08 * df["retards_paiement"]
    - 0.015 * df["anciennete"]
    + 0.00001 * df["montant_mensuel"]
)
p_true = 1 / (1 + np.exp(-score))
print("\nOracle (formule génératrice connue) - plafond théorique :")
print(f"  accuracy = {accuracy_score(y, (p_true > 0.5).astype(int)):.3f}")
print(f"  roc_auc  = {roc_auc_score(y, p_true):.3f}")

# Importance des variables du Random Forest retenu
rf = joblib.load(ROOT / "models" / "model.pkl")
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nImportance des variables (Random Forest sauvegardé) :")
print(importances.round(3).to_string())
