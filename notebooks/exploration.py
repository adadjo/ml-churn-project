"""Exploration des données de churn (Partie 5 du TP).

Usage :
    cd notebooks && python exploration.py        (comme dans l'énoncé)
    python notebooks/exploration.py              (depuis la racine, aussi valable)

En environnement sans écran (CI, SSH, WSL) : MPLBACKEND=Agg python notebooks/exploration.py
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Les chemins sont calculés à partir de l'emplacement du script : ils fonctionnent
# quel que soit le dossier depuis lequel on lance la commande.
ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(ROOT / "data" / "raw" / "churn.csv")

print(df.head())

print("\nDimensions :")
print(df.shape)

print("\nTypes :")
print(df.dtypes)

print("\nValeurs manquantes :")
print(df.isnull().sum())

print("\nStatistiques :")
print(df.describe())

print("\nRépartition du churn :")
print(df["churn"].value_counts())

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="churn")
plt.title("Répartition des clients selon le churn")
plt.savefig(ROOT / "data" / "churn_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Corrélations")
plt.savefig(ROOT / "data" / "correlation.png", dpi=150, bbox_inches="tight")
plt.show()
