"""Génération du dataset pédagogique de churn télécom (Partie 4 du TP).

Usage (depuis la racine du projet) :
    python src/generate_data.py
"""
import os

import pandas as pd
import numpy as np

np.random.seed(42)

n = 2000

age = np.random.randint(18, 70, n)
anciennete = np.random.randint(1, 120, n)
revenu = np.random.randint(150000, 1000000, n)
nb_reclamations = np.random.poisson(2, n)
montant_mensuel = np.random.randint(5000, 75000, n)
retards_paiement = np.random.poisson(1, n)

score = (
    0.04 * nb_reclamations
    + 0.08 * retards_paiement
    - 0.015 * anciennete
    + 0.00001 * montant_mensuel
)

probabilite = 1 / (1 + np.exp(-score))

churn = (np.random.random(n) < probabilite).astype(int)

df = pd.DataFrame({
    "age": age,
    "anciennete": anciennete,
    "revenu": revenu,
    "nb_reclamations": nb_reclamations,
    "montant_mensuel": montant_mensuel,
    "retards_paiement": retards_paiement,
    "churn": churn
})

os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/churn.csv", index=False)

print(df.head())
print()
print(df["churn"].value_counts())
