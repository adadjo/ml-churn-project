"""Tests E2E (bout en bout) avec Selenium - Partie 17 du TP.

Prérequis : Google Chrome installé et l'application démarrée
(uvicorn api.main:app  ou  docker compose up -d).

    pytest -m e2e tests/test_selenium.py

Variables d'environnement (toutes optionnelles) :
    BASE_URL           URL de l'application                      (défaut : http://localhost:8000)
    CHROME_BINARY      chemin du binaire Chrome / Chromium
    CHROMEDRIVER_PATH  chemin de chromedriver (sinon Selenium Manager le télécharge)
    SCREENSHOT_DIR     si défini, des captures d'écran y sont enregistrées
"""
import os
from pathlib import Path

import pytest
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000").rstrip("/")

# Ces tests sont exclus de la commande "pytest" par défaut (voir pytest.ini).
pytestmark = pytest.mark.e2e

CLIENT = {
    "age": 35,
    "anciennete": 5,
    "revenu": 450000,
    "nb_reclamations": 2,
    "montant_mensuel": 25000,
    "retards_paiement": 2,
}


@pytest.fixture
def driver():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1100,1000")
    if os.getenv("CHROME_BINARY"):
        options.binary_location = os.environ["CHROME_BINARY"]

    if os.getenv("CHROMEDRIVER_PATH"):
        service = Service(executable_path=os.environ["CHROMEDRIVER_PATH"])
    else:
        service = Service()  # Selenium Manager télécharge le bon driver

    browser = webdriver.Chrome(service=service, options=options)
    try:
        yield browser
    finally:
        browser.quit()


def _capture(driver, filename):
    folder = os.getenv("SCREENSHOT_DIR")
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        driver.save_screenshot(str(Path(folder) / filename))


def test_documentation_swagger(driver):
    """Test de l'énoncé, corrigé.

    L'énoncé vérifie "FastAPI" in driver.title, mais le titre de /docs est
    "<titre de l'application> - Swagger UI", soit ici "Churn Prediction API - Swagger UI".
    """
    driver.get(f"{BASE_URL}/docs")

    # Attente explicite plutôt qu'une assertion immédiate : un E2E ne doit pas
    # dépendre de l'instant exact où le navigateur expose le titre de la page.
    WebDriverWait(driver, 10).until(lambda d: "Swagger UI" in d.title)
    assert "Churn Prediction API" in driver.title


def test_parcours_utilisateur_prediction(driver):
    # 1. Ouverture de l'application
    driver.get(f"{BASE_URL}/ui/")
    WebDriverWait(driver, 10).until(lambda d: "churn" in d.title.lower())

    # 2. Saisie des données client
    for field, value in CLIENT.items():
        element = driver.find_element(By.ID, field)
        element.clear()
        element.send_keys(str(value))

    # 3. Clic sur "Prédire"  -> 4. Appel API (fetch /predict)
    driver.find_element(By.ID, "predict-btn").click()

    # 5. Affichage du résultat
    WebDriverWait(driver, 10).until(
        lambda d: d.find_element(By.ID, "prediction").text in ("CHURN", "NO_CHURN")
    )

    # 6. Vérification du résultat
    probability = float(driver.find_element(By.ID, "probability").get_attribute("data-value"))
    assert 0.0 <= probability <= 1.0
    assert driver.find_element(By.ID, "model-version").text == "1.0.0"

    _capture(driver, "e2e_interface_prediction.png")
