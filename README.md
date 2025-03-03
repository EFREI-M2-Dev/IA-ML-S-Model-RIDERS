# EXO 1 : MODEL ML S

Ce projet entraîne un modèle de **Machine Learning supervisé** pour prédire la survie des passagers du Titanic en utilisant un dataset Kaggle.

## 📌 Installation et Exécution

1. **Créer un environnement virtuel**

```bash
python3 -m venv venv
source venv/bin/activate
```

2. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

3. **Lancer le programme**

```bash
  python main.py
```

## 📊 Détails du modèle

-   **Modèle utilisé** : Random Forest Classifier
-   **Données** : Titanic Dataset (prétraité : suppression des colonnes inutiles, gestion des valeurs manquantes, normalisation)
-   **Évaluation** : Accuracy + Validation croisée

## 📜 Résultat attendu

Le script entraîne le modèle et affiche des métriques de performance comme l'**accuracy** et un **rapport de classification**.
