# EnergyQuant

Outil d'analyse et de forecasting des marchés de l'énergie européens.

## Fonctionnalités

- **Récupération de données** : prix spot et futures (électricité, gaz, charbon, CO2)
- **Analyse des spreads** : spark spread, dark spread, clean spreads, spreads géographiques
- **Volatilité** : volatilité historique, GARCH, cônes de volatilité
- **Risque** : VaR, CVaR, drawdown, stress tests
- **Forecasting** : modèle Prophet avec régresseurs externes
- **Dashboard** : visualisation interactive des marchés et des prévisions

## Structure

```
EnergyQuant/
├── data/
│   ├── fetcher.py       # Acquisition des données de marché
│   └── cleaner.py       # Nettoyage et normalisation
├── analysis/
│   ├── spreads.py       # Calcul des spreads
│   ├── volatility.py    # Analyse de volatilité
│   └── risk.py          # Métriques de risque
├── forecasting/
│   ├── prophet_model.py # Modèle Prophet
│   └── evaluator.py     # Évaluation des prévisions
├── dashboard/
│   └── app.py           # Dashboard interactif
├── tests/
├── cli.py               # Interface CLI
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python cli.py --help
```
