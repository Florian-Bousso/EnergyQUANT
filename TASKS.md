# TASKS — EnergyQuant

## En cours

## A faire

### Data
- [ ] Implémenter `fetcher.py` : connexion à l'API ENTSO-E (transparence.entsoe.eu)
- [ ] Implémenter `fetcher.py` : récupération des prix EEX / EPEX Spot
- [ ] Implémenter `cleaner.py` : pipeline de nettoyage des séries temporelles

### Analyse
- [ ] `spreads.py` : calcul spark spread, dark spread, clean spark/dark spread
- [ ] `spreads.py` : spreads géographiques (France, Allemagne, Nordpool...)
- [ ] `volatility.py` : volatilité rolling et GARCH(1,1)
- [ ] `volatility.py` : cônes de volatilité
- [ ] `risk.py` : VaR historique et paramétrique
- [ ] `risk.py` : CVaR / Expected Shortfall

### Forecasting
- [ ] `prophet_model.py` : entraînement de base sur prix spot
- [ ] `prophet_model.py` : ajout des régresseurs (température, renouvelables)
- [ ] `evaluator.py` : walk-forward validation
- [ ] `evaluator.py` : comparaison de modèles (Prophet vs baseline)

### Dashboard
- [ ] `app.py` : choix du framework (Dash ou Streamlit)
- [ ] `app.py` : graphiques de prix et volumes
- [ ] `app.py` : onglet spreads
- [ ] `app.py` : onglet prévisions

### CLI
- [ ] `cli.py` : commandes `fetch`, `analyze`, `forecast`, `serve`

### Tests
- [ ] Tests unitaires pour `cleaner.py`
- [ ] Tests unitaires pour `spreads.py`
- [ ] Tests d'intégration pipeline complet

## Terminé
