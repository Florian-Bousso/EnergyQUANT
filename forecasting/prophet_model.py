"""
Modèle de forecasting des prix énergétiques basé sur Prophet (Meta).

Entraînement, configuration et inférence du modèle Prophet pour la
prévision des séries temporelles de prix de l'énergie. Gère les
saisonnalités propres aux marchés énergétiques (hebdomadaire, annuelle,
jours fériés), les régresseurs externes (température, vent, solaire)
et l'export des prévisions avec intervalles de confiance.
"""

import pandas as pd
from prophet import Prophet


def prepare_data(prices: pd.Series) -> pd.DataFrame:
    """
    Convertit une Series de prix en DataFrame au format attendu par Prophet.

    Prophet exige deux colonnes exactement : 'ds' (DatetimeIndex) et 'y'
    (valeurs numériques). Les données quart-horaires ou horaires sont
    agrégées en moyennes journalières pour réduire le bruit intra-journalier
    et accélérer l'entraînement. Les jours sans données sont supprimés.

    Paramètres
    ----------
    prices : pd.Series
        Série temporelle de prix avec index DatetimeIndex (EUR/MWh).
        Accepte toutes les fréquences (quart-horaire, horaire, journalière).

    Retourne
    --------
    pd.DataFrame
        DataFrame avec colonnes 'ds' (datetime) et 'y' (float),
        indexé de 0 à N-1, trié par date croissante.
    """
    daily = prices.resample("D").mean().dropna()
    df = daily.reset_index()
    df.columns = ["ds", "y"]
    df["ds"] = pd.to_datetime(df["ds"]).dt.tz_localize(None)
    return df.sort_values("ds").reset_index(drop=True)


def train_model(df: pd.DataFrame) -> Prophet:
    """
    Instancie et entraîne un modèle Prophet sur des prix journaliers.

    Configuration retenue pour les marchés électriques :
    - growth='flat' : pas de tendance linéaire. Les prix spot électricité
      sont mean-reverting — une tendance haussière extrapolée sur 7 jours
      produirait des prévisions économiquement aberrantes.
    - Saisonnalité hebdomadaire : activée — 90 jours = ~13 semaines,
      suffisant pour apprendre l'effet week-end (baisse de consommation).
    - Saisonnalité annuelle : désactivée — 90 jours ne couvrent pas un cycle
      complet, Prophet extrapolerait une tendance saisonnière fictive.
    - changepoint_prior_scale=0.05 : régularisation conservatrice pour
      éviter le surapprentissage sur des séries à pics extrêmes

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame au format Prophet avec colonnes 'ds' et 'y',
        tel que retourné par prepare_data().

    Retourne
    --------
    Prophet
        Modèle entraîné, prêt pour la génération de prévisions.
    """
    model = Prophet(
        growth="flat",
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05,
    )
    model.fit(df)
    return model


def forecast(model: Prophet, periods: int = 7) -> pd.DataFrame:
    """
    Génère une prévision sur un horizon futur à partir d'un modèle entraîné.

    Crée un DataFrame de dates futures en fréquence journalière, puis appelle
    la méthode de prédiction de Prophet. Le résultat inclut les estimations
    ponctuelles et les intervalles de confiance à 80 % (défaut Prophet).

    Paramètres
    ----------
    model : Prophet
        Modèle Prophet préalablement entraîné via train_model().
    periods : int
        Nombre de jours à prévoir au-delà de la dernière date d'entraînement.
        Défaut : 7 (une semaine).

    Retourne
    --------
    pd.DataFrame
        DataFrame Prophet avec au minimum les colonnes :
        - ds          : date de prévision
        - yhat        : prix prévu en EUR/MWh
        - yhat_lower  : borne basse de l'intervalle de confiance
        - yhat_upper  : borne haute de l'intervalle de confiance
    """
    future = model.make_future_dataframe(periods=periods, freq="D")
    return model.predict(future)


def forecast_summary(forecast_df: pd.DataFrame) -> dict:
    """
    Extrait les prévisions J+1 à J+7 depuis le DataFrame Prophet.

    Filtre les lignes futures (au-delà de la dernière date d'entraînement)
    et structure les résultats sous forme de dictionnaire indexé par date.

    Paramètres
    ----------
    forecast_df : pd.DataFrame
        DataFrame retourné par forecast(), contenant les colonnes
        ds, yhat, yhat_lower, yhat_upper.

    Retourne
    --------
    dict
        Dictionnaire dont les clés sont des dates (str au format YYYY-MM-DD)
        et les valeurs sont des dicts avec :
        - price      (float) : prix prévu en EUR/MWh
        - lower      (float) : borne basse de l'intervalle de confiance
        - upper      (float) : borne haute de l'intervalle de confiance
    """
    # Les lignes futures sont les 7 dernières (make_future_dataframe ajoute
    # exactement `periods` lignes au-delà de l'historique)
    future_rows = forecast_df.tail(7)[["ds", "yhat", "yhat_lower", "yhat_upper"]]

    return {
        row["ds"].strftime("%Y-%m-%d"): {
            "price": round(row["yhat"], 2),
            "lower": round(row["yhat_lower"], 2),
            "upper": round(row["yhat_upper"], 2),
        }
        for _, row in future_rows.iterrows()
    }


if __name__ == "__main__":
    import sys
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.fetcher import fetch_day_ahead_prices

    end = datetime.now(timezone.utc).replace(tzinfo=None)
    start = end - timedelta(days=90)

    print(f"Récupération des prix day-ahead France du {start.date()} au {end.date()} (90 jours)...")
    prices = fetch_day_ahead_prices("FR", start, end)["price_eur_mwh"]
    print(f"{len(prices)} points chargés → agrégation en jours...")

    df = prepare_data(prices)
    print(f"{len(df)} jours disponibles pour l'entraînement.\n")

    print("Entraînement du modèle Prophet...")
    model = train_model(df)

    print("Génération des prévisions J+1 à J+7...\n")
    forecast_df = forecast(model, periods=7)
    summary = forecast_summary(forecast_df)

    print("=" * 52)
    print("Prévisions des prix day-ahead — France")
    print("=" * 52)
    print(f"  {'Date':<12}  {'Prévu':>10}  {'Bas':>10}  {'Haut':>10}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")
    for date, vals in summary.items():
        print(
            f"  {date:<12}  {vals['price']:>9.2f}€  "
            f"{vals['lower']:>9.2f}€  {vals['upper']:>9.2f}€"
        )
    print()
    prices_list = [v["price"] for v in summary.values()]
    print(f"  Moyenne prévue : {sum(prices_list) / len(prices_list):.2f} EUR/MWh")
