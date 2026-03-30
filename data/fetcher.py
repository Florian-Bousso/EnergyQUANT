"""
Récupération des données de marché énergétique européen.

Ce module est responsable de l'acquisition des données brutes depuis
les sources externes : APIs de marchés (ENTSO-E, EEX, EPEX), flux RSS,
fichiers CSV ou bases de données. Il expose des fonctions permettant
de télécharger et stocker localement les prix spot, futures, volumes
et données fondamentales (production, consommation, météo).
"""

import os
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from entsoe import EntsoePandasClient
from entsoe.exceptions import NoMatchingDataError

load_dotenv()


def fetch_day_ahead_prices(country_code: str, start: datetime, end: datetime) -> pd.DataFrame:
    """
    Récupère les prix spot day-ahead horaires depuis l'API ENTSO-E.

    Les prix sont exprimés en EUR/MWh. Les données proviennent du marché
    day-ahead de la zone de prix correspondant au code pays fourni.

    Paramètres
    ----------
    country_code : str
        Code pays au format ENTSO-E (ex. "FR", "DE_LU", "ES", "BE").
    start : datetime
        Date et heure de début de la période (timezone-aware ou naive en UTC).
    end : datetime
        Date et heure de fin de la période (timezone-aware ou naive en UTC).

    Retourne
    --------
    pd.DataFrame
        DataFrame indexé par timestamp (UTC) avec une colonne "price_eur_mwh".

    Lève
    ----
    EnvironmentError
        Si la variable ENTSOE_API_KEY est absente du fichier .env.
    ValueError
        Si aucune donnée n'est disponible pour la période ou la zone demandée.
    """
    api_key = os.getenv("ENTSOE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "Clé API ENTSO-E introuvable. "
            "Ajoutez ENTSOE_API_KEY=<votre_clé> dans le fichier .env"
        )

    client = EntsoePandasClient(api_key=api_key)

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")

    try:
        series = client.query_day_ahead_prices(country_code, start=start_ts, end=end_ts)
    except NoMatchingDataError:
        raise ValueError(
            f"Aucune donnée day-ahead disponible pour la zone '{country_code}' "
            f"entre {start_ts.date()} et {end_ts.date()}."
        )

    df = series.rename("price_eur_mwh").to_frame()
    df.index.name = "timestamp"
    return df


if __name__ == "__main__":
    end = datetime.utcnow()
    start = end - timedelta(days=7)

    print(f"Récupération des prix day-ahead France du {start.date()} au {end.date()}...")
    df = fetch_day_ahead_prices("FR", start, end)

    print(f"\nNombre de points : {len(df)}")
    print(f"Période : {df.index[0]} → {df.index[-1]}")
    print(f"\nPrix min : {df['price_eur_mwh'].min():.2f} EUR/MWh")
    print(f"Prix max : {df['price_eur_mwh'].max():.2f} EUR/MWh")
    print(f"Prix moyen : {df['price_eur_mwh'].mean():.2f} EUR/MWh")
    print(f"\nAperçu :\n{df.head(10)}")
