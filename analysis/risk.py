"""
Mesure et gestion du risque sur les marchés de l'énergie.

Calcul des métriques de risque : VaR (Value at Risk), CVaR/Expected Shortfall,
drawdown maximal, corrélations inter-commodités, stress tests et analyse
de scénarios. Fournit des outils d'aide à la décision pour la gestion
des positions et la couverture de risque de prix.

Note méthodologique
-------------------
Les marchés électriques spot présentent des prix négatifs, des pics extrêmes
et une forte mean-reversion qui rendent les rendements logarithmiques inadaptés.
Ce module travaille donc en variations absolues de prix (EUR/MWh), ce qui :
- Tolère nativement les prix négatifs sans plancher artificiel
- Exprime la VaR et la CVaR directement en EUR/MWh, unité économique naturelle
- Donne une volatilité annualisée en EUR/MWh comparable entre marchés
"""

import numpy as np
import pandas as pd
from scipy import stats


# Nombre de périodes par an selon la fréquence de l'index pandas
_PERIODS_PER_YEAR = {
    "15min": 35_040,
    "15T":   35_040,
    "h":      8_760,
    "H":      8_760,
    "D":        365,
}


def _detect_periods_per_year(prices: pd.Series) -> int:
    """
    Détecte automatiquement la fréquence des données et retourne le nombre
    de périodes par an correspondant.

    Tente d'inférer la fréquence depuis l'index DatetimeIndex. Si la détection
    échoue (index non-datetime, fréquence irrégulière), calcule la médiane des
    intervalles entre observations.

    Paramètres
    ----------
    prices : pd.Series
        Série temporelle avec index DatetimeIndex.

    Retourne
    --------
    int
        Nombre de périodes par an (ex. 8760 pour horaire, 35040 pour 15 min).
    """
    if isinstance(prices.index, pd.DatetimeIndex) and len(prices) > 1:
        freq = pd.infer_freq(prices.index)
        if freq and freq in _PERIODS_PER_YEAR:
            return _PERIODS_PER_YEAR[freq]
        # Fallback : médiane des deltas en minutes
        median_minutes = prices.index.to_series().diff().dropna().median().total_seconds() / 60
        if median_minutes <= 15:
            return 35_040
        if median_minutes <= 60:
            return 8_760
    return 8_760  # défaut horaire


def compute_price_changes(prices: pd.Series) -> pd.Series:
    """
    Calcule les variations absolues de prix en EUR/MWh.

    Sur les marchés électriques spot, les variations absolues sont préférées
    aux rendements logarithmiques : les prix peuvent être négatifs, ils
    présentent des pics extrêmes et une forte mean-reversion qui rendent
    l'hypothèse log-normale inadaptée. La variation absolue reste définie
    et économiquement interprétable dans tous les cas.

    Formule : Δp_t = P_t - P_{t-1}

    Paramètres
    ----------
    prices : pd.Series
        Série temporelle de prix (EUR/MWh). Accepte les valeurs négatives.

    Retourne
    --------
    pd.Series
        Série des variations absolues en EUR/MWh, sans NaN.
    """
    return prices.diff().dropna()


def compute_volatility(prices: pd.Series, window: int = 24) -> pd.Series:
    """
    Calcule la volatilité historique glissante des prix en EUR/MWh.

    La volatilité est l'écart-type glissant des variations absolues de prix,
    annualisé par la racine carrée du nombre de périodes par an. Elle est
    exprimée en EUR/MWh et représente l'amplitude typique des mouvements
    de prix sur une année.

    La fréquence est détectée automatiquement depuis l'index :
    - données horaires  : window=24  → 1 jour, annualisation × √8760
    - données 15 min    : window=96  → 1 jour, annualisation × √35040

    Paramètres
    ----------
    prices : pd.Series
        Série temporelle de prix avec index DatetimeIndex (EUR/MWh).
    window : int
        Taille de la fenêtre glissante en nombre de périodes.
        Défaut : 24 (adapté aux données horaires).

    Retourne
    --------
    pd.Series
        Volatilité glissante annualisée en EUR/MWh, indexée comme prices.
        Les premières valeurs (window - 1) sont NaN.
    """
    changes = compute_price_changes(prices)
    periods_per_year = _detect_periods_per_year(prices)
    return changes.rolling(window=window).std() * np.sqrt(periods_per_year)


def compute_var(
    prices: pd.Series,
    confidence: float = 0.95,
    window: int = 168,
) -> pd.Series:
    """
    Calcule la Value at Risk (VaR) historique glissante en EUR/MWh.

    La VaR historique est le quantile des pertes de prix observées sur la
    fenêtre glissante. Une VaR à 95 % de 30 EUR/MWh signifie que la baisse
    de prix sur une période ne dépassera 30 EUR/MWh que 5 % du temps,
    sur la base des données historiques.

    Paramètres
    ----------
    prices : pd.Series
        Série temporelle de prix (EUR/MWh).
    confidence : float
        Niveau de confiance (défaut : 0.95).
    window : int
        Taille de la fenêtre glissante en nombre de périodes.
        Défaut : 168 (1 semaine en données horaires).

    Retourne
    --------
    pd.Series
        VaR glissante en EUR/MWh. Valeur positive = perte potentielle.
    """
    changes = compute_price_changes(prices)
    var = changes.rolling(window=window).quantile(1 - confidence)
    return -var


def compute_cvar(
    prices: pd.Series,
    confidence: float = 0.95,
    window: int = 168,
) -> pd.Series:
    """
    Calcule la CVaR (Conditional Value at Risk / Expected Shortfall) glissante
    en EUR/MWh.

    La CVaR est la moyenne des pertes qui dépassent la VaR. Elle complète
    la VaR en capturant l'amplitude des baisses de prix extrêmes, pas seulement
    leur seuil. C'est la mesure de risque cohérente de référence en risk management.

    Paramètres
    ----------
    prices : pd.Series
        Série temporelle de prix (EUR/MWh).
    confidence : float
        Niveau de confiance (défaut : 0.95).
    window : int
        Taille de la fenêtre glissante en nombre de périodes.
        Défaut : 168 (1 semaine en données horaires).

    Retourne
    --------
    pd.Series
        CVaR glissante en EUR/MWh. Valeur positive = perte moyenne
        dans le pire (1 - confidence) % des cas.
    """
    changes = compute_price_changes(prices)

    def _cvar_window(w: np.ndarray) -> float:
        threshold = np.quantile(w, 1 - confidence)
        tail = w[w <= threshold]
        return -tail.mean() if len(tail) > 0 else np.nan

    return changes.rolling(window=window).apply(_cvar_window, raw=True)


def risk_summary(prices: pd.Series) -> dict:
    """
    Calcule un résumé complet des métriques de risque d'une série de prix.

    Toutes les métriques sont exprimées dans des unités économiques directement
    interprétables pour un marché électrique européen :

    - Volatilité (écart-type des prix journaliers) en EUR/MWh et en % du
      prix moyen (coefficient de variation). Valeurs typiques : 10–50 EUR/MWh,
      soit 20–80 % du prix moyen.
    - VaR 95% en EUR/MWh : baisse de prix maximale à 95 % de confiance
      sur une période. Valeurs typiques : 20–80 EUR/MWh.
    - CVaR 95% en EUR/MWh : perte moyenne au-delà de la VaR.
      Valeurs typiques : 40–150 EUR/MWh.

    Paramètres
    ----------
    prices : pd.Series
        Série temporelle de prix avec index DatetimeIndex (EUR/MWh).

    Retourne
    --------
    dict
        Dictionnaire contenant :
        - mean_price (float)              : prix moyen journalier en EUR/MWh
        - volatility_eur_mwh (float)      : écart-type des prix journaliers en EUR/MWh
        - volatility_pct (float)          : coefficient de variation en %
        - var_95_eur_mwh (float)          : VaR 95% en EUR/MWh
        - cvar_95_eur_mwh (float)         : CVaR 95% en EUR/MWh
        - skewness (float)                : asymétrie des variations de prix
        - kurtosis (float)                : aplatissement (excès) des variations
        - periods_per_year (int)          : fréquence détectée, pour vérification
    """
    # Volatilité : écart-type des prix journaliers moyens, sans annualisation.
    # Le resample sur 'D' élimine le bruit intra-journalier des données
    # quart-horaires tout en préservant la dispersion réelle entre jours.
    daily_prices = prices.resample("D").mean().dropna()
    mean_price = daily_prices.mean()
    n_daily = len(daily_prices)

    vol_eur = daily_prices.std() if n_daily > 1 else None
    vol_pct = (vol_eur / mean_price * 100) if (vol_eur is not None and mean_price != 0) else None

    # VaR / CVaR : calculées sur les données brutes (quart-horaires ou horaires)
    # pour capturer le risque instantané de mouvement de prix.
    n = len(compute_price_changes(prices))
    var_series = compute_var(prices, confidence=0.95, window=n)
    cvar_series = compute_cvar(prices, confidence=0.95, window=n)

    var_clean = var_series.dropna()
    cvar_clean = cvar_series.dropna()

    return {
        "mean_price":         mean_price,
        "volatility_eur_mwh": vol_eur,
        "volatility_pct":     vol_pct,
        "var_95_eur_mwh":     var_clean.iloc[-1] if not var_clean.empty else None,
        "cvar_95_eur_mwh":    cvar_clean.iloc[-1] if not cvar_clean.empty else None,
        "skewness":           float(stats.skew(daily_prices)) if n_daily > 2 else None,
        "kurtosis":           float(stats.kurtosis(daily_prices)) if n_daily > 3 else None,
        "periods_per_year":   _detect_periods_per_year(prices),
    }


if __name__ == "__main__":
    import sys
    from datetime import datetime, timedelta
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from data.fetcher import fetch_day_ahead_prices

    end = datetime.utcnow()
    start = end - timedelta(days=30)

    print(f"Récupération des prix day-ahead France du {start.date()} au {end.date()}...")
    prices = fetch_day_ahead_prices("FR", start, end)["price_eur_mwh"]
    print(f"{len(prices)} points de données chargés.")

    summary = risk_summary(prices)

    def fmt(value, fmt_spec, unit="", fallback="n/a"):
        return f"{format(value, fmt_spec)}{unit}" if value is not None else fallback

    print(f"\nFréquence détectée : {summary['periods_per_year']} périodes/an")
    print(f"Prix moyen         : {fmt(summary['mean_price'], '.2f', ' EUR/MWh')}")
    print()
    print("=" * 52)
    print("Résumé de risque — France (30 derniers jours)")
    print("=" * 52)
    print(f"  Volatilité (std jour.) : {fmt(summary['volatility_eur_mwh'], '.2f', ' EUR/MWh')}")
    print(f"  Coefficient variation  : {fmt(summary['volatility_pct'], '.1f', ' %')}")
    print(f"  VaR 95%               : {fmt(summary['var_95_eur_mwh'], '.2f', ' EUR/MWh')}")
    print(f"  CVaR 95%              : {fmt(summary['cvar_95_eur_mwh'], '.2f', ' EUR/MWh')}")
    print(f"  Skewness              : {fmt(summary['skewness'], '.3f')}")
    print(f"  Kurtosis (excès)      : {fmt(summary['kurtosis'], '.3f')}")

    skew = summary["skewness"]
    kurt = summary["kurtosis"]
    if skew is not None and kurt is not None:
        print("\nInterprétation :")
        print(f"  Variations {'asymétriques à la hausse' if skew > 0 else 'asymétriques à la baisse'} (skew={skew:+.2f})")
        if kurt > 1:
            print(f"  Queues épaisses — pics de prix extrêmes probables (kurtosis={kurt:.2f})")
        else:
            print(f"  Distribution proche de la normale (kurtosis={kurt:.2f})")
