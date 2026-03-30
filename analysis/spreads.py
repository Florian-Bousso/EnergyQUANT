"""
Analyse des spreads de marché énergétique européen.

Calcul et suivi des spreads clés : spark spread (gaz vs électricité),
dark spread (charbon vs électricité), clean spreads (avec coût carbone),
spreads géographiques inter-zones et spreads temporels (peak/off-peak).
"""

from typing import Union

import pandas as pd


Numeric = Union[float, pd.Series]


def compute_spark_spread(
    power_price: Numeric,
    gas_price: Numeric,
    efficiency: float = 0.49,
) -> Numeric:
    """
    Calcule le spark spread entre l'électricité et le gaz naturel.

    Le spark spread mesure la rentabilité théorique d'une centrale à gaz :
    c'est la différence entre le prix de l'électricité produite et le coût
    du gaz nécessaire pour la produire, compte tenu du rendement thermique.

    Formule : spark_spread = power_price - (gas_price / efficiency)

    Paramètres
    ----------
    power_price : float ou pd.Series
        Prix de l'électricité en EUR/MWh.
    gas_price : float ou pd.Series
        Prix du gaz naturel en EUR/MWh (base thermique).
    efficiency : float
        Rendement thermique de la centrale à gaz (défaut : 0.49, soit 49 %).

    Retourne
    --------
    float ou pd.Series
        Spark spread en EUR/MWh, même type que les entrées.
    """
    return power_price - (gas_price / efficiency)


def compute_dark_spread(
    power_price: Numeric,
    coal_price: Numeric,
    efficiency: float = 0.35,
) -> Numeric:
    """
    Calcule le dark spread entre l'électricité et le charbon.

    Le dark spread mesure la rentabilité théorique d'une centrale à charbon :
    c'est la différence entre le prix de l'électricité produite et le coût
    du charbon nécessaire pour la produire, compte tenu du rendement thermique.

    Formule : dark_spread = power_price - (coal_price / efficiency)

    Paramètres
    ----------
    power_price : float ou pd.Series
        Prix de l'électricité en EUR/MWh.
    coal_price : float ou pd.Series
        Prix du charbon en EUR/MWh (base thermique).
    efficiency : float
        Rendement thermique de la centrale à charbon (défaut : 0.35, soit 35 %).

    Retourne
    --------
    float ou pd.Series
        Dark spread en EUR/MWh, même type que les entrées.
    """
    return power_price - (coal_price / efficiency)


def compute_clean_spark_spread(
    power_price: Numeric,
    gas_price: Numeric,
    carbon_price: Numeric,
    efficiency: float = 0.49,
    emission_factor: float = 0.202,
) -> Numeric:
    """
    Calcule le clean spark spread en intégrant le coût des quotas carbone.

    Le clean spark spread affine le spark spread classique en déduisant le
    coût d'achat des quotas CO2 (EUA) liés à la combustion du gaz. C'est
    la mesure de rentabilité de référence pour les centrales à gaz dans le
    cadre du marché européen du carbone (EU ETS).

    Formule : clean_spark_spread = spark_spread - (carbon_price * emission_factor)
              où spark_spread = power_price - (gas_price / efficiency)

    Paramètres
    ----------
    power_price : float ou pd.Series
        Prix de l'électricité en EUR/MWh.
    gas_price : float ou pd.Series
        Prix du gaz naturel en EUR/MWh (base thermique).
    carbon_price : float ou pd.Series
        Prix des quotas CO2 (EUA) en EUR/tCO2.
    efficiency : float
        Rendement thermique de la centrale à gaz (défaut : 0.49, soit 49 %).
    emission_factor : float
        Facteur d'émission du gaz naturel en tCO2/MWh électrique produit
        (défaut : 0.202, valeur standard pour le gaz naturel).

    Retourne
    --------
    float ou pd.Series
        Clean spark spread en EUR/MWh, même type que les entrées.
    """
    spark = compute_spark_spread(power_price, gas_price, efficiency)
    carbon_cost = carbon_price * emission_factor
    return spark - carbon_cost


def spread_summary(spreads: pd.Series) -> dict:
    """
    Calcule les statistiques descriptives d'une série de spreads.

    Paramètres
    ----------
    spreads : pd.Series
        Série temporelle de spreads en EUR/MWh.

    Retourne
    --------
    dict
        Dictionnaire contenant :
        - mean (float) : moyenne du spread
        - std (float)  : écart-type du spread
        - min (float)  : valeur minimale
        - max (float)  : valeur maximale
        - pct_positive (float) : pourcentage de périodes où le spread est positif
    """
    return {
        "mean": spreads.mean(),
        "std": spreads.std(),
        "min": spreads.min(),
        "max": spreads.max(),
        "pct_positive": (spreads > 0).mean() * 100,
    }


if __name__ == "__main__":
    import numpy as np

    rng = np.random.default_rng(42)
    n = 168  # une semaine horaire

    power = pd.Series(60 + rng.normal(0, 8, n), name="power_eur_mwh")
    gas = pd.Series(30 + rng.normal(0, 3, n), name="gas_eur_mwh")
    coal = pd.Series(12 + rng.normal(0, 1.5, n), name="coal_eur_mwh")
    carbon = pd.Series(65 + rng.normal(0, 4, n), name="carbon_eur_tco2")

    spark = compute_spark_spread(power, gas)
    dark = compute_dark_spread(power, coal)
    clean_spark = compute_clean_spark_spread(power, gas, carbon)

    print("=" * 50)
    print("Hypothèses de marché (moyennes)")
    print("=" * 50)
    print(f"  Electricité : {power.mean():.2f} EUR/MWh")
    print(f"  Gaz         : {gas.mean():.2f} EUR/MWh")
    print(f"  Charbon     : {coal.mean():.2f} EUR/MWh")
    print(f"  Carbone     : {carbon.mean():.2f} EUR/tCO2")

    for label, series in [("Spark spread", spark), ("Dark spread", dark), ("Clean spark spread", clean_spark)]:
        stats = spread_summary(series)
        print(f"\n--- {label} ---")
        print(f"  Moyenne      : {stats['mean']:+.2f} EUR/MWh")
        print(f"  Écart-type   : {stats['std']:.2f} EUR/MWh")
        print(f"  Min / Max    : {stats['min']:+.2f} / {stats['max']:+.2f} EUR/MWh")
        print(f"  % positif    : {stats['pct_positive']:.1f} %")
