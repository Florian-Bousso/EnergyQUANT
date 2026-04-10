"""
Simplified merit order model for the French electricity market.

The merit order ranks power generation technologies by their short-run
marginal cost (SRMC), from cheapest to most expensive. The intersection
of this stack with the demand level determines the marginal technology —
the one that sets the wholesale electricity price at any given moment.

Marginal cost formulas
----------------------
Coal SRMC  = (coal_price_eur_tonne / 8.14) / 0.35  +  carbon_price * 0.34
Gas CCGT   = gas_price / 0.49  +  carbon_price * 0.202

where:
    8.14     : thermal content of coal (MWh/tonne)
    0.35     : coal plant thermal efficiency
    0.49     : CCGT thermal efficiency
    0.34     : coal emission factor (tCO2/MWh_elec)
    0.202    : gas emission factor (tCO2/MWh_elec)
"""

import pandas as pd

CAPACITY_SOURCE = "RTE — French installed capacity, end of 2025"


def get_merit_order(
    gas_price: float,
    coal_price_tonne: float,
    carbon_price: float,
) -> pd.DataFrame:
    """
    Build the French merit order stack for given commodity prices.

    Computes the short-run marginal cost for each technology and returns
    the full stack sorted by ascending marginal cost, with cumulative
    installed capacity to identify the marginal producer at any demand level.

    Parameters
    ----------
    gas_price : float
        TTF natural gas price in EUR/MWh (thermal).
    coal_price_tonne : float
        ARA coal price in EUR/tonne.
    carbon_price : float
        EU carbon allowance (EUA) price in EUR/tCO2.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - technology          (str)   : technology name
        - installed_gw        (float) : nameplate installed capacity in GW
        - capacity_factor     (float) : average capacity factor (0–1)
        - capacity_gw         (float) : available capacity (installed × CF) in GW
        - marginal_cost       (float) : short-run marginal cost in EUR/MWh
        - cumulative_capacity (float) : cumulative available capacity from cheapest, in GW
        Sorted by marginal_cost ascending.
    """
    coal_srmc = (coal_price_tonne / 8.14) / 0.35 + carbon_price * 0.34
    gas_ccgt_srmc = gas_price / 0.49 + carbon_price * 0.202

    technologies = [
        # technology            installed_gw  capacity_factor  available_gw  marginal_cost
        {"technology": "Nuclear",           "installed_gw": 63.0, "capacity_factor": 0.72, "capacity_gw": 45.4, "marginal_cost": 8.0},
        {"technology": "Hydro run-of-river","installed_gw": 12.9, "capacity_factor": 0.45, "capacity_gw":  5.8, "marginal_cost": 5.0},
        {"technology": "Hydro reservoir",   "installed_gw": 12.8, "capacity_factor": 0.75, "capacity_gw":  9.6, "marginal_cost": 10.0},
        {"technology": "Wind onshore",      "installed_gw": 23.9, "capacity_factor": 0.25, "capacity_gw":  6.0, "marginal_cost": 0.0},
        {"technology": "Wind offshore",     "installed_gw":  1.9, "capacity_factor": 0.35, "capacity_gw":  0.7, "marginal_cost": 0.0},
        {"technology": "Solar",             "installed_gw": 30.4, "capacity_factor": 0.15, "capacity_gw":  4.6, "marginal_cost": 0.0},
        {"technology": "Coal",              "installed_gw":  1.8, "capacity_factor": 0.50, "capacity_gw":  0.9, "marginal_cost": round(coal_srmc, 2)},
        {"technology": "Gas CCGT",          "installed_gw": 12.4, "capacity_factor": 1.00, "capacity_gw": 12.4, "marginal_cost": round(gas_ccgt_srmc, 2)},
        {"technology": "Oil",               "installed_gw":  3.0, "capacity_factor": 1.00, "capacity_gw":  3.0, "marginal_cost": 200.0},
    ]

    df = pd.DataFrame(technologies).sort_values("marginal_cost", ignore_index=True)
    df["cumulative_capacity"] = df["capacity_gw"].cumsum()
    return df


def get_marginal_technology(merit_order_df: pd.DataFrame, demand_gw: float) -> dict:
    """
    Identify the marginal technology for a given demand level.

    The marginal technology is the most expensive plant that must be
    dispatched to meet demand — it sets the market-clearing price.
    If demand exceeds total installed capacity, the last technology
    in the stack is returned.

    Parameters
    ----------
    merit_order_df : pd.DataFrame
        Merit order DataFrame as returned by get_merit_order().
    demand_gw : float
        Demand level in GW.

    Returns
    -------
    dict
        Dictionary containing:
        - technology   (str)   : name of the marginal technology
        - marginal_cost (float): its short-run marginal cost in EUR/MWh
        - margin_gw    (float) : spare capacity above demand within this technology's block
    """
    for _, row in merit_order_df.iterrows():
        if row["cumulative_capacity"] >= demand_gw:
            margin = row["cumulative_capacity"] - demand_gw
            return {
                "technology":    row["technology"],
                "marginal_cost": row["marginal_cost"],
                "margin_gw":     round(margin, 2),
            }
    # Demand exceeds total capacity — return last entry
    last = merit_order_df.iloc[-1]
    return {
        "technology":    last["technology"],
        "marginal_cost": last["marginal_cost"],
        "margin_gw":     0.0,
    }


if __name__ == "__main__":
    GAS_PRICE      = 45    # EUR/MWh (TTF)
    COAL_PRICE     = 110   # EUR/tonne (ARA)
    CARBON_PRICE   = 65    # EUR/tCO2 (EUA)
    DEMAND_GW      = 60    # GW

    df = get_merit_order(GAS_PRICE, COAL_PRICE, CARBON_PRICE)

    print("=" * 82)
    print("French Merit Order")
    print(f"Gas={GAS_PRICE} EUR/MWh | Coal={COAL_PRICE} EUR/t | Carbon={CARBON_PRICE} EUR/tCO2")
    print(f"Source: {CAPACITY_SOURCE}")
    print("=" * 82)
    print(f"  {'Technology':<22} {'Installed':>9} {'CF':>6} {'Available':>10} {'SRMC (EUR/MWh)':>15} {'Cum. (GW)':>10}")
    print(f"  {'-'*22} {'-'*9} {'-'*6} {'-'*10} {'-'*15} {'-'*10}")
    for _, row in df.iterrows():
        print(
            f"  {row['technology']:<22} {row['installed_gw']:>9.1f} "
            f"{row['capacity_factor']:>6.0%} {row['capacity_gw']:>10.1f} "
            f"{row['marginal_cost']:>15.2f} {row['cumulative_capacity']:>10.1f}"
        )

    marginal = get_marginal_technology(df, DEMAND_GW)
    total_installed = df["installed_gw"].sum()
    total_available = df["capacity_gw"].sum()
    print(f"\nDemand: {DEMAND_GW} GW  (total installed: {total_installed:.0f} GW, total available: {total_available:.1f} GW)")
    print(f"Marginal technology : {marginal['technology']}")
    print(f"Marginal cost       : {marginal['marginal_cost']:.2f} EUR/MWh")
    print(f"Spare margin        : {marginal['margin_gw']:.1f} GW within this block")
