# T4_02 helper: Q1 population forecast and constrained-demand recomputation.

import numpy as np


def run_q1_pipeline(config):
    """Run the same Q1 population and constrained-demand logic used by T1."""
    pop_current = config["population_matrix"].astype(float).copy()
    transition_matrix = config["transition_matrix"].astype(float)
    transition_params = config.get("transition_params", {})
    death_rate = transition_params.get("death_rate", 0.05)
    growth_rate = transition_params.get("elder_growth_rate", 0.07)

    forecast_history = []
    for year in range(1, 6):
        transitioned = (1.0 - death_rate) * (pop_current @ transition_matrix)
        new_elderly = np.zeros_like(pop_current, dtype=float)
        new_elderly[:, 0] = growth_rate * pop_current.sum(axis=1)
        pop_current = transitioned + new_elderly
        forecast_history.append({
            "year": year,
            "total_all": float(pop_current.sum()),
            "total_by_type": tuple(float(v) for v in pop_current.sum(axis=0)),
        })

    demand_rate = config["demand_rate_matrix"].astype(float)
    prices = config["service_prices"].astype(float)
    income = config["monthly_income_per_capita"].astype(float)
    caps = config["spending_cap_ratios"].astype(float)
    theoretical_expense = demand_rate @ prices
    spending_limit = income[:, None] * caps[None, :]
    ratio = np.divide(
        spending_limit,
        theoretical_expense[None, :],
        out=np.ones_like(spending_limit),
        where=theoretical_expense[None, :] > 0,
    )
    reduction = np.minimum(1.0, ratio)

    constrained_monthly = np.zeros((config["n_communities"], config["n_services"]), dtype=float)
    for type_idx in range(config["n_elderly_types"]):
        constrained_monthly += (
            pop_current[:, type_idx:type_idx + 1]
            * demand_rate[type_idx:type_idx + 1, :]
            * reduction[:, type_idx:type_idx + 1]
        )
    constrained_daily = constrained_monthly / config.get("time_constants", {}).get("days_per_month", 30)

    return pop_current, constrained_daily, forecast_history
