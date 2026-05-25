# T4_02 helper: Q2 site layout enumeration, assignment, and capacity audit.

import itertools

import numpy as np

from T2_02_satisfaction_assignment import solve_layout_assignment


CAPACITY_LIMIT_RATIO = 1.02
LOAD_TOLERANCE = 1e-6


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    return float(np.sum(values * weights) / total) if total > 0 else 0.0


def station_label(config, station_idx):
    return config["communities"][station_idx]


def compute_capacity_audit(q2_solution):
    audit = {}
    for station_idx in q2_solution["active_stations"]:
        nominal = float(q2_solution["station_capacities"][station_idx])
        load = float(q2_solution["station_loads"].get(station_idx, 0.0))
        elastic_limit = nominal * CAPACITY_LIMIT_RATIO
        audit[station_idx] = {
            "nominal_capacity": nominal,
            "elastic_limit": elastic_limit,
            "effective_load": load,
            "utilization_rate": load / nominal if nominal > 0 else 0.0,
            "overload_rate": max(0.0, load - nominal) / nominal if nominal > 0 else 0.0,
            "elastic_feasible": load <= elastic_limit + LOAD_TOLERANCE,
        }
    return audit


def add_q2_metrics(config, q2_solution, pop_year5, constrained_daily, has_full_coverage_feasible):
    pop_totals = pop_year5.sum(axis=1)
    total_pop = float(pop_totals.sum())
    assignments = q2_solution.get("assignments", {})
    covered = sorted(int(i) for i in assignments if assignments[i] != -1)
    uncovered = [i for i in range(config["n_communities"]) if i not in covered]
    covered_pop = float(sum(pop_totals[i] for i in covered))
    uncovered_pop = float(total_pop - covered_pop)
    coverage_rate = covered_pop / total_pop if total_pop > 0 else 0.0

    community_satisfaction = np.zeros(config["n_communities"], dtype=float)
    raw_satisfaction = q2_solution.get("community_satisfactions", {})
    if isinstance(raw_satisfaction, dict):
        for idx, value in raw_satisfaction.items():
            community_satisfaction[int(idx)] = float(value)
    else:
        raw_satisfaction = np.asarray(raw_satisfaction, dtype=float)
        community_satisfaction[:len(raw_satisfaction)] = raw_satisfaction
    avg_satisfaction_covered = weighted_mean(
        [community_satisfaction[i] for i in covered],
        [pop_totals[i] for i in covered],
    )
    avg_satisfaction_system = avg_satisfaction_covered * coverage_rate

    uncovered_daily_by_service = constrained_daily[uncovered].sum(axis=0) if uncovered else np.zeros(config["n_services"])
    q2_solution.update({
        "covered_communities": covered,
        "uncovered_communities": uncovered,
        "coverage_rate": float(coverage_rate),
        "covered_elderly": covered_pop,
        "uncovered_elderly": uncovered_pop,
        "avg_satisfaction_covered": float(avg_satisfaction_covered),
        "avg_satisfaction_system": float(avg_satisfaction_system),
        "capacity_limit_ratio": CAPACITY_LIMIT_RATIO,
        "station_elastic_capacity_limits": {
            j: float(q2_solution["station_capacities"][j]) * CAPACITY_LIMIT_RATIO
            for j in q2_solution["active_stations"]
        },
        "capacity_audit": compute_capacity_audit(q2_solution),
        "standard_capacity_audit": compute_capacity_audit(q2_solution),
        "degradation_triggered": (not has_full_coverage_feasible) or coverage_rate < 1.0 - 1e-9,
        "unmet_demand": {
            "uncovered_communities": [station_label(config, i) for i in uncovered],
            "uncovered_communities_count": len(uncovered),
            "uncovered_population": uncovered_pop,
            "uncovered_population_by_type": {
                "自理": float(pop_year5[uncovered, 0].sum()) if uncovered else 0.0,
                "半失能": float(pop_year5[uncovered, 1].sum()) if uncovered else 0.0,
                "失能": float(pop_year5[uncovered, 2].sum()) if uncovered else 0.0,
            },
            "uncovered_daily_demand_by_service": {
                service: float(uncovered_daily_by_service[idx])
                for idx, service in enumerate(config["service_names"])
            },
            "uncovered_daily_constrained_demand": float(uncovered_daily_by_service.sum()),
            "uncovered_annual_constrained_demand": float(uncovered_daily_by_service.sum() * 365.0 / 10000.0),
            "coverage_gap_pct": float((1.0 - coverage_rate) * 100.0),
        },
    })
    return q2_solution


def enumerate_budget_feasible_layouts(config, budget_limit):
    costs = np.array([
        0.0,
        config["station_specs"]["small"]["build_cost"],
        config["station_specs"]["medium"]["build_cost"],
        config["station_specs"]["large"]["build_cost"],
    ], dtype=float)
    for layout_tuple in itertools.product([0, 1, 2, 3], repeat=config["n_communities"]):
        layout = np.array(layout_tuple, dtype=np.int8)
        build_cost = float(costs[layout].sum())
        if build_cost <= budget_limit + 1e-9 and layout.sum() > 0:
            yield layout.tolist(), build_cost


def run_q2_pipeline(config, pop_year5, constrained_daily, budget_limit):
    """Solve Q2 under the inherited 2% elastic-capacity口径."""
    community_daily_demand = constrained_daily.sum(axis=1)
    elderly_by_community = pop_year5.sum(axis=1)
    coverage_mask = (config["distance_matrix"] <= 1000).astype(int)

    evaluated_count = 0
    budget_feasible_count = 0
    capacity_feasible_count = 0
    valid_solutions = []

    for layout, build_cost in enumerate_budget_feasible_layouts(config, budget_limit):
        evaluated_count += 1
        budget_feasible_count += 1
        solution = solve_layout_assignment(
            layout,
            build_cost,
            config,
            community_daily_demand,
            elderly_by_community,
            coverage_mask,
            capacity_overload_ratio=CAPACITY_LIMIT_RATIO - 1.0,
        )
        if solution is None:
            continue
        capacity_feasible_count += 1
        add_q2_metrics(config, solution, pop_year5, constrained_daily, has_full_coverage_feasible=True)
        valid_solutions.append(solution)

    if not valid_solutions:
        return None, {
            "total_layouts": 4 ** config["n_communities"],
            "budget_feasible_layouts": budget_feasible_count,
            "capacity_feasible_layouts": 0,
        }

    has_full_coverage_feasible = any(sol["coverage_rate"] >= 1.0 - 1e-9 for sol in valid_solutions)
    for sol in valid_solutions:
        add_q2_metrics(config, sol, pop_year5, constrained_daily, has_full_coverage_feasible)

    valid_solutions.sort(
        key=lambda sol: (
            -round(sol["coverage_rate"], 10),
            -round(sol["avg_satisfaction_covered"], 10),
            sol["build_cost"],
            tuple(sol["layout"]),
        )
    )
    return valid_solutions[0], {
        "total_layouts": 4 ** config["n_communities"],
        "budget_feasible_layouts": budget_feasible_count,
        "capacity_feasible_layouts": capacity_feasible_count,
    }
