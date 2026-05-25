# T4_02 helper: Q3 pricing, satisfaction feedback, and financial evaluation.

import copy
import itertools
import pickle
from pathlib import Path

import numpy as np


MODEL_DIR = Path("model")
CAPACITY_LIMIT_RATIO = 1.02
PAID_SERVICE_COUNT = 5
PROFIT_MARGIN_MIN = 0.0
PROFIT_MARGIN_MAX = 8.0
MAX_T3_S2_ITERATIONS = 30
SATISFACTION_TOLERANCE = 1e-8
LOAD_TOLERANCE = 1e-6
BEAM_WIDTH = 5000


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def scale_key(scale_code):
    return {1: "small", 2: "medium", 3: "large"}[int(scale_code)]


def get_s1_score(distance, rules):
    for seg in rules["segments"]:
        if seg.get("min_distance", -1) < distance <= seg.get("max_distance", float("inf")):
            return float(seg["score"])
    return 0.0


def get_s2_score(utilization, rules):
    for seg in rules["segments"]:
        if seg.get("min_utilization", -1) < utilization <= seg.get("max_utilization", float("inf")):
            return float(seg["score"])
    return 0.60


def get_s3_score(price, base_price, rules):
    if base_price <= 0:
        return 1.0
    ratio = price / base_price
    for seg in rules["segments"]:
        if seg.get("min_ratio", -1) < ratio <= seg.get("max_ratio", float("inf")):
            return float(seg["score"])
    return 0.60


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = weights.sum()
    return float(np.sum(values * weights) / total) if total > 0 else 0.0


def build_q3_fixed_inputs(config, pop_year5, constrained_daily, q2_solution):
    active = list(q2_solution["active_stations"])
    assignments = {int(k): int(v) for k, v in q2_solution["assignments"].items() if int(v) != -1}
    covered = sorted(assignments)
    pop_totals = pop_year5.sum(axis=1)
    s1_by_community = np.zeros(config["n_communities"], dtype=float)
    for community_idx in covered:
        station_idx = assignments[community_idx]
        s1_by_community[community_idx] = get_s1_score(
            config["distance_matrix"][community_idx, station_idx],
            config["satisfaction_rules"]["distance"],
        )

    return {
        "active_stations": active,
        "assignments": assignments,
        "covered_communities": covered,
        "covered_mask": np.array([idx in assignments for idx in range(config["n_communities"])], dtype=bool),
        "elderly_total_by_community": pop_totals,
        "covered_elderly": float(sum(pop_totals[i] for i in covered)),
        "daily_demand": constrained_daily,
        "daily_paid_demand": constrained_daily[:, :PAID_SERVICE_COUNT],
        "daily_paid_demand_totals": constrained_daily[:, :PAID_SERVICE_COUNT].sum(axis=1),
        "s1_by_community": s1_by_community,
    }


def load_station_templates(config):
    candidate_path = MODEL_DIR / "T3_02_candidate_price_plans.pkl"
    if candidate_path.exists():
        data = load_pickle(candidate_path)
        templates = data.get("station_templates", [])
        if templates:
            return templates

    multipliers = [0.60, 0.65, 0.70, 0.75, 0.80, 0.825, 0.85, 0.875, 0.90, 0.95, 1.00]
    templates = []
    for template_id, multiplier in enumerate(multipliers):
        prices = {
            service: float(config["service_prices"][idx] * multiplier)
            for idx, service in enumerate(config["service_names"][:PAID_SERVICE_COUNT])
        }
        prices[config["service_names"][5]] = 0.0
        templates.append({
            "template_id": template_id,
            "family": "fallback_grid",
            "label": f"all paid services {multiplier:.3f}x base price",
            "multipliers": tuple([multiplier] * PAID_SERVICE_COUNT),
            "uniform_multiplier": multiplier,
            "prices": prices,
            "price_tuple": tuple(prices[service] for service in config["service_names"][:PAID_SERVICE_COUNT]),
        })
    return templates


def package_plan(active_stations, templates_by_station, plan_id):
    station_prices = {
        station_idx: dict(template["prices"])
        for station_idx, template in templates_by_station.items()
    }
    station_template_ids = {
        station_idx: int(template["template_id"])
        for station_idx, template in templates_by_station.items()
    }
    station_uniform_multipliers = {
        station_idx: float(template.get("uniform_multiplier", np.mean(template["multipliers"])))
        for station_idx, template in templates_by_station.items()
    }
    price_plan_tuple = tuple(
        float(station_prices[station_idx][service])
        for station_idx in active_stations
        for service in list(station_prices[station_idx])[:PAID_SERVICE_COUNT]
    )
    return {
        "plan_id": plan_id,
        "station_template_ids": station_template_ids,
        "station_uniform_multipliers": station_uniform_multipliers,
        "station_prices": station_prices,
        "price_plan_tuple": price_plan_tuple,
    }


def station_service_s3(config, station_prices):
    base_prices = config["service_prices"][:PAID_SERVICE_COUNT]
    paid_services = config["service_names"][:PAID_SERVICE_COUNT]
    return np.array([
        get_s3_score(station_prices[service], base_prices[idx], config["satisfaction_rules"]["price"])
        for idx, service in enumerate(paid_services)
    ], dtype=float)


def compute_weighted_s3_by_community(config, plan, fixed_inputs):
    weighted_s3 = np.zeros(config["n_communities"], dtype=float)
    station_s3 = {
        station_idx: station_service_s3(config, plan["station_prices"][station_idx])
        for station_idx in fixed_inputs["active_stations"]
    }
    for community_idx in fixed_inputs["covered_communities"]:
        station_idx = fixed_inputs["assignments"][community_idx]
        demand = fixed_inputs["daily_paid_demand"][community_idx]
        total = fixed_inputs["daily_paid_demand_totals"][community_idx]
        weighted_s3[community_idx] = float(np.sum(demand * station_s3[station_idx]) / total) if total > 0 else 1.0
    return weighted_s3, station_s3


def compute_satisfaction_and_loads(config, fixed_inputs, response_scores, weighted_s3):
    satisfaction = np.zeros(config["n_communities"], dtype=float)
    station_daily_by_service = {
        station_idx: np.zeros(config["n_services"], dtype=float)
        for station_idx in fixed_inputs["active_stations"]
    }
    for community_idx in fixed_inputs["covered_communities"]:
        station_idx = fixed_inputs["assignments"][community_idx]
        satisfaction[community_idx] = (
            config["satisfaction_weights"]["distance"] * fixed_inputs["s1_by_community"][community_idx]
            + config["satisfaction_weights"]["response"] * response_scores[station_idx]
            + config["satisfaction_weights"]["price"] * weighted_s3[community_idx]
        )
        station_daily_by_service[station_idx] += (
            fixed_inputs["daily_demand"][community_idx] * satisfaction[community_idx]
        )
    station_loads = {
        station_idx: float(station_daily_by_service[station_idx].sum())
        for station_idx in fixed_inputs["active_stations"]
    }
    station_utilizations = {
        station_idx: station_loads[station_idx] / config["_scenario_station_capacities"][station_idx]
        for station_idx in fixed_inputs["active_stations"]
    }
    return satisfaction, station_daily_by_service, station_loads, station_utilizations


def choose_feedback_state(states):
    return max(
        states,
        key=lambda state: (
            state["avg_satisfaction_covered"],
            -max(state["station_utilizations"].values(), default=0.0),
        ),
    )


def solve_q3_s2_feedback(config, fixed_inputs, weighted_s3):
    response_scores = {
        station_idx: get_s2_score(
            config["_scenario_initial_utilizations"].get(station_idx, 0.0),
            config["satisfaction_rules"]["response"],
        )
        for station_idx in fixed_inputs["active_stations"]
    }
    seen = {}
    states = []
    for iteration in range(1, MAX_T3_S2_ITERATIONS + 1):
        satisfaction, station_daily_by_service, station_loads, station_utilizations = compute_satisfaction_and_loads(
            config, fixed_inputs, response_scores, weighted_s3
        )
        avg_satisfaction = weighted_mean(
            satisfaction[fixed_inputs["covered_mask"]],
            fixed_inputs["elderly_total_by_community"][fixed_inputs["covered_mask"]],
        )
        state = {
            "satisfaction_by_community": satisfaction,
            "station_effective_daily_by_service": station_daily_by_service,
            "station_loads": station_loads,
            "station_utilizations": station_utilizations,
            "station_response_scores": dict(response_scores),
            "avg_satisfaction_covered": avg_satisfaction,
            "iterations": iteration,
            "converged": False,
            "cycle_detected": False,
        }
        states.append(state)
        next_response_scores = {
            station_idx: get_s2_score(station_utilizations[station_idx], config["satisfaction_rules"]["response"])
            for station_idx in fixed_inputs["active_stations"]
        }
        if all(abs(next_response_scores[j] - response_scores[j]) <= SATISFACTION_TOLERANCE for j in fixed_inputs["active_stations"]):
            state["converged"] = True
            return state
        key = tuple((j, response_scores[j]) for j in fixed_inputs["active_stations"])
        if key in seen:
            cycle_state = choose_feedback_state(states[seen[key]:])
            cycle_state["cycle_detected"] = True
            return cycle_state
        seen[key] = len(states) - 1
        response_scores = next_response_scores
    return choose_feedback_state(states)


def compute_station_financials(config, plan, station_daily_by_service, fixed_inputs):
    days = config.get("time_constants", {}).get("days_per_year", 365)
    direct_costs = config["direct_costs"].astype(float)
    paid_services = config["service_names"][:PAID_SERVICE_COUNT]
    station_financials = {}

    for station_idx in fixed_inputs["active_stations"]:
        layout_scale = int(config["_scenario_layout"][station_idx])
        spec = config["station_specs"][scale_key(layout_scale)]
        daily_by_service = station_daily_by_service[station_idx]
        annual_by_service = daily_by_service * days
        annual_service_revenue = sum(
            annual_by_service[idx] * plan["station_prices"][station_idx][service] / 10000.0
            for idx, service in enumerate(paid_services)
        )
        annual_variable_cost = float(np.sum(annual_by_service * direct_costs) / 10000.0)
        daily_paid_visits = float(daily_by_service[:PAID_SERVICE_COUNT].sum())
        daily_subsidy_request = daily_paid_visits * 2.0
        daily_subsidy_cap = float(spec.get("subsidy_cap_day", {1: 1000, 2: 1800, 3: 2600}[layout_scale]))
        daily_subsidy_actual = min(daily_subsidy_request, daily_subsidy_cap)
        annual_subsidy = daily_subsidy_actual * days / 10000.0
        annual_fixed_management_cost = spec["daily_fixed_cost"] * days / 10000.0
        annual_depreciation = spec["build_cost"] / 20.0
        annual_operating_cost = annual_fixed_management_cost + annual_depreciation
        annual_profit = annual_service_revenue + annual_subsidy - annual_variable_cost - annual_operating_cost
        profit_margin = annual_profit / annual_operating_cost * 100.0 if annual_operating_cost > 0 else 0.0
        station_financials[station_idx] = {
            "annual_effective_visits": float(annual_by_service.sum()),
            "effective_daily_by_service": tuple(float(v) for v in daily_by_service),
            "paid_effective_daily_visits": daily_paid_visits,
            "daily_subsidy_request_yuan": float(daily_subsidy_request),
            "daily_subsidy_cap_yuan": daily_subsidy_cap,
            "daily_subsidy_actual_yuan": float(daily_subsidy_actual),
            "annual_service_revenue": float(annual_service_revenue),
            "annual_variable_cost": annual_variable_cost,
            "annual_fixed_management_cost": float(annual_fixed_management_cost),
            "annual_depreciation": float(annual_depreciation),
            "annual_operating_cost": float(annual_operating_cost),
            "annual_subsidy": float(annual_subsidy),
            "annual_total_revenue": float(annual_service_revenue + annual_subsidy),
            "annual_total_cost": float(annual_operating_cost + annual_variable_cost),
            "annual_profit": float(annual_profit),
            "profit_margin": float(profit_margin),
        }

    totals = {
        key: float(sum(fin[key] for fin in station_financials.values()))
        for key in [
            "annual_effective_visits",
            "annual_service_revenue",
            "annual_variable_cost",
            "annual_fixed_management_cost",
            "annual_depreciation",
            "annual_operating_cost",
            "annual_subsidy",
            "annual_total_revenue",
            "annual_total_cost",
            "annual_profit",
        ]
    }
    totals["profit_margin"] = (
        totals["annual_profit"] / totals["annual_operating_cost"] * 100.0
        if totals["annual_operating_cost"] > 0 else 0.0
    )
    return station_financials, totals


def evaluate_station_margin_constraints(station_financials):
    margins = {
        station_idx: float(financials["profit_margin"])
        for station_idx, financials in station_financials.items()
    }
    failed = [
        station_idx
        for station_idx, margin in margins.items()
        if margin < PROFIT_MARGIN_MIN - 1e-9 or margin > PROFIT_MARGIN_MAX + 1e-9
    ]
    return {
        "station_profit_margins": margins,
        "station_margin_feasible": len(failed) == 0,
        "station_margin_failed": failed,
        "min_station_profit_margin": float(min(margins.values(), default=0.0)),
        "max_station_profit_margin": float(max(margins.values(), default=0.0)),
    }


def evaluate_price_plan(config, fixed_inputs, plan):
    weighted_s3, station_s3 = compute_weighted_s3_by_community(config, plan, fixed_inputs)
    feedback_state = solve_q3_s2_feedback(config, fixed_inputs, weighted_s3)
    station_financials, system_financials = compute_station_financials(
        config, plan, feedback_state["station_effective_daily_by_service"], fixed_inputs
    )
    station_overload_rates = {}
    station_capacity_feasible = {}
    for station_idx in fixed_inputs["active_stations"]:
        capacity = config["_scenario_station_capacities"][station_idx]
        elastic = capacity * CAPACITY_LIMIT_RATIO
        load = feedback_state["station_loads"][station_idx]
        station_overload_rates[station_idx] = max(0.0, load - capacity) / capacity if capacity > 0 else 0.0
        station_capacity_feasible[station_idx] = load <= elastic + LOAD_TOLERANCE
    capacity_feasible = all(station_capacity_feasible.values())
    margin_audit = evaluate_station_margin_constraints(station_financials)
    system_margin_feasible = PROFIT_MARGIN_MIN <= system_financials["profit_margin"] <= PROFIT_MARGIN_MAX
    avg_system = feedback_state["avg_satisfaction_covered"] * config["_scenario_coverage_rate"]
    elder_payment = system_financials["annual_service_revenue"]
    return {
        "plan_id": plan["plan_id"],
        "station_template_ids": dict(plan["station_template_ids"]),
        "station_uniform_multipliers": dict(plan["station_uniform_multipliers"]),
        "price_plan_tuple": tuple(float(v) for v in plan["price_plan_tuple"]),
        "station_prices": plan["station_prices"],
        "station_s3_by_service": {j: tuple(float(v) for v in station_s3[j]) for j in station_s3},
        "weighted_s3_by_community": tuple(float(v) for v in weighted_s3),
        "satisfaction_by_community": tuple(float(v) for v in feedback_state["satisfaction_by_community"]),
        "station_response_scores": dict(feedback_state["station_response_scores"]),
        "station_loads": dict(feedback_state["station_loads"]),
        "station_utilizations": dict(feedback_state["station_utilizations"]),
        "station_overload_rates": station_overload_rates,
        "station_capacity_feasible": station_capacity_feasible,
        "capacity_feasible": capacity_feasible,
        "max_overload_rate": float(max(station_overload_rates.values(), default=0.0)),
        "station_financials": station_financials,
        "system_financials": system_financials,
        "station_profit_margins": margin_audit["station_profit_margins"],
        "station_margin_feasible": margin_audit["station_margin_feasible"],
        "station_margin_failed": margin_audit["station_margin_failed"],
        "min_station_profit_margin": margin_audit["min_station_profit_margin"],
        "max_station_profit_margin": margin_audit["max_station_profit_margin"],
        "system_margin_feasible": system_margin_feasible,
        "avg_satisfaction_covered": float(feedback_state["avg_satisfaction_covered"]),
        "avg_satisfaction_system": float(avg_system),
        "weighted_avg_satisfaction": float(feedback_state["avg_satisfaction_covered"]),
        "elder_payment": float(elder_payment),
        "s2_iterations": feedback_state["iterations"],
        "s2_converged": feedback_state["converged"],
        "s2_cycle_detected": feedback_state["cycle_detected"],
        "is_feasible": capacity_feasible and margin_audit["station_margin_feasible"],
    }


def q3_sort_key(result):
    sys_fin = result["system_financials"]
    return (
        -round(result["avg_satisfaction_covered"], 10),
        round(result["elder_payment"], 10),
        round(sys_fin["annual_subsidy"], 10),
        abs(sys_fin["profit_margin"]),
        result["price_plan_tuple"],
    )


def build_template_combinations(active, templates):
    for combo in itertools.product(templates, repeat=len(active)):
        yield dict(zip(active, combo))


def approximate_partial_score(config, fixed_inputs, templates_by_station):
    covered_scores = []
    covered_weights = []
    for community_idx in fixed_inputs["covered_communities"]:
        station_idx = fixed_inputs["assignments"][community_idx]
        if station_idx not in templates_by_station:
            continue
        template = templates_by_station[station_idx]
        weighted_s3 = float(np.mean([
            get_s3_score(template["prices"][service], config["service_prices"][idx], config["satisfaction_rules"]["price"])
            for idx, service in enumerate(config["service_names"][:PAID_SERVICE_COUNT])
        ]))
        s2 = get_s2_score(
            config["_scenario_initial_utilizations"].get(station_idx, 0.0),
            config["satisfaction_rules"]["response"],
        )
        score = (
            config["satisfaction_weights"]["distance"] * fixed_inputs["s1_by_community"][community_idx]
            + config["satisfaction_weights"]["response"] * s2
            + config["satisfaction_weights"]["price"] * weighted_s3
        )
        covered_scores.append(score)
        covered_weights.append(fixed_inputs["elderly_total_by_community"][community_idx])
    avg_score = weighted_mean(covered_scores, covered_weights) if covered_scores else 0.0
    mean_multiplier = np.mean([
        template.get("uniform_multiplier", np.mean(template["multipliers"]))
        for template in templates_by_station.values()
    ])
    return avg_score, -float(mean_multiplier)


def build_beam_template_sets(config, fixed_inputs, templates):
    beams = [({}, (0.0, 0.0))]
    active = fixed_inputs["active_stations"]
    for station_idx in active:
        candidates = []
        for partial, _score in beams:
            for template in templates:
                next_partial = dict(partial)
                next_partial[station_idx] = template
                score = approximate_partial_score(config, fixed_inputs, next_partial)
                candidates.append((next_partial, score))
        candidates.sort(key=lambda item: (-item[1][0], -item[1][1], tuple(t["template_id"] for t in item[0].values())))
        beams = candidates[:BEAM_WIDTH]
    return [partial for partial, _score in beams]


def run_q3_pipeline(config, pop_year5, constrained_daily, q2_solution):
    """Solve Q3 with station-level, service-level pricing and official subsidy caps."""
    if q2_solution is None or not q2_solution.get("active_stations"):
        return None

    fixed_inputs = build_q3_fixed_inputs(config, pop_year5, constrained_daily, q2_solution)
    if fixed_inputs["covered_elderly"] <= 0:
        return None

    scenario_config = copy.deepcopy(config)
    scenario_config["_scenario_layout"] = q2_solution["layout"]
    scenario_config["_scenario_station_capacities"] = q2_solution["station_capacities"]
    scenario_config["_scenario_initial_utilizations"] = q2_solution["station_utilizations"]
    scenario_config["_scenario_coverage_rate"] = q2_solution["coverage_rate"]

    templates = load_station_templates(config)
    active = fixed_inputs["active_stations"]
    if len(active) <= 3:
        template_sets = build_template_combinations(active, templates)
        search_mode = "full"
        candidate_count = len(templates) ** len(active)
    else:
        template_sets = build_beam_template_sets(scenario_config, fixed_inputs, templates)
        search_mode = "compressed_beam"
        candidate_count = len(template_sets)

    best = None
    feasible_count = 0
    for plan_id, templates_by_station in enumerate(template_sets):
        plan = package_plan(active, templates_by_station, plan_id)
        result = evaluate_price_plan(scenario_config, fixed_inputs, plan)
        if not result["is_feasible"]:
            continue
        feasible_count += 1
        if best is None or q3_sort_key(result) < q3_sort_key(best):
            best = result

    if best is None:
        return {
            "pricing_search_mode": search_mode,
            "candidate_count": int(candidate_count),
            "feasible_count": 0,
            "is_feasible": False,
            "station_prices": {},
            "station_financials": {},
            "system_financials": {},
            "station_profit_margins": {},
            "station_margin_feasible": False,
            "station_margin_failed": [],
            "min_station_profit_margin": 0.0,
            "max_station_profit_margin": 0.0,
            "system_margin_feasible": False,
            "annual_subsidy": 0.0,
            "annual_profit": 0.0,
            "profit_margin": 0.0,
            "avg_satisfaction_covered": 0.0,
            "avg_satisfaction_system": 0.0,
            "coverage_rate": q2_solution["coverage_rate"],
        }

    best.update({
        "pricing_search_mode": search_mode,
        "candidate_count": int(candidate_count),
        "feasible_count": int(feasible_count),
        "annual_subsidy": best["system_financials"]["annual_subsidy"],
        "annual_profit": best["system_financials"]["annual_profit"],
        "profit_margin": best["system_financials"]["profit_margin"],
        "min_station_profit_margin": best["min_station_profit_margin"],
        "max_station_profit_margin": best["max_station_profit_margin"],
        "coverage_rate": q2_solution["coverage_rate"],
    })
    return best
