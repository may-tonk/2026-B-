import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np


MAX_T3_S2_ITERATIONS = 20
SATISFACTION_TOLERANCE = 1e-10
LOAD_TOLERANCE = 1e-6


def get_s3(price, base_price):
    """Return price satisfaction from the attachment price bands."""
    ratio = float(price) / float(base_price)
    if ratio <= 1.0 + 1e-4:
        return 1.00
    if ratio <= 1.1 + 1e-4:
        return 0.90
    if ratio <= 1.2 + 1e-4:
        return 0.75
    return 0.60


def calc_s2(utilization):
    """Return response satisfaction using the same attachment bands as T2."""
    if utilization <= 0.60:
        return 1.00
    if utilization <= 0.75:
        return 0.93
    if utilization <= 0.85:
        return 0.85
    if utilization <= 0.95:
        return 0.72
    return 0.60


def load_required_inputs():
    """Load candidate pricing data and the fixed T0/T1/T2 interfaces."""
    paths = {
        'plans': Path('model/T3_02_candidate_price_plans.pkl'),
        'fin_cfg': Path('model/T3_01_financial_model_config.pkl'),
        'demand': Path('model/T1_03_constrained_demand_daily_by_service.npy'),
        'population': Path('model/T1_01_population_year5.npy'),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        print(f"[错误] 缺少前置数据文件: {missing}")
        sys.exit(1)

    with paths['plans'].open('rb') as plans_file:
        candidate_data = pickle.load(plans_file)
    with paths['fin_cfg'].open('rb') as config_file:
        financial_config = pickle.load(config_file)

    daily_demand = np.load(paths['demand'])
    population = np.load(paths['population'])
    return candidate_data, financial_config, daily_demand, population


def build_fixed_assignment_inputs(financial_config, daily_demand, population):
    """Prepare fixed T2 assignment arrays reused by every pricing candidate."""
    assignments = {
        int(community_idx): int(station_idx)
        for community_idx, station_idx in financial_config['assignments'].items()
    }
    active_stations = list(financial_config['active_stations'])
    paid_services = list(financial_config['paid_services'])
    service_names = list(financial_config['service_names'])
    paid_service_count = len(paid_services)
    num_communities = daily_demand.shape[0]

    assigned_station = np.full(num_communities, -1, dtype=int)
    s1_by_community = np.zeros(num_communities, dtype=float)
    for community_idx, station_idx in assignments.items():
        detail = financial_config['satisfaction_details_t2'].get(community_idx)
        if detail is None:
            raise KeyError(f'T2 满意度拆分缺少小区 {community_idx}')
        assigned_station[community_idx] = station_idx
        s1_by_community[community_idx] = float(detail['s1'])

    covered_mask = assigned_station >= 0
    elderly_total_by_community = np.asarray(population, dtype=float).sum(axis=1)
    covered_elderly = float(elderly_total_by_community[covered_mask].sum())
    if covered_elderly <= 0:
        raise ValueError('T3 固定归属下没有被覆盖老人，无法评价定价方案')

    station_to_position = {
        station_idx: position
        for position, station_idx in enumerate(active_stations)
    }
    assigned_station_positions = np.full(num_communities, -1, dtype=int)
    for community_idx in np.flatnonzero(covered_mask):
        assigned_station_positions[community_idx] = station_to_position[
            assigned_station[community_idx]
        ]

    return {
        'assignments': assignments,
        'active_stations': active_stations,
        'paid_services': paid_services,
        'service_names': service_names,
        'paid_service_count': paid_service_count,
        'assigned_station': assigned_station,
        'assigned_station_positions': assigned_station_positions,
        'covered_mask': covered_mask,
        's1_by_community': s1_by_community,
        'daily_demand': np.asarray(daily_demand, dtype=float),
        'daily_paid_demand': np.asarray(daily_demand[:, :paid_service_count], dtype=float),
        'daily_paid_demand_totals': np.asarray(daily_demand[:, :paid_service_count], dtype=float).sum(axis=1),
        'elderly_total_by_community': elderly_total_by_community,
        'covered_elderly': covered_elderly,
    }


def build_station_s3(plan, financial_config, fixed_inputs):
    """Compute station-service S3 scores from distributed station prices."""
    base_prices = np.asarray(
        financial_config['service_prices_base'][:fixed_inputs['paid_service_count']],
        dtype=float,
    )
    station_s3 = {}
    for station_idx in fixed_inputs['active_stations']:
        station_prices = plan['station_prices'][station_idx]
        station_s3[station_idx] = np.asarray([
            get_s3(station_prices[service], base_prices[service_idx])
            for service_idx, service in enumerate(fixed_inputs['paid_services'])
        ], dtype=float)
    return station_s3


def compute_weighted_s3_by_community(station_s3, fixed_inputs):
    """Compute each community's S3 from its assigned station price vector."""
    weighted_s3 = np.zeros_like(fixed_inputs['s1_by_community'])
    for community_idx in np.flatnonzero(fixed_inputs['covered_mask']):
        station_idx = fixed_inputs['assigned_station'][community_idx]
        paid_demand = fixed_inputs['daily_paid_demand'][community_idx]
        paid_total = fixed_inputs['daily_paid_demand_totals'][community_idx]
        if paid_total <= 0:
            weighted_s3[community_idx] = 1.0
            continue
        weighted_s3[community_idx] = float(
            np.sum(paid_demand * station_s3[station_idx]) / paid_total
        )
    return weighted_s3


def compute_satisfaction_and_station_visits(response_scores, weighted_s3, financial_config, fixed_inputs):
    """Compute composite satisfaction and station service visits for one S2 state."""
    weights = {'distance': 0.2, 'response': 0.3, 'price': 0.5}
    satisfaction = np.zeros_like(fixed_inputs['s1_by_community'])
    for community_idx in np.flatnonzero(fixed_inputs['covered_mask']):
        station_idx = fixed_inputs['assigned_station'][community_idx]
        satisfaction[community_idx] = (
            weights['distance'] * fixed_inputs['s1_by_community'][community_idx]
            + weights['response'] * response_scores[station_idx]
            + weights['price'] * weighted_s3[community_idx]
        )

    station_effective_daily_by_service = {
        station_idx: np.zeros(len(fixed_inputs['service_names']), dtype=float)
        for station_idx in fixed_inputs['active_stations']
    }
    for community_idx in np.flatnonzero(fixed_inputs['covered_mask']):
        station_idx = fixed_inputs['assigned_station'][community_idx]
        station_effective_daily_by_service[station_idx] += (
            fixed_inputs['daily_demand'][community_idx] * satisfaction[community_idx]
        )

    station_loads = {
        station_idx: float(service_visits.sum())
        for station_idx, service_visits in station_effective_daily_by_service.items()
    }
    station_utilizations = {
        station_idx: station_loads[station_idx] / financial_config['station_capacities'][station_idx]
        for station_idx in fixed_inputs['active_stations']
    }
    return satisfaction, station_effective_daily_by_service, station_loads, station_utilizations


def response_state_key(response_scores, active_stations):
    """Build a stable key for response satisfaction band cycle detection."""
    return tuple(response_scores[station_idx] for station_idx in active_stations)


def weighted_average_satisfaction(satisfaction, fixed_inputs):
    """Population-weight fixed-assignment satisfaction."""
    weighted_total = np.sum(
        satisfaction[fixed_inputs['covered_mask']]
        * fixed_inputs['elderly_total_by_community'][fixed_inputs['covered_mask']]
    )
    return float(weighted_total / fixed_inputs['covered_elderly'])


def choose_feedback_state(states):
    """Choose the best recorded state if S2 bands repeat."""
    return max(
        states,
        key=lambda state: (
            state['weighted_avg_satisfaction'],
            -max(state['station_utilizations'].values(), default=0.0),
        ),
    )


def solve_s2_feedback(weighted_s3, financial_config, fixed_inputs):
    """Update station response scores after distributed prices change demand."""
    response_scores = {
        station_idx: calc_s2(financial_config['station_utilizations_t2'][station_idx])
        for station_idx in fixed_inputs['active_stations']
    }
    seen_states = {}
    states = []

    for iteration in range(1, MAX_T3_S2_ITERATIONS + 1):
        satisfaction, station_daily_by_service, station_loads, station_utilizations = (
            compute_satisfaction_and_station_visits(
                response_scores,
                weighted_s3,
                financial_config,
                fixed_inputs,
            )
        )
        state = {
            'satisfaction_by_community': satisfaction,
            'station_effective_daily_by_service': station_daily_by_service,
            'station_loads': station_loads,
            'station_utilizations': station_utilizations,
            'station_response_scores': dict(response_scores),
            'weighted_avg_satisfaction': weighted_average_satisfaction(satisfaction, fixed_inputs),
            'iterations': iteration,
            'converged': False,
            'cycle_detected': False,
        }
        states.append(state)

        next_response_scores = {
            station_idx: calc_s2(station_utilizations[station_idx])
            for station_idx in fixed_inputs['active_stations']
        }
        if all(
            abs(next_response_scores[station_idx] - response_scores[station_idx])
            <= SATISFACTION_TOLERANCE
            for station_idx in fixed_inputs['active_stations']
        ):
            state['converged'] = True
            return state

        current_key = response_state_key(response_scores, fixed_inputs['active_stations'])
        previous_state_idx = seen_states.get(current_key)
        if previous_state_idx is not None:
            cycle_state = choose_feedback_state(states[previous_state_idx:])
            cycle_state['cycle_detected'] = True
            return cycle_state
        seen_states[current_key] = len(states) - 1
        response_scores = next_response_scores

    return choose_feedback_state(states)


def compute_capacity_metrics(station_loads, financial_config, fixed_inputs):
    """Audit station loads against attachment and inherited elastic capacities."""
    station_overload_rates = {}
    station_capacity_feasible = {}
    for station_idx in fixed_inputs['active_stations']:
        attachment_capacity = financial_config['station_capacities'][station_idx]
        elastic_limit = financial_config['station_elastic_capacity_limits'][station_idx]
        load = station_loads[station_idx]
        station_overload_rates[station_idx] = max(0.0, load - attachment_capacity) / attachment_capacity
        station_capacity_feasible[station_idx] = load <= elastic_limit + LOAD_TOLERANCE
    return {
        'station_overload_rates': station_overload_rates,
        'station_capacity_feasible': station_capacity_feasible,
        'capacity_feasible': all(station_capacity_feasible.values()),
        'max_overload_rate': float(max(station_overload_rates.values(), default=0.0)),
    }


def compute_station_financials(plan, station_daily_by_service, financial_config, fixed_inputs):
    """Compute station and system financial indicators for one price plan."""
    days_per_year = financial_config['days_per_year']
    direct_costs = np.asarray(financial_config['direct_costs'], dtype=float)
    subsidy_per_visit = financial_config['subsidy_per_visit']
    station_financials = {}

    for station_idx in fixed_inputs['active_stations']:
        daily_by_service = np.asarray(station_daily_by_service[station_idx], dtype=float)
        annual_by_service = daily_by_service * days_per_year
        station_prices = plan['station_prices'][station_idx]
        annual_service_revenue = sum(
            annual_by_service[service_idx] * station_prices[service] / 10000.0
            for service_idx, service in enumerate(fixed_inputs['paid_services'])
        )
        annual_variable_cost = float(np.sum(annual_by_service * direct_costs) / 10000.0)
        daily_paid_visits = float(daily_by_service[:fixed_inputs['paid_service_count']].sum())
        daily_subsidy_request_yuan = daily_paid_visits * subsidy_per_visit
        daily_subsidy_cap_yuan = financial_config['daily_subsidy_caps_yuan'][station_idx]
        daily_subsidy_actual_yuan = min(daily_subsidy_request_yuan, daily_subsidy_cap_yuan)
        annual_subsidy = daily_subsidy_actual_yuan * days_per_year / 10000.0
        annual_fixed_management_cost = (
            financial_config['daily_fixed_costs_yuan'][station_idx] * days_per_year / 10000.0
        )
        annual_depreciation = financial_config['station_annual_depreciation_wan'][station_idx]
        annual_operating_cost = annual_fixed_management_cost + annual_depreciation
        service_profit = annual_service_revenue - annual_variable_cost
        annual_profit = service_profit + annual_subsidy - annual_operating_cost
        profit_margin = (
            annual_profit / annual_operating_cost * 100.0
            if annual_operating_cost > 0 else 0.0
        )

        station_financials[station_idx] = {
            'annual_effective_visits': float(annual_by_service.sum()),
            'effective_daily_by_service': tuple(float(value) for value in daily_by_service),
            'paid_effective_daily_visits': daily_paid_visits,
            'daily_subsidy_request_yuan': daily_subsidy_request_yuan,
            'daily_subsidy_cap_yuan': float(daily_subsidy_cap_yuan),
            'daily_subsidy_actual_yuan': float(daily_subsidy_actual_yuan),
            'annual_service_revenue': float(annual_service_revenue),
            'annual_variable_cost': annual_variable_cost,
            'annual_fixed_management_cost': float(annual_fixed_management_cost),
            'annual_depreciation': float(annual_depreciation),
            'annual_operating_cost': float(annual_operating_cost),
            'annual_subsidy': float(annual_subsidy),
            'annual_total_revenue': float(annual_service_revenue + annual_subsidy),
            'annual_total_cost': float(annual_operating_cost + annual_variable_cost),
            'annual_profit': float(annual_profit),
            'profit_margin': float(profit_margin),
        }

    annual_service_revenue = sum(
        financials['annual_service_revenue']
        for financials in station_financials.values()
    )
    annual_variable_cost = sum(
        financials['annual_variable_cost']
        for financials in station_financials.values()
    )
    annual_subsidy = sum(
        financials['annual_subsidy']
        for financials in station_financials.values()
    )
    annual_fixed_management_cost = sum(
        financials['annual_fixed_management_cost']
        for financials in station_financials.values()
    )
    annual_depreciation = sum(
        financials['annual_depreciation']
        for financials in station_financials.values()
    )
    annual_operating_cost = sum(
        financials['annual_operating_cost']
        for financials in station_financials.values()
    )
    annual_profit = annual_service_revenue + annual_subsidy - annual_variable_cost - annual_operating_cost
    profit_margin = (
        annual_profit / annual_operating_cost * 100.0
        if annual_operating_cost > 0 else 0.0
    )
    system_financials = {
        'annual_effective_visits': float(
            sum(financials['annual_effective_visits'] for financials in station_financials.values())
        ),
        'annual_service_revenue': float(annual_service_revenue),
        'annual_variable_cost': float(annual_variable_cost),
        'annual_subsidy': float(annual_subsidy),
        'annual_fixed_management_cost': float(annual_fixed_management_cost),
        'annual_depreciation': float(annual_depreciation),
        'annual_operating_cost': float(annual_operating_cost),
        'annual_total_revenue': float(annual_service_revenue + annual_subsidy),
        'annual_total_cost': float(annual_operating_cost + annual_variable_cost),
        'annual_profit': float(annual_profit),
        'profit_margin': float(profit_margin),
    }
    return station_financials, system_financials


def evaluate_plan(plan, financial_config, fixed_inputs):
    """Evaluate one distributed station pricing plan."""
    station_s3 = build_station_s3(plan, financial_config, fixed_inputs)
    weighted_s3 = compute_weighted_s3_by_community(station_s3, fixed_inputs)
    feedback_state = solve_s2_feedback(weighted_s3, financial_config, fixed_inputs)
    capacity_metrics = compute_capacity_metrics(
        feedback_state['station_loads'],
        financial_config,
        fixed_inputs,
    )
    station_financials, system_financials = compute_station_financials(
        plan,
        feedback_state['station_effective_daily_by_service'],
        financial_config,
        fixed_inputs,
    )

    return {
        'plan_id': plan['plan_id'],
        'station_template_ids': dict(plan['station_template_ids']),
        'station_uniform_multipliers': dict(plan.get('station_uniform_multipliers', {})),
        'price_plan_tuple': tuple(plan['price_plan_tuple']),
        'station_s3_by_service': {
            station_idx: tuple(float(value) for value in s3_values)
            for station_idx, s3_values in station_s3.items()
        },
        'weighted_s3_by_community': tuple(float(value) for value in weighted_s3),
        'satisfaction_by_community': tuple(
            float(value) for value in feedback_state['satisfaction_by_community']
        ),
        'station_response_scores': dict(feedback_state['station_response_scores']),
        'station_loads': dict(feedback_state['station_loads']),
        'station_utilizations': dict(feedback_state['station_utilizations']),
        'station_financials': station_financials,
        'system_financials': system_financials,
        'weighted_avg_satisfaction': feedback_state['weighted_avg_satisfaction'],
        's2_iterations': feedback_state['iterations'],
        's2_converged': feedback_state['converged'],
        's2_cycle_detected': feedback_state['cycle_detected'],
        **capacity_metrics,
    }


def print_sample_result(result, candidate_data, fixed_inputs):
    """Print a short sample result for the first evaluated distributed plan."""
    station_templates = {
        template['template_id']: template
        for template in candidate_data['station_templates']
    }
    print('【样例方案 0】')
    for station_idx in fixed_inputs['active_stations']:
        template_id = result['station_template_ids'][station_idx]
        template = station_templates[template_id]
        financials = result['station_financials'][station_idx]
        print(
            f"  - 站点 {station_idx}: 模板 {template_id}，{template['label']}，"
            f"负荷 {result['station_loads'][station_idx]:.1f}，"
            f"补贴 {financials['annual_subsidy']:.2f} 万元，"
            f"利润率 {financials['profit_margin']:.2f}%"
        )
    print(f"  全区满意度: {result['weighted_avg_satisfaction']:.4f}")
    print(f"  全区利润率: {result['system_financials']['profit_margin']:.2f}%")
    print(f"  容量弹性可行: {result['capacity_feasible']}\n")


def main():
    print('========== 问题 3.2：分站价格方案评价 ==========\n')
    start_time = time.time()

    candidate_data, financial_config, daily_demand, population = load_required_inputs()
    candidate_plans = candidate_data['candidate_plans']
    fixed_inputs = build_fixed_assignment_inputs(financial_config, daily_demand, population)

    print('【评价参数确认】')
    print(f"候选分站价格方案数: {len(candidate_plans):,}")
    print(f"活动站点数: {len(fixed_inputs['active_stations'])}")
    print(f"付费服务数: {len(fixed_inputs['paid_services'])}")
    print('固定 T2 口径: 站点布局、规模和小区归属不变。')
    print('重新核算口径: 分站 S3、有效人次、站点负荷、S2、补贴和逐站财务。')
    print(
        f"容量口径: 附件容量 × {financial_config['capacity_limit_ratio']:.2f} "
        '作为后续过滤的弹性上限。\n'
    )

    evaluated_results = []
    for plan_idx, plan in enumerate(candidate_plans, start=1):
        evaluated_results.append(evaluate_plan(plan, financial_config, fixed_inputs))
        if plan_idx % 25000 == 0:
            print(f'  已评价 {plan_idx:,} 个分站价格方案...')

    elapsed = time.time() - start_time
    print(f'  已评价 {len(evaluated_results):,} 个分站价格方案。')
    print(f'评价耗时: {elapsed:.2f} 秒\n')

    print_sample_result(evaluated_results[0], candidate_data, fixed_inputs)
    satisfactions = [result['weighted_avg_satisfaction'] for result in evaluated_results]
    margins = [result['system_financials']['profit_margin'] for result in evaluated_results]
    elastic_capacity_feasible_count = sum(
        int(result['capacity_feasible']) for result in evaluated_results
    )
    print('【评价统计】')
    print(f"满意度范围: {min(satisfactions):.4f} ~ {max(satisfactions):.4f}")
    print(f"全区利润率范围: {min(margins):.2f}% ~ {max(margins):.2f}%")
    print(
        f"弹性容量可行方案: {elastic_capacity_feasible_count:,} "
        f"({elastic_capacity_feasible_count / len(evaluated_results) * 100:.2f}%)"
    )
    print(
        f"S2 档位收敛方案: {sum(int(result['s2_converged']) for result in evaluated_results):,}"
    )
    print(
        f"S2 档位循环近似方案: {sum(int(result['s2_cycle_detected']) for result in evaluated_results):,}\n"
    )

    os.makedirs('model', exist_ok=True)
    output_path = Path('model/T3_03_price_evaluation_results.pkl')
    with output_path.open('wb') as output_file:
        pickle.dump(evaluated_results, output_file)

    print('分站价格方案评价完成。')
    print(f'结果已保存至: {output_path}')
    print('准备进入约束过滤阶段。')
    print('================================\n')


if __name__ == '__main__':
    main()
