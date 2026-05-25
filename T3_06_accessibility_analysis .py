import os
import pickle
import sys
from pathlib import Path

import numpy as np


K_KEYS = ['healthy', 'semi_disabled', 'disabled']
K_NAMES = ['自理老人', '半失能老人', '失能老人']
DAYS_PER_MONTH = 30
MONTHS_PER_YEAR = 12


def load_pickle(path):
    with Path(path).open('rb') as file:
        return pickle.load(file)


def load_data():
    required = [
        'model/T0_01_config.pkl',
        'model/T1_01_population_year5.npy',
        'model/T1_03_constrained_demand_monthly_by_elderly_type.pkl',
        'model/T2_04_best_solution.pkl',
        'model/T3_05_optimal_price_plan.pkl',
        'model/T3_01_financial_model_config.pkl',
    ]
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        print('[ERROR] Missing required files:')
        for path in missing:
            print(f'  - {path}')
        sys.exit(1)

    config = load_pickle('model/T0_01_config.pkl')
    pop_year5 = np.load('model/T1_01_population_year5.npy')
    constrained_monthly = load_pickle('model/T1_03_constrained_demand_monthly_by_elderly_type.pkl')
    t2_best = load_pickle('model/T2_04_best_solution.pkl')
    t3_plan = load_pickle('model/T3_05_optimal_price_plan.pkl')
    financial_config = load_pickle('model/T3_01_financial_model_config.pkl')
    return config, pop_year5, constrained_monthly, t2_best, t3_plan, financial_config


def distance_score(distance, rules):
    for segment in rules['segments']:
        if segment.get('min_distance', -1) < distance <= segment.get('max_distance', float('inf')):
            return segment['score']
    return 0.0


def grade_spatial(value):
    if value >= 95:
        return '优'
    if value >= 80:
        return '良'
    if value >= 60:
        return '中'
    return '差'


def grade_economic(value):
    if value <= 50:
        return '优'
    if value <= 80:
        return '良'
    if value <= 100:
        return '中'
    return '差'


def grade_service(value):
    if value >= 0.85:
        return '优'
    if value >= 0.80:
        return '良'
    if value >= 0.70:
        return '中'
    return '差'


def weighted_mean(values, weights):
    weights = np.asarray(weights, dtype=float)
    if weights.sum() <= 0:
        return 0.0
    values = np.asarray(values, dtype=float)
    return float(np.sum(values * weights) / np.sum(weights))


def station_price_vector(t3_plan, station_idx, paid_services):
    station_prices = t3_plan.get('station_prices')
    if not station_prices or station_idx not in station_prices:
        raise KeyError(
            'T3_05 optimal plan does not contain station_prices. '
            'Please rerun T3_05 after the latest patch.'
        )
    return np.asarray([station_prices[station_idx][service] for service in paid_services], dtype=float)


def allocate_station_subsidy_to_types(constrained_monthly, assignments, satisfaction, station_idx, station_fin):
    type_effective_visits = []
    for k_key in K_KEYS:
        demand = constrained_monthly[k_key]
        effective = 0.0
        for community_idx in range(demand.shape[0]):
            if assignments.get(community_idx, -1) == station_idx:
                effective += float(np.sum(demand[community_idx, :5]) * satisfaction[community_idx])
        type_effective_visits.append(effective)

    type_effective_visits = np.asarray(type_effective_visits, dtype=float)
    total_effective = float(type_effective_visits.sum())
    monthly_actual_subsidy = station_fin['annual_subsidy'] * 10000.0 / MONTHS_PER_YEAR
    if total_effective <= 0:
        return np.zeros(3)
    return monthly_actual_subsidy * type_effective_visits / total_effective


def main():
    print('========== 问题 3.3：三类老人服务可及性分析 ==========\n')
    config, pop_year5, constrained_monthly, t2_best, t3_plan, financial_config = load_data()

    communities = config['communities']
    paid_services = list(financial_config['paid_services'])
    base_prices = np.asarray(financial_config['service_prices_base'][:len(paid_services)], dtype=float)
    assignments = t2_best['assignments']
    active_stations = list(financial_config['active_stations'])
    dist_matrix = config['distance_matrix']
    income = np.asarray(config['monthly_income_per_capita'], dtype=float)
    cap_ratios = np.asarray(config['spending_cap_ratios'], dtype=float)
    demand_rates = np.asarray(config['demand_rate_matrix'], dtype=float)
    satisfaction = np.asarray(t3_plan['satisfaction_by_community'], dtype=float)
    weighted_s3 = np.asarray(t3_plan['weighted_s3_by_community'], dtype=float)
    station_response_scores = t3_plan['station_response_scores']

    pop_totals = pop_year5.sum(axis=0)
    total_pop = float(pop_totals.sum())
    metrics = {}

    station_subsidy_by_type = np.zeros(3)
    for station_idx in active_stations:
        station_subsidy_by_type += allocate_station_subsidy_to_types(
            constrained_monthly,
            assignments,
            satisfaction,
            station_idx,
            t3_plan['station_financials'][station_idx],
        )

    for k, k_key in enumerate(K_KEYS):
        pop_k = np.asarray(pop_year5[:, k], dtype=float)
        demand_k = np.asarray(constrained_monthly[k_key], dtype=float)
        covered = np.asarray([assignments.get(i, -1) != -1 for i in range(len(communities))], dtype=bool)
        covered_pop = float(np.sum(pop_k[covered]))
        total_k = float(pop_totals[k])

        dist_segments = np.zeros(4)
        distances = []
        s1_values = []
        s2_values = []
        s3_values = []
        satisfaction_values = []

        gross_opt_cost = 0.0
        gross_base_cost = 0.0
        for i in range(len(communities)):
            station_idx = assignments.get(i, -1)
            if station_idx == -1:
                continue
            distance = float(dist_matrix[i, station_idx])
            if distance <= 300:
                dist_segments[0] += pop_k[i]
            if distance <= 500:
                dist_segments[1] += pop_k[i]
            if distance <= 650:
                dist_segments[2] += pop_k[i]
            if distance <= 1000:
                dist_segments[3] += pop_k[i]

            distances.append(distance)
            s1_values.append(distance_score(distance, config['satisfaction_rules']['distance']))
            s2_values.append(float(station_response_scores[station_idx]))
            s3_values.append(float(weighted_s3[i]))
            satisfaction_values.append(float(satisfaction[i]))

            paid_demand = demand_k[i, :len(paid_services)]
            station_prices = station_price_vector(t3_plan, station_idx, paid_services)
            gross_opt_cost += float(np.dot(paid_demand, station_prices))
            gross_base_cost += float(np.dot(paid_demand, base_prices))

        avg_income = weighted_mean(income, pop_k)
        consumption_cap = avg_income * float(cap_ratios[k])
        opt_payment_pc = gross_opt_cost / total_k if total_k > 0 else 0.0
        base_payment_pc = gross_base_cost / total_k if total_k > 0 else 0.0
        subsidy_pc = station_subsidy_by_type[k] / total_k if total_k > 0 else 0.0
        burden_base = base_payment_pc / consumption_cap * 100.0 if consumption_cap > 0 else 0.0
        burden_opt = opt_payment_pc / consumption_cap * 100.0 if consumption_cap > 0 else 0.0

        theoretical_demand = float(np.sum(demand_rates[k]))
        actual_monthly_demand = float(np.sum(demand_k) / total_k) if total_k > 0 else 0.0
        release_rate = actual_monthly_demand / theoretical_demand * 100.0 if theoretical_demand > 0 else 0.0

        metrics[k_key] = {
            'name': K_NAMES[k],
            'population': total_k,
            'population_ratio': total_k / total_pop * 100.0 if total_pop > 0 else 0.0,
            'coverage_rate': covered_pop / total_k * 100.0 if total_k > 0 else 0.0,
            'distance_coverage_rates': dist_segments / total_k * 100.0 if total_k > 0 else np.zeros(4),
            'average_distance': weighted_mean(
                [dist_matrix[i, assignments[i]] for i in range(len(communities)) if assignments.get(i, -1) != -1],
                [pop_k[i] for i in range(len(communities)) if assignments.get(i, -1) != -1],
            ),
            'average_income': avg_income,
            'monthly_consumption_cap': consumption_cap,
            'theoretical_monthly_demand_pc': theoretical_demand,
            'actual_monthly_demand_pc': actual_monthly_demand,
            'demand_release_rate': release_rate,
            'base_payment_pc': base_payment_pc,
            'optimal_payment_pc': opt_payment_pc,
            'payment_reduction_pc': base_payment_pc - opt_payment_pc,
            'fiscal_subsidy_pc': subsidy_pc,
            'burden_base': burden_base,
            'burden_optimal': burden_opt,
            'satisfaction_avg': weighted_mean(
                [satisfaction[i] for i in range(len(communities)) if assignments.get(i, -1) != -1],
                [pop_k[i] for i in range(len(communities)) if assignments.get(i, -1) != -1],
            ),
            's1_avg': weighted_mean(
                [distance_score(dist_matrix[i, assignments[i]], config['satisfaction_rules']['distance'])
                 for i in range(len(communities)) if assignments.get(i, -1) != -1],
                [pop_k[i] for i in range(len(communities)) if assignments.get(i, -1) != -1],
            ),
            's2_avg': weighted_mean(
                [station_response_scores[assignments[i]]
                 for i in range(len(communities)) if assignments.get(i, -1) != -1],
                [pop_k[i] for i in range(len(communities)) if assignments.get(i, -1) != -1],
            ),
            's3_avg': weighted_mean(
                [weighted_s3[i] for i in range(len(communities)) if assignments.get(i, -1) != -1],
                [pop_k[i] for i in range(len(communities)) if assignments.get(i, -1) != -1],
            ),
        }

    print('【分站真实定价口径】')
    for station_idx in active_stations:
        station_name = communities[station_idx]
        multiplier = t3_plan.get('station_uniform_multipliers', {}).get(station_idx)
        multiplier_text = f'{multiplier:.3f}' if multiplier is not None else '—'
        print(f'  {station_name}站：倍率 {multiplier_text}，价格 {[t3_plan["station_prices"][station_idx][s] for s in paid_services]}')
    print()

    print('【三类老人经济可及性】')
    print(f"{'类型':<10} {'基准月支付':>10} {'最优月支付':>10} {'支付降幅':>10} {'最优负担率':>10} {'人均财政补贴':>12}")
    for k_key in K_KEYS:
        m = metrics[k_key]
        print(
            f"{m['name']:<10} "
            f"{m['base_payment_pc']:>10.2f} "
            f"{m['optimal_payment_pc']:>10.2f} "
            f"{m['payment_reduction_pc']:>10.2f} "
            f"{m['burden_optimal']:>9.2f}% "
            f"{m['fiscal_subsidy_pc']:>12.2f}"
        )
    print('\n说明：最优月支付按小区归属站点的真实分站价格计算；财政补贴为服务站实际获得补贴按有效付费人次分摊的强度指标，不视为直接现金发放给老人。\n')

    print('【三类老人综合可及性评价】')
    print(f"{'类型':<10} {'空间':>6} {'经济':>6} {'服务':>6} {'满意度':>10}")
    for k_key in K_KEYS:
        m = metrics[k_key]
        print(
            f"{m['name']:<10} "
            f"{grade_spatial(m['coverage_rate']):>6} "
            f"{grade_economic(m['burden_optimal']):>6} "
            f"{grade_service(m['satisfaction_avg']):>6} "
            f"{m['satisfaction_avg']:>10.4f}"
        )

    output = {
        'metrics': metrics,
        'metadata': {
            'pricing_source': 'model/T3_05_optimal_price_plan.pkl',
            'economic_accessibility_method': (
                'Actual station prices by community assignment; subsidy reported as fiscal support intensity.'
            ),
            'active_stations': active_stations,
            'paid_services': paid_services,
            'station_prices': t3_plan['station_prices'],
            'station_uniform_multipliers': t3_plan.get('station_uniform_multipliers', {}),
            'capacity_limit_ratio': financial_config['capacity_limit_ratio'],
        },
    }

    os.makedirs('model', exist_ok=True)
    with open('model/T3_06_accessibility_metrics.pkl', 'wb') as file:
        pickle.dump(output, file)

    print('\n可及性分析已保存至: model/T3_06_accessibility_metrics.pkl')
    print('================================')


if __name__ == '__main__':
    main()
