import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np


SCALE_NAMES = {1: 'small', 2: 'medium', 3: 'large'}
SCALE_NAMES_ZH = {0: '未建', 1: '小型', 2: '中型', 3: '大型'}
PRICE_SATISFACTION_T2 = 1.0


def load_evaluation_data():
    """Load deterministic Scheme B T2 solutions and their shared inputs."""
    solutions_path = Path('model/T2_02_feasible_solutions.pkl')
    population_path = Path('model/T1_01_population_year5.npy')
    demand_path = Path('model/T1_03_constrained_demand_daily_by_service.npy')
    config_path = Path('model/T0_01_config.pkl')
    coverage_path = Path('model/T0_02_coverage_mask.npy')

    required_paths = [
        solutions_path,
        population_path,
        demand_path,
        config_path,
        coverage_path,
    ]
    if not all(path.exists() for path in required_paths):
        print('[错误] 找不到前置依赖数据，请确保 T0/T1/T2.2 均已执行成功。')
        sys.exit(1)

    with solutions_path.open('rb') as solution_file:
        feasible_solutions = pickle.load(solution_file)
    with config_path.open('rb') as config_file:
        config = pickle.load(config_file)

    population_year5 = np.load(population_path)
    daily_demand_by_service = np.load(demand_path)
    coverage_mask = np.load(coverage_path)
    return feasible_solutions, config, population_year5, daily_demand_by_service, coverage_mask


def calc_s1(distance):
    """Return distance satisfaction."""
    if distance <= 300:
        return 1.00
    if distance <= 500:
        return 0.90
    if distance <= 650:
        return 0.75
    if distance <= 1000:
        return 0.60
    return 0.0


def calc_s2(utilization):
    """Return response satisfaction from attachment-capacity utilization."""
    if utilization <= 0.60:
        return 1.00
    if utilization <= 0.75:
        return 0.93
    if utilization <= 0.85:
        return 0.85
    if utilization <= 0.95:
        return 0.72
    return 0.60


def get_station_spec(config, scale):
    """Return attachment-backed station parameters for a scale code."""
    return config['station_specs'][SCALE_NAMES[scale]]


def compute_coverage_metrics(assignments, elderly_by_community):
    """Compute coverage by the communities with a final single-station assignment."""
    covered_communities = sorted(assignments)
    covered_elderly = float(np.asarray(elderly_by_community, dtype=float)[covered_communities].sum())
    total_elderly = float(np.asarray(elderly_by_community, dtype=float).sum())
    coverage_rate = covered_elderly / total_elderly if total_elderly > 0 else 0.0
    return covered_communities, covered_elderly, coverage_rate


def compute_weighted_satisfaction(community_satisfactions, covered_communities, elderly_by_community):
    """Population-weight covered-community satisfaction."""
    if not covered_communities:
        return 0.0
    satisfaction = np.asarray(community_satisfactions, dtype=float)
    elderly = np.asarray(elderly_by_community, dtype=float)
    covered_elderly = elderly[covered_communities].sum()
    return float(
        np.sum(satisfaction[covered_communities] * elderly[covered_communities]) / covered_elderly
    ) if covered_elderly > 0 else 0.0


def compute_annual_financials(
    daily_demand_by_service,
    assignments,
    community_satisfactions,
    layout,
    active_stations,
    build_cost_wan,
    config,
):
    """Compute annual T2 visits, revenue, costs, depreciation, and profit."""
    service_prices = np.asarray(config['service_prices'], dtype=float)
    direct_costs = np.asarray(config['direct_costs'], dtype=float)
    days_per_year = config['time_constants']['days_per_year']
    depreciation_years = config['time_constants']['depreciation_years']

    annual_effective_visits = 0.0
    annual_revenue = 0.0
    annual_direct_cost = 0.0
    station_effective_daily_by_service = {
        station_idx: np.zeros_like(service_prices, dtype=float)
        for station_idx in active_stations
    }

    for community_idx, station_idx in assignments.items():
        effective_daily_by_service = (
            np.asarray(daily_demand_by_service[community_idx], dtype=float)
            * float(community_satisfactions[community_idx])
        )
        station_effective_daily_by_service[station_idx] += effective_daily_by_service
        annual_effective_visits += float(effective_daily_by_service.sum() * days_per_year)
        annual_revenue += float(np.sum(effective_daily_by_service * service_prices) * days_per_year)
        annual_direct_cost += float(np.sum(effective_daily_by_service * direct_costs) * days_per_year)

    annual_fixed_cost = sum(
        get_station_spec(config, layout[station_idx])['daily_fixed_cost']
        for station_idx in active_stations
    ) * days_per_year

    annual_revenue_wan = annual_revenue / 10000.0
    annual_direct_cost_wan = annual_direct_cost / 10000.0
    annual_fixed_cost_wan = annual_fixed_cost / 10000.0
    annual_depreciation_wan = build_cost_wan / depreciation_years
    annual_profit_wan = (
        annual_revenue_wan
        - annual_direct_cost_wan
        - annual_fixed_cost_wan
        - annual_depreciation_wan
    )
    return {
        'annual_effective_visits': annual_effective_visits,
        'annual_revenue': annual_revenue_wan,
        'annual_direct_cost': annual_direct_cost_wan,
        'annual_fixed_cost': annual_fixed_cost_wan,
        'annual_depreciation': annual_depreciation_wan,
        'annual_profit': annual_profit_wan,
        'station_effective_daily_by_service': station_effective_daily_by_service,
    }


def compute_capacity_metrics(solution):
    """Expose attachment-capacity excess under an elastic feasibility limit."""
    station_excesses = {
        station_idx: max(
            0.0,
            solution['station_loads'][station_idx] - solution['station_capacities'][station_idx],
        )
        for station_idx in solution['active_stations']
    }
    station_overload_rates = {
        station_idx: station_excesses[station_idx] / solution['station_capacities'][station_idx]
        for station_idx in solution['active_stations']
    }
    return {
        'station_excesses': station_excesses,
        'station_overload_rates': station_overload_rates,
        'total_capacity_excess': float(sum(station_excesses.values())),
        'max_capacity_excess': float(max(station_excesses.values(), default=0.0)),
        'max_overload_rate': float(max(station_overload_rates.values(), default=0.0)),
    }


def build_satisfaction_details(solution, config, coverage_mask):
    """Expose per-community S1/S2/S3 scores for tables, charts, and checks."""
    details = {}
    distance_matrix = config['distance_matrix']
    weights = config['satisfaction_weights']
    for community_idx, station_idx in solution['assignments'].items():
        if not coverage_mask[community_idx, station_idx]:
            raise ValueError('T2_02 归属结果包含超出覆盖半径的站点')
        utilization = solution['station_utilizations'][station_idx]
        s1_score = calc_s1(distance_matrix[community_idx, station_idx])
        s2_score = calc_s2(utilization)
        composite_score = (
            weights['distance'] * s1_score
            + weights['response'] * s2_score
            + weights['price'] * PRICE_SATISFACTION_T2
        )
        details[community_idx] = {
            'station_idx': station_idx,
            'distance': float(distance_matrix[community_idx, station_idx]),
            's1': s1_score,
            's2': s2_score,
            's3': PRICE_SATISFACTION_T2,
            'utilization': float(utilization),
            'composite': float(composite_score),
        }
    return details


def evaluate_solution(solution, config, elderly_by_community, daily_demand_by_service, coverage_mask):
    """Evaluate one deterministic Scheme B solution."""
    assignments = solution['assignments']
    covered_communities, covered_elderly, coverage_rate = compute_coverage_metrics(
        assignments,
        elderly_by_community,
    )
    avg_satisfaction = compute_weighted_satisfaction(
        solution['community_satisfactions'],
        covered_communities,
        elderly_by_community,
    )
    financials = compute_annual_financials(
        daily_demand_by_service,
        assignments,
        solution['community_satisfactions'],
        solution['layout'],
        solution['active_stations'],
        solution['build_cost'],
        config,
    )
    capacity_metrics = compute_capacity_metrics(solution)
    satisfaction_details = build_satisfaction_details(solution, config, coverage_mask)

    evaluated = dict(solution)
    evaluated.update(financials)
    evaluated.update(capacity_metrics)
    evaluated.update({
        'covered_communities': covered_communities,
        'covered_elderly': covered_elderly,
        'coverage_rate': coverage_rate,
        'avg_satisfaction': avg_satisfaction,
        'satisfaction_details': satisfaction_details,
    })
    return evaluated


def sort_key(solution):
    """Layer coverage, satisfaction, capacity excess, profit, then build cost."""
    return (
        solution['coverage_rate'],
        solution['avg_satisfaction'],
        -solution['max_overload_rate'],
        solution['annual_profit'],
        -solution['build_cost'],
    )


def describe_scale_counts(layout):
    scale_counts = {1: 0, 2: 0, 3: 0}
    for scale in layout:
        if scale > 0:
            scale_counts[scale] += 1
    return f"小型 {scale_counts[1]} 个, 中型 {scale_counts[2]} 个, 大型 {scale_counts[3]} 个"


def print_top_solutions(evaluated_solutions):
    print('【排名前 10 的方案汇总】\n')
    for idx, solution in enumerate(evaluated_solutions[:10]):
        title = f"排名 {solution['rank']}（最优方案）:" if idx == 0 else f"排名 {solution['rank']}:"
        profit_margin = (
            solution['annual_profit'] / solution['annual_revenue'] * 100
        ) if solution['annual_revenue'] > 0 else 0.0
        print(title)
        print(f"├─ 站点配置: {solution['layout']}")
        print(f"├─ 建设成本: {solution['build_cost']:.1f} 万元")
        print(f"├─ 站点数:   {describe_scale_counts(solution['layout'])}")
        print(f"├─ 覆盖率:   {solution['coverage_rate'] * 100:.2f}%")
        print(f"├─ 平均满意度: {solution['avg_satisfaction']:.4f}")
        if idx == 0:
            print(f"├─ 被覆盖老人数: {solution['covered_elderly']:.0f} 人")
            print(f"├─ 年有效人次: {solution['annual_effective_visits']:,.0f} 次")
            print(f"├─ 年营收:     {solution['annual_revenue']:.2f} 万元")
            print(f"├─ 年直接支出: {solution['annual_direct_cost']:.2f} 万元")
            print(f"├─ 年固定成本: {solution['annual_fixed_cost']:.2f} 万元")
            print(f"├─ 年折旧成本: {solution['annual_depreciation']:.2f} 万元")
            print(f"├─ 年度利润:   {solution['annual_profit']:.2f} 万元")
            print(f"└─ 利润率:     {profit_margin:.2f}%\n")
        else:
            print(f"└─ 年度利润:   {solution['annual_profit']:.2f} 万元\n")


def print_best_solution(best_solution, config, elderly_by_community, daily_demand_by_service):
    comm_names = config['communities']
    print('【最优方案详细信息】\n')
    print('站点布局:')
    for station_idx, scale in enumerate(best_solution['layout']):
        if scale == 0:
            print(f'  小区{comm_names[station_idx]}: 不建')
            continue
        capacity = best_solution['station_capacities'][station_idx]
        build_cost = get_station_spec(config, scale)['build_cost']
        print(
            f'  小区{comm_names[station_idx]}: {SCALE_NAMES_ZH[scale]}站 '
            f'(日容量{capacity:.0f}, 建设成本{build_cost:g}万)'
        )
    print(f"总建设成本: {best_solution['build_cost']:.1f} 万元\n")

    print('服务覆盖与满意度:')
    print(
        f"{'小区':<4}| {'老人数':<6}| {'日需求':<7}| {'归属站点':<7}| "
        f"{'距离':<6}| {'S1':<4}| {'ρ':<6}| {'S2':<4}| {'S':<5}| {'有效人次/日':<10}"
    )
    print('-' * 98)
    for community_idx, community_name in enumerate(comm_names):
        elderly_count = elderly_by_community[community_idx]
        daily_demand = daily_demand_by_service[community_idx].sum()
        detail = best_solution['satisfaction_details'].get(community_idx)
        if detail is None:
            print(
                f"{community_name:<4} | {elderly_count:<7.0f} | {daily_demand:<8.1f} | {'—':<8} | "
                f"{'—':<7} | {'—':<4} | {'—':<6} | {'—':<4} | {'—':<5} | {'—':<10}"
            )
            continue
        effective_visits = best_solution['community_effective_daily_visits'][community_idx]
        station_name = comm_names[detail['station_idx']]
        print(
            f"{community_name:<4} | {elderly_count:<7.0f} | {daily_demand:<8.1f} | {station_name:<8} | "
            f"{detail['distance']:<7.0f} | {detail['s1']:<4.2f} | {detail['utilization']:<6.2f} | "
            f"{detail['s2']:<4.2f} | {detail['composite']:<5.2f} | {effective_visits:<10.1f}"
        )

    print('\n站点负荷详情:')
    print(f"{'站点':<4}| {'规模':<4}| {'附件容量':<8}| {'有效负荷':<10}| {'超载率':<8}| {'利用率':<8}| {'S2':<4}")
    print('-' * 70)
    for station_idx in best_solution['active_stations']:
        scale = SCALE_NAMES_ZH[best_solution['layout'][station_idx]]
        capacity = best_solution['station_capacities'][station_idx]
        load = best_solution['station_loads'][station_idx]
        utilization = best_solution['station_utilizations'][station_idx]
        overload_rate = best_solution['station_overload_rates'][station_idx]
        print(
            f"{comm_names[station_idx]:<4} | {scale:<4} | {capacity:<9.0f} | {load:<11.1f} | "
            f"{overload_rate * 100:<8.2f}% | {utilization * 100:<8.1f}% | {calc_s2(utilization):<4.2f}"
        )

    print('\n全区综合指标:')
    print(
        f"  覆盖老人总数: {best_solution['covered_elderly']:.0f} 人"
        f"（占 {best_solution['coverage_rate'] * 100:.1f}%）"
    )
    print(f"  平均满意度（老人数量加权）: {best_solution['avg_satisfaction']:.4f}")
    print('  年度综合指标:')
    print(f"    - 年有效服务总人次: {best_solution['annual_effective_visits']:,.0f} 次")
    print(f"    - 年营收:         {best_solution['annual_revenue']:.2f} 万元")
    print(f"    - 年直接支出:     {best_solution['annual_direct_cost']:.2f} 万元")
    print(f"    - 年固定成本:     {best_solution['annual_fixed_cost']:.2f} 万元")
    print(f"    - 年折旧成本:     {best_solution['annual_depreciation']:.2f} 万元")
    print(f"    - 年度利润:       {best_solution['annual_profit']:.2f} 万元")


def build_summary_statistics(evaluated_solutions):
    return {
        'total_evaluated': len(evaluated_solutions),
        'full_coverage_count': sum(
            1 for solution in evaluated_solutions
            if np.isclose(solution['coverage_rate'], 1.0)
        ),
        'coverage_at_least_95_count': sum(
            1 for solution in evaluated_solutions
            if solution['coverage_rate'] >= 0.95
        ),
        'coverage_below_90_count': sum(
            1 for solution in evaluated_solutions
            if solution['coverage_rate'] < 0.90
        ),
    }


def main():
    print('========== 问题 2.3：覆盖率、满意度、利润评价与分层排序（方案B修订版） ==========\n')
    start_time = time.time()

    feasible_solutions, config, population_year5, daily_demand_by_service, coverage_mask = load_evaluation_data()
    elderly_by_community = population_year5.sum(axis=1)
    total_elderly = elderly_by_community.sum()

    print('【输入数据确认】')
    print(f'容量可行方案总数: {len(feasible_solutions):,}')
    print(f'第 5 年总老人数: {total_elderly:.0f} 人')
    print(f'消费约束后日需求矩阵: {daily_demand_by_service.shape}')
    print('评价口径: 使用 T2_02 唯一归属与方案B实际有效服务人次\n')
    if feasible_solutions:
        print(
            f"容量弹性: {feasible_solutions[0].get('capacity_overload_ratio', 0.0) * 100:.0f}% "
            '小幅弹性上限\n'
        )
    print('正在计算评价指标...')

    evaluated_solutions = [
        evaluate_solution(
            solution,
            config,
            elderly_by_community,
            daily_demand_by_service,
            coverage_mask,
        )
        for solution in feasible_solutions
    ]
    evaluated_solutions.sort(key=sort_key, reverse=True)
    for rank_idx, solution in enumerate(evaluated_solutions, start=1):
        solution['rank'] = rank_idx

    print_top_solutions(evaluated_solutions)
    summary_statistics = build_summary_statistics(evaluated_solutions)
    print('【分层排序统计】')
    print(f"容量可行评价方案: {summary_statistics['total_evaluated']:,}")
    print(f"  - 全覆盖方案: {summary_statistics['full_coverage_count']:,}")
    print(f"  - 覆盖率 >= 95% 方案: {summary_statistics['coverage_at_least_95_count']:,}")
    print(f"  - 覆盖率 < 90% 方案: {summary_statistics['coverage_below_90_count']:,}\n")

    best_solution = evaluated_solutions[0] if evaluated_solutions else None
    if best_solution is not None:
        print_best_solution(best_solution, config, elderly_by_community, daily_demand_by_service)

    save_data = {
        'all_solutions': evaluated_solutions,
        'best_solution_index': 0 if best_solution is not None else None,
        'summary_statistics': summary_statistics,
    }
    os.makedirs('model', exist_ok=True)
    evaluation_path = Path('model/T2_03_evaluation_results.pkl')
    with evaluation_path.open('wb') as evaluation_file:
        pickle.dump(save_data, evaluation_file)

    if best_solution is not None:
        best_solution_path = Path('model/T2_04_best_solution.pkl')
        with best_solution_path.open('wb') as best_file:
            pickle.dump(best_solution, best_file)

        legacy_path = Path('model/T2_03_best_solution.pkl')
        with legacy_path.open('wb') as legacy_file:
            pickle.dump(best_solution, legacy_file)
    else:
        best_solution_path = None

    elapsed = time.time() - start_time
    print(f'\n评价与排序完成！总耗时 {elapsed:.2f} 秒。')
    print(f'结果已保存至: {evaluation_path}')
    if best_solution_path is not None:
        print(f'完整最优方案已保存至: {best_solution_path}')
        print('兼容副本已保存至: model/T2_03_best_solution.pkl')
    else:
        print('本轮未产生最优方案；请先处理 T2_02 严格归属规则下的可行性问题。')
    print('================================')


if __name__ == '__main__':
    main()
