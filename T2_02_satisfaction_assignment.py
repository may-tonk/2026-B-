import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from T2_02_assignment_optimizer import (
    AssignmentOption,
    optimize_capacity_constrained_assignments,
)


SCALE_NAMES = {1: 'small', 2: 'medium', 3: 'large'}
SCALE_NAMES_ZH = {0: '未建', 1: '小型', 2: '中型', 3: '大型'}
PRICE_SATISFACTION_T2 = 1.0
MAX_ITERATIONS = 30
LOAD_TOLERANCE = 1e-6
SATISFACTION_TOLERANCE = 1e-10
INITIAL_RESPONSE_SEEDS = (1.0, 0.60)
DEFAULT_CAPACITY_OVERLOAD_RATIO = 0.02


def load_data():
    """Load T0, T1, and budget-feasible T2 layout data."""
    layouts_path = Path('model/T2_01_feasible_layouts.pkl')
    config_path = Path('model/T0_01_config.pkl')
    demand_path = Path('model/T1_03_constrained_demand_daily_by_service.npy')
    coverage_path = Path('model/T0_02_coverage_mask.npy')
    population_path = Path('model/T1_01_population_year5.npy')

    if not all(path.exists() for path in [layouts_path, config_path, demand_path, coverage_path, population_path]):
        print('[错误] 找不到前置依赖数据，请确保 T2_01 和 T1 已运行。')
        sys.exit(1)

    with layouts_path.open('rb') as layout_file:
        feasible_layouts = pickle.load(layout_file)
    with config_path.open('rb') as config_file:
        config = pickle.load(config_file)

    daily_demand_by_service = np.load(demand_path)
    coverage_mask = np.load(coverage_path)
    population_year5 = np.load(population_path)
    community_daily_demand = daily_demand_by_service.sum(axis=1)
    elderly_by_community = population_year5.sum(axis=1)
    return feasible_layouts, config, community_daily_demand, elderly_by_community, coverage_mask


def calc_s1(distance):
    """Return distance satisfaction for an in-radius station candidate."""
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


def get_station_capacities(layout, station_specs):
    """Map active layout entries to attachment capacities."""
    return {
        station_idx: float(station_specs[SCALE_NAMES[scale]]['daily_capacity'])
        for station_idx, scale in enumerate(layout)
        if scale > 0
    }


def get_accessible_stations(active_stations, coverage_mask, community_idx):
    """Return active stations inside the 1000m coverage mask."""
    return [
        station_idx
        for station_idx in active_stations
        if coverage_mask[community_idx, station_idx]
    ]


def build_s1_matrix(distance_matrix, coverage_mask, active_stations):
    """Precompute S1 scores for active reachable station pairs."""
    num_communities = distance_matrix.shape[0]
    s1_matrix = np.zeros((num_communities, num_communities), dtype=float)
    for community_idx in range(num_communities):
        for station_idx in active_stations:
            if coverage_mask[community_idx, station_idx]:
                s1_matrix[community_idx, station_idx] = calc_s1(
                    distance_matrix[community_idx, station_idx]
                )
    return s1_matrix


def compose_satisfaction(s1_score, s2_score, weights):
    """Compose T2 satisfaction from distance, response, and baseline price scores."""
    return (
        weights['distance'] * s1_score
        + weights['response'] * s2_score
        + weights['price'] * PRICE_SATISFACTION_T2
    )


def build_assignment_options(
    active_stations,
    coverage_mask,
    distance_matrix,
    s1_matrix,
    response_scores,
    community_daily_demand,
    weights,
):
    """Build only tied highest-score options for one assignment step."""
    num_communities = coverage_mask.shape[0]
    options_by_community = {}

    for community_idx in range(num_communities):
        accessible = get_accessible_stations(active_stations, coverage_mask, community_idx)
        reachable_options = []
        for station_idx in accessible:
            satisfaction = compose_satisfaction(
                s1_matrix[community_idx, station_idx],
                response_scores[station_idx],
                weights,
            )
            reachable_options.append(
                AssignmentOption(
                    community_idx=community_idx,
                    station_idx=station_idx,
                    satisfaction=float(satisfaction),
                    effective_load=float(community_daily_demand[community_idx] * satisfaction),
                    distance=float(distance_matrix[community_idx, station_idx]),
                )
            )
        if not reachable_options:
            options_by_community[community_idx] = []
            continue

        highest_satisfaction = max(option.satisfaction for option in reachable_options)
        options_by_community[community_idx] = [
            option
            for option in reachable_options
            if option.satisfaction >= highest_satisfaction - SATISFACTION_TOLERANCE
        ]
    return options_by_community


def build_iteration_state(
    optimized_assignment,
    station_capacities,
    elderly_by_community,
):
    """Package an assignment iteration for convergence and cycle handling."""
    community_satisfactions = np.asarray(optimized_assignment['community_satisfactions'], dtype=float)
    effective_daily_visits = np.asarray(optimized_assignment['community_effective_daily_visits'], dtype=float)
    station_loads = optimized_assignment['station_loads']
    station_utilizations = optimized_assignment['station_utilizations']
    covered_mask = community_satisfactions > 0
    covered_elderly = float(np.asarray(elderly_by_community, dtype=float)[covered_mask].sum())
    weighted_satisfaction = float(
        np.sum(community_satisfactions[covered_mask] * np.asarray(elderly_by_community, dtype=float)[covered_mask])
        / covered_elderly
    ) if covered_elderly > 0 else 0.0
    return {
        'assignments': dict(optimized_assignment['assignments']),
        'community_satisfactions': community_satisfactions.copy(),
        'community_effective_daily_visits': effective_daily_visits.copy(),
        'station_loads': dict(station_loads),
        'station_utilizations': station_utilizations,
        'max_utilization': max(station_utilizations.values(), default=0.0),
        'avg_satisfaction': weighted_satisfaction,
        'assignment_objective': optimized_assignment['objective'],
    }


def is_capacity_feasible(state, station_capacities, capacity_limit_ratio=1.0):
    """Check effective Scheme B loads against the selected capacity limit."""
    return all(
        load <= station_capacities[station_idx] * capacity_limit_ratio + LOAD_TOLERANCE
        for station_idx, load in state['station_loads'].items()
    )


def validates_highest_satisfaction_rule(
    state,
    active_stations,
    coverage_mask,
    distance_matrix,
    s1_matrix,
    community_daily_demand,
    weights,
):
    """Check final assignments against highest scores under final utilizations."""
    final_response_scores = {
        station_idx: calc_s2(state['station_utilizations'][station_idx])
        for station_idx in active_stations
    }
    highest_options = build_assignment_options(
        active_stations,
        coverage_mask,
        distance_matrix,
        s1_matrix,
        final_response_scores,
        community_daily_demand,
        weights,
    )
    expected_covered = {
        community_idx
        for community_idx, options in highest_options.items()
        if options
    }
    if set(state['assignments']) != expected_covered:
        return False

    for community_idx, station_idx in state['assignments'].items():
        assigned_option = next(
            (
                option
                for option in highest_options[community_idx]
                if option.station_idx == station_idx
            ),
            None,
        )
        if assigned_option is None:
            return False
        if abs(
            assigned_option.satisfaction
            - state['community_satisfactions'][community_idx]
        ) > SATISFACTION_TOLERANCE:
            return False
        if abs(
            assigned_option.effective_load
            - state['community_effective_daily_visits'][community_idx]
        ) > LOAD_TOLERANCE:
            return False
    return True


def select_cycle_state(cycle_states, station_capacities, capacity_limit_ratio):
    """Choose the best feasible state if assignment iteration enters a cycle."""
    feasible_states = [
        state for state in cycle_states
        if is_capacity_feasible(state, station_capacities, capacity_limit_ratio)
    ]
    if not feasible_states:
        return None
    return max(
        feasible_states,
        key=lambda state: (
            state['avg_satisfaction'],
            -state['max_utilization'],
        ),
    )


def solve_layout_assignment(
    layout,
    build_cost,
    config,
    community_daily_demand,
    elderly_by_community,
    coverage_mask,
    capacity_overload_ratio=DEFAULT_CAPACITY_OVERLOAD_RATIO,
):
    """Solve one layout with capacity-constrained Scheme B satisfaction feedback."""
    active_stations = [station_idx for station_idx, scale in enumerate(layout) if scale > 0]
    if not active_stations:
        return None

    station_capacities = get_station_capacities(layout, config['station_specs'])
    capacity_limit_ratio = 1.0 + capacity_overload_ratio
    distance_matrix = config['distance_matrix']
    weights = config['satisfaction_weights']
    s1_matrix = build_s1_matrix(distance_matrix, coverage_mask, active_stations)

    candidate_trajectories = []
    for initial_response_score in INITIAL_RESPONSE_SEEDS:
        previous_loads = {station_idx: 0.0 for station_idx in active_stations}
        response_scores = {station_idx: initial_response_score for station_idx in active_stations}
        seen_feedback_states = {}
        iteration_states = []
        final_state = None
        converged = False
        cycle_detected = False

        for _ in range(MAX_ITERATIONS):
            options_by_community = build_assignment_options(
                active_stations,
                coverage_mask,
                distance_matrix,
                s1_matrix,
                response_scores,
                community_daily_demand,
                weights,
            )
            optimized_assignment = optimize_capacity_constrained_assignments(
                options_by_community,
                station_capacities,
                elderly_by_community,
                len(community_daily_demand),
                capacity_limit_ratio,
            )
            if optimized_assignment is None:
                break

            state = build_iteration_state(
                optimized_assignment,
                station_capacities,
                elderly_by_community,
            )
            iteration_states.append(state)

            assignment_stable = (
                len(iteration_states) > 1
                and state['assignments'] == iteration_states[-2]['assignments']
            )
            load_change = max(
                abs(state['station_loads'][station_idx] - previous_loads[station_idx])
                for station_idx in active_stations
            )
            next_response_scores = {
                station_idx: calc_s2(state['station_utilizations'][station_idx])
                for station_idx in active_stations
            }
            response_stable = next_response_scores == response_scores
            if assignment_stable and response_stable and load_change <= LOAD_TOLERANCE:
                final_state = state
                converged = True
                break

            feedback_key = (
                tuple(state['assignments'].get(idx, -1) for idx in range(len(community_daily_demand))),
                tuple((station_idx, response_scores[station_idx]) for station_idx in active_stations),
            )
            previous_state_idx = seen_feedback_states.get(feedback_key)
            if previous_state_idx is not None:
                cycle_detected = True
                final_state = select_cycle_state(
                    iteration_states[previous_state_idx:],
                    station_capacities,
                    capacity_limit_ratio,
                )
                break
            seen_feedback_states[feedback_key] = len(iteration_states) - 1

            previous_loads = state['station_loads']
            response_scores = next_response_scores
        else:
            final_state = iteration_states[-1] if iteration_states else None

        highest_satisfaction_valid = (
            final_state is not None
            and validates_highest_satisfaction_rule(
                final_state,
                active_stations,
                coverage_mask,
                distance_matrix,
                s1_matrix,
                community_daily_demand,
                weights,
            )
        )
        if (
            final_state is not None
            and is_capacity_feasible(final_state, station_capacities, capacity_limit_ratio)
            and highest_satisfaction_valid
        ):
            candidate_trajectories.append({
                'state': final_state,
                'iterations': len(iteration_states),
                'converged': converged,
                'cycle_detected': cycle_detected,
            })

    if not candidate_trajectories:
        return None

    chosen_trajectory = max(
        candidate_trajectories,
        key=lambda trajectory: (
            trajectory['state']['avg_satisfaction'],
            -trajectory['state']['max_utilization'],
            trajectory['converged'],
        ),
    )
    final_state = chosen_trajectory['state']

    return {
        'layout': list(layout),
        'build_cost': float(build_cost),
        'active_stations': active_stations,
        'assignments': final_state['assignments'],
        'community_satisfactions': final_state['community_satisfactions'],
        'community_effective_daily_visits': final_state['community_effective_daily_visits'],
        'station_loads': final_state['station_loads'],
        'station_capacities': station_capacities,
        'station_utilizations': final_state['station_utilizations'],
        'iterations': chosen_trajectory['iterations'],
        'converged': chosen_trajectory['converged'],
        'cycle_detected': chosen_trajectory['cycle_detected'],
        'highest_satisfaction_validated': True,
        'capacity_overload_ratio': float(capacity_overload_ratio),
        'capacity_limit_ratio': float(capacity_limit_ratio),
    }


def main():
    print('========== 问题 2.2：满意度驱动小区归属与容量校验（方案B修订版） ==========')
    start_time = time.time()

    feasible_layouts, config, community_daily_demand, elderly_by_community, coverage_mask = load_data()
    capacity_overload_ratio = DEFAULT_CAPACITY_OVERLOAD_RATIO
    valid_solutions = []
    invalid_capacity_or_cycle_count = 0
    total_iterations = 0
    converged_count = 0
    cycle_count = 0

    print('\n【输入与规则确认】')
    print(f"预算可行布局数: {len(feasible_layouts):,}")
    print(f"消费约束后全区日需求: {community_daily_demand.sum():.2f} 次/日")
    print('容量口径: 站点实际有效服务人次 = 小区日需求 x 综合满意度')
    print(
        '附件容量: 小型 '
        f"{config['station_specs']['small']['daily_capacity']:.0f}, 中型 "
        f"{config['station_specs']['medium']['daily_capacity']:.0f}, 大型 "
        f"{config['station_specs']['large']['daily_capacity']:.0f} 人次/日"
    )
    print('归属规则: 每个被覆盖小区只保留综合满意度最高的可达站点')
    print('并列处理: 仅在满意度并列最高的站点之间做容量可行择优')
    print(
        f'容量规则: 采用 {capacity_overload_ratio * 100:.0f}% 小幅弹性上限，'
        '超过弹性容量的布局直接删除'
    )
    print('输出说明: 仍保留附件容量、实际负荷和相对附件容量的利用率\n')
    print('开始对预算可行布局执行确定性归属迭代...')

    for layout_idx, layout_info in enumerate(feasible_layouts):
        solution = solve_layout_assignment(
            layout_info['layout'],
            layout_info['build_cost'],
            config,
            community_daily_demand,
            elderly_by_community,
            coverage_mask,
            capacity_overload_ratio,
        )
        if solution is None:
            invalid_capacity_or_cycle_count += 1
        else:
            solution['layout_idx'] = layout_idx
            valid_solutions.append(solution)
            total_iterations += solution['iterations']
            converged_count += int(solution['converged'])
            cycle_count += int(solution['cycle_detected'])

        if layout_idx > 0 and layout_idx % 3000 == 0:
            print(f'已处理 {layout_idx:,} 个布局，当前保留容量可行方案 {len(valid_solutions):,} 个...')

    os.makedirs('model', exist_ok=True)
    output_path = Path('model/T2_02_feasible_solutions.pkl')
    with output_path.open('wb') as solution_file:
        pickle.dump(valid_solutions, solution_file)

    elapsed = time.time() - start_time
    avg_iterations = total_iterations / len(valid_solutions) if valid_solutions else 0.0

    print('\n【统计汇总】')
    print(f'处理完成的预算可行布局: {len(feasible_layouts):,}')
    print(f'  - 容量可行方案: {len(valid_solutions):,}')
    print(f'  - 超容量或无可行稳定状态方案: {invalid_capacity_or_cycle_count:,}')
    print(f'  - 归属稳定收敛方案: {converged_count:,}')
    print(f'  - 触发循环状态选择方案: {cycle_count:,}')
    print(f'容量可行方案平均迭代次数: {avg_iterations:.2f}')
    print(f'已保存: {output_path}')

    if valid_solutions:
        sample = valid_solutions[0]
        print('\n【样例方案】')
        print(f"建设成本: {sample['build_cost']:.1f} 万元")
        for station_idx in sample['active_stations']:
            scale = sample['layout'][station_idx]
            load = sample['station_loads'][station_idx]
            capacity = sample['station_capacities'][station_idx]
            utilization = sample['station_utilizations'][station_idx]
            print(
                f"  - 站点 {config['communities'][station_idx]} ({SCALE_NAMES_ZH[scale]}): "
                f'有效负荷 {load:.2f} / 容量 {capacity:.0f} '
                f'(利用率 {utilization * 100:.1f}%)'
            )
        print(f"唯一归属小区数: {len(sample['assignments'])}")

    print(f'\n所有布局处理总耗时: {elapsed:.2f} 秒')
    print('================================================\n')


if __name__ == '__main__':
    main()
