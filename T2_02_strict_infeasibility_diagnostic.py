import pickle
from collections import Counter
from pathlib import Path

import numpy as np

from T2_02_assignment_optimizer import optimize_capacity_constrained_assignments
from T2_02_satisfaction_assignment import (
    INITIAL_RESPONSE_SEEDS,
    LOAD_TOLERANCE,
    MAX_ITERATIONS,
    build_assignment_options,
    build_iteration_state,
    build_s1_matrix,
    calc_s2,
    get_station_capacities,
    load_data,
)


OUTPUT_PATH = Path('model/T2_02_strict_infeasibility_diagnostics.pkl')
REPORT_PATH = Path('T2_02_strict_infeasibility_report.md')


def assignment_tuple(assignments, num_communities):
    return tuple(assignments.get(community_idx, -1) for community_idx in range(num_communities))


def overload_metrics(station_loads, station_capacities):
    overloads = {
        station_idx: max(0.0, station_loads[station_idx] - station_capacities[station_idx])
        for station_idx in station_capacities
    }
    total_overload = float(sum(overloads.values()))
    max_overload = float(max(overloads.values(), default=0.0))
    max_utilization = float(
        max(
            station_loads[station_idx] / station_capacities[station_idx]
            for station_idx in station_capacities
        )
    ) if station_capacities else 0.0
    return overloads, total_overload, max_overload, max_utilization


def package_attempt(assignments, chosen_options, station_loads, station_capacities, num_communities):
    overloads, total_overload, max_overload, max_utilization = overload_metrics(
        station_loads,
        station_capacities,
    )
    return {
        'assignments': dict(assignments),
        'chosen_options': dict(chosen_options),
        'station_loads': dict(station_loads),
        'station_capacities': dict(station_capacities),
        'station_overloads': overloads,
        'total_overload': total_overload,
        'max_overload': max_overload,
        'max_utilization': max_utilization,
        'assignment_tuple': assignment_tuple(assignments, num_communities),
    }


def attempt_key(attempt):
    return (
        attempt['total_overload'],
        attempt['max_overload'],
        attempt['max_utilization'],
        attempt['assignment_tuple'],
    )


def find_minimum_overload_attempt(options_by_community, station_capacities, num_communities):
    """Find the least overloaded assignment among strict highest-score ties."""
    covered_communities = [
        community_idx
        for community_idx, options in options_by_community.items()
        if options
    ]
    station_loads = {station_idx: 0.0 for station_idx in station_capacities}
    assignments = {}
    chosen_options = {}

    search_order = sorted(
        covered_communities,
        key=lambda community_idx: (
            len(options_by_community[community_idx]),
            -max(option.effective_load for option in options_by_community[community_idx]),
            community_idx,
        ),
    )
    best = None

    def current_total_overload():
        return sum(
            max(0.0, station_loads[station_idx] - station_capacities[station_idx])
            for station_idx in station_capacities
        )

    def search(search_pos):
        nonlocal best

        if best is not None and current_total_overload() > best['total_overload'] + LOAD_TOLERANCE:
            return
        if search_pos == len(search_order):
            candidate = package_attempt(
                assignments,
                chosen_options,
                station_loads,
                station_capacities,
                num_communities,
            )
            if best is None or attempt_key(candidate) < attempt_key(best):
                best = candidate
            return

        community_idx = search_order[search_pos]
        ranked_options = sorted(
            options_by_community[community_idx],
            key=lambda option: (
                max(
                    0.0,
                    station_loads[option.station_idx]
                    + option.effective_load
                    - station_capacities[option.station_idx],
                ),
                station_loads[option.station_idx] / station_capacities[option.station_idx],
                option.distance,
                option.station_idx,
            ),
        )
        for option in ranked_options:
            previous_load = station_loads[option.station_idx]
            station_loads[option.station_idx] = previous_load + option.effective_load
            assignments[community_idx] = option.station_idx
            chosen_options[community_idx] = option
            search(search_pos + 1)
            station_loads[option.station_idx] = previous_load
            assignments.pop(community_idx)
            chosen_options.pop(community_idx)

    search(0)
    return best


def covered_metrics(options_by_community, elderly_by_community):
    covered_communities = sorted(
        community_idx
        for community_idx, options in options_by_community.items()
        if options
    )
    covered_elderly = float(np.asarray(elderly_by_community, dtype=float)[covered_communities].sum())
    total_elderly = float(np.asarray(elderly_by_community, dtype=float).sum())
    return covered_communities, covered_elderly, covered_elderly / total_elderly


def sources_by_station(attempt):
    sources = {station_idx: [] for station_idx in attempt['station_capacities']}
    for community_idx, station_idx in attempt['assignments'].items():
        option = attempt['chosen_options'][community_idx]
        sources[station_idx].append({
            'community_idx': community_idx,
            'effective_load': float(option.effective_load),
            'satisfaction': float(option.satisfaction),
            'distance': float(option.distance),
        })
    return sources


def failure_record(
    layout_idx,
    layout,
    build_cost,
    seed,
    failure_iteration,
    response_scores,
    options_by_community,
    attempt,
    elderly_by_community,
):
    covered_communities, covered_elderly, coverage_rate = covered_metrics(
        options_by_community,
        elderly_by_community,
    )
    attempt['sources_by_station'] = sources_by_station(attempt)
    return {
        'layout_idx': layout_idx,
        'layout': list(layout),
        'build_cost': float(build_cost),
        'seed': float(seed),
        'failure_iteration': int(failure_iteration),
        'response_scores': dict(response_scores),
        'covered_communities': covered_communities,
        'covered_community_count': len(covered_communities),
        'covered_elderly': covered_elderly,
        'coverage_rate': coverage_rate,
        'attempt': attempt,
    }


def diagnose_layout_trajectory(
    layout_idx,
    layout_info,
    seed,
    config,
    community_daily_demand,
    elderly_by_community,
    coverage_mask,
):
    layout = layout_info['layout']
    active_stations = [station_idx for station_idx, scale in enumerate(layout) if scale > 0]
    if not active_stations:
        return None

    station_capacities = get_station_capacities(layout, config['station_specs'])
    distance_matrix = config['distance_matrix']
    weights = config['satisfaction_weights']
    s1_matrix = build_s1_matrix(distance_matrix, coverage_mask, active_stations)
    response_scores = {station_idx: seed for station_idx in active_stations}
    previous_loads = {station_idx: 0.0 for station_idx in active_stations}
    previous_assignments = None
    seen_states = set()

    for iteration in range(1, MAX_ITERATIONS + 1):
        options_by_community = build_assignment_options(
            active_stations,
            coverage_mask,
            distance_matrix,
            s1_matrix,
            response_scores,
            community_daily_demand,
            weights,
        )
        optimized = optimize_capacity_constrained_assignments(
            options_by_community,
            station_capacities,
            elderly_by_community,
            len(community_daily_demand),
        )
        if optimized is None:
            attempt = find_minimum_overload_attempt(
                options_by_community,
                station_capacities,
                len(community_daily_demand),
            )
            return failure_record(
                layout_idx,
                layout,
                layout_info['build_cost'],
                seed,
                iteration,
                response_scores,
                options_by_community,
                attempt,
                elderly_by_community,
            )

        state = build_iteration_state(optimized, station_capacities, elderly_by_community)
        next_response_scores = {
            station_idx: calc_s2(state['station_utilizations'][station_idx])
            for station_idx in active_stations
        }
        load_change = max(
            abs(state['station_loads'][station_idx] - previous_loads[station_idx])
            for station_idx in active_stations
        )
        if (
            previous_assignments == state['assignments']
            and next_response_scores == response_scores
            and load_change <= LOAD_TOLERANCE
        ):
            return None

        feedback_key = (
            assignment_tuple(state['assignments'], len(community_daily_demand)),
            tuple((station_idx, response_scores[station_idx]) for station_idx in active_stations),
        )
        if feedback_key in seen_states:
            return None
        seen_states.add(feedback_key)
        previous_assignments = state['assignments']
        previous_loads = state['station_loads']
        response_scores = next_response_scores

    return None


def closest_failure_key(record):
    attempt = record['attempt']
    return (
        -record['coverage_rate'],
        attempt['total_overload'],
        attempt['max_overload'],
        -record['failure_iteration'],
        record['build_cost'],
        record['layout_idx'],
        record['seed'],
    )


def station_label(config, station_idx):
    return config['communities'][station_idx]


def format_layout(config, layout):
    scale_names = {1: '小型', 2: '中型', 3: '大型'}
    return ', '.join(
        f"{config['communities'][station_idx]}-{scale_names[scale]}"
        for station_idx, scale in enumerate(layout)
        if scale > 0
    )


def format_overloaded_stations(config, record):
    attempt = record['attempt']
    station_rows = []
    for station_idx, overload in attempt['station_overloads'].items():
        if overload <= LOAD_TOLERANCE:
            continue
        station_rows.append(
            f"{station_label(config, station_idx)} "
            f"{attempt['station_loads'][station_idx]:.1f}/"
            f"{attempt['station_capacities'][station_idx]:.0f} "
            f"(超 {overload:.1f})"
        )
    return '；'.join(station_rows) if station_rows else '无'


def build_report(summary, closest_records, config):
    lines = [
        '# T2 严格最高满意度归属无解诊断',
        '',
        '## 1. 诊断口径',
        '',
        '- 仅保留 1000 米内可达站点。',
        '- 每个小区只保留综合满意度并列最高的站点候选。',
        '- 若严格最高满意度候选没有容量可行组合，则在这些候选内寻找最小超载尝试，用于定位冲突，不作为可行解。',
        '',
        '## 2. 总体结论',
        '',
        f"- 预算可行布局数：{summary['budget_layout_count']:,}。",
        f"- 诊断到的严格最高满意度容量冲突轨迹：{summary['failure_record_count']:,}。",
        f"- 初始 `S2=1.00` 第一轮容量冲突轨迹：{summary['seed_failure_counts'].get(1.0, 0):,}。",
        f"- 初始 `S2=0.60` 反馈轨迹容量冲突：{summary['seed_failure_counts'].get(0.6, 0):,}。",
        f"- 最高诊断覆盖率：{summary['max_coverage_rate'] * 100:.2f}%。",
        '',
        '## 3. 最接近可行的严格失败轨迹',
        '',
        '| 排名 | 初始 S2 | 失败轮次 | 诊断覆盖率 | 布局 | 最小总超载 | 最大单站超载 | 超载站点 |',
        '| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |',
    ]
    for rank, record in enumerate(closest_records, start=1):
        attempt = record['attempt']
        lines.append(
            f"| {rank} | {record['seed']:.2f} | {record['failure_iteration']} | "
            f"{record['coverage_rate'] * 100:.2f}% | {format_layout(config, record['layout'])} | "
            f"{attempt['total_overload']:.1f} | {attempt['max_overload']:.1f} | "
            f"{format_overloaded_stations(config, record)} |"
        )

    if closest_records:
        record = closest_records[0]
        attempt = record['attempt']
        lines.extend([
            '',
            '## 4. 最接近轨迹的站点来源',
            '',
            f"- 布局：{format_layout(config, record['layout'])}。",
            f"- 失败轮次：第 {record['failure_iteration']} 轮。",
            f"- 该轮最高满意度归属诊断覆盖率：{record['coverage_rate'] * 100:.2f}%。",
            '',
            '| 站点 | 容量 | 最小超载尝试负荷 | 超载量 | 最高满意度归属来源 |',
            '| --- | ---: | ---: | ---: | --- |',
        ])
        for station_idx in sorted(attempt['station_capacities']):
            sources = attempt['sources_by_station'][station_idx]
            source_text = '，'.join(
                f"{station_label(config, source['community_idx'])}"
                f"({source['effective_load']:.1f}, S={source['satisfaction']:.2f})"
                for source in sorted(sources, key=lambda source: source['community_idx'])
            ) or '无'
            lines.append(
                f"| {station_label(config, station_idx)} | "
                f"{attempt['station_capacities'][station_idx]:.0f} | "
                f"{attempt['station_loads'][station_idx]:.1f} | "
                f"{attempt['station_overloads'][station_idx]:.1f} | {source_text} |"
            )

    lines.extend([
        '',
        '## 5. 解释',
        '',
        '最小超载尝试仍只在“并列最高满意度”候选中搜索。若这里仍有正超载量，说明当前响应满意度状态下，不存在既满足最高满意度选择又满足附件容量的单一归属组合。',
        '',
    ])
    return '\n'.join(lines)


def main():
    print('========== T2 严格最高满意度归属无解诊断 ==========')
    feasible_layouts, config, community_daily_demand, elderly_by_community, coverage_mask = load_data()
    failure_records = []

    for layout_idx, layout_info in enumerate(feasible_layouts):
        for seed in INITIAL_RESPONSE_SEEDS:
            record = diagnose_layout_trajectory(
                layout_idx,
                layout_info,
                seed,
                config,
                community_daily_demand,
                elderly_by_community,
                coverage_mask,
            )
            if record is not None:
                failure_records.append(record)
        if layout_idx > 0 and layout_idx % 3000 == 0:
            print(f'已诊断 {layout_idx:,} 个预算可行布局，记录失败轨迹 {len(failure_records):,} 条...')

    seed_failure_counts = Counter(record['seed'] for record in failure_records)
    failure_iteration_counts = Counter(record['failure_iteration'] for record in failure_records)
    max_coverage_rate = max((record['coverage_rate'] for record in failure_records), default=0.0)
    closest_records = sorted(failure_records, key=closest_failure_key)[:10]
    summary = {
        'budget_layout_count': len(feasible_layouts),
        'failure_record_count': len(failure_records),
        'seed_failure_counts': dict(seed_failure_counts),
        'failure_iteration_counts': dict(failure_iteration_counts),
        'max_coverage_rate': max_coverage_rate,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open('wb') as output_file:
        pickle.dump({
            'summary': summary,
            'closest_records': closest_records,
            'failure_records': failure_records,
        }, output_file)
    REPORT_PATH.write_text(build_report(summary, closest_records, config), encoding='utf-8')

    print('\n【总体统计】')
    print(f"预算可行布局: {summary['budget_layout_count']:,}")
    print(f"失败诊断轨迹: {summary['failure_record_count']:,}")
    print(f"按初始 S2 分布: {summary['seed_failure_counts']}")
    print(f"按失败轮次分布: {summary['failure_iteration_counts']}")
    print(f"最高诊断覆盖率: {summary['max_coverage_rate'] * 100:.2f}%")
    print('\n【最接近可行的严格失败轨迹】')
    for rank, record in enumerate(closest_records[:5], start=1):
        attempt = record['attempt']
        print(
            f"{rank}. seed={record['seed']:.2f}, 第{record['failure_iteration']}轮, "
            f"覆盖率={record['coverage_rate'] * 100:.2f}%, "
            f"总超载={attempt['total_overload']:.1f}, 最大超载={attempt['max_overload']:.1f}, "
            f"布局={format_layout(config, record['layout'])}"
        )
        print(f"   超载站点: {format_overloaded_stations(config, record)}")
    print(f'\n诊断数据已保存: {OUTPUT_PATH}')
    print(f'诊断报告已保存: {REPORT_PATH}')
    print('==============================================')


if __name__ == '__main__':
    main()
