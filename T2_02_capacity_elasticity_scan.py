import pickle
from pathlib import Path

import numpy as np

from T2_02_satisfaction_assignment import load_data, solve_layout_assignment


OVERLOAD_RATIOS = (0.00, 0.01, 0.02, 0.03, 0.05)
OUTPUT_PATH = Path('model/T2_02_capacity_elasticity_scan.pkl')
REPORT_PATH = Path('T2_02_capacity_elasticity_scan_report.md')


def capacity_excess_metrics(solution):
    excesses = {
        station_idx: max(
            0.0,
            solution['station_loads'][station_idx] - solution['station_capacities'][station_idx],
        )
        for station_idx in solution['active_stations']
    }
    overload_rates = {
        station_idx: excesses[station_idx] / solution['station_capacities'][station_idx]
        for station_idx in solution['active_stations']
    }
    return {
        'station_excesses': excesses,
        'station_overload_rates': overload_rates,
        'total_excess': float(sum(excesses.values())),
        'max_excess': float(max(excesses.values(), default=0.0)),
        'max_overload_rate': float(max(overload_rates.values(), default=0.0)),
    }


def covered_metrics(solution, elderly_by_community):
    covered_communities = sorted(solution['assignments'])
    elderly = np.asarray(elderly_by_community, dtype=float)
    covered_elderly = float(elderly[covered_communities].sum())
    total_elderly = float(elderly.sum())
    community_satisfactions = np.asarray(solution['community_satisfactions'], dtype=float)
    avg_satisfaction = (
        float(np.sum(community_satisfactions[covered_communities] * elderly[covered_communities]) / covered_elderly)
        if covered_elderly > 0
        else 0.0
    )
    return {
        'covered_communities': covered_communities,
        'covered_elderly': covered_elderly,
        'coverage_rate': covered_elderly / total_elderly if total_elderly > 0 else 0.0,
        'avg_satisfaction': avg_satisfaction,
    }


def enrich_solution(solution, elderly_by_community):
    enriched = dict(solution)
    enriched.update(covered_metrics(solution, elderly_by_community))
    enriched.update(capacity_excess_metrics(solution))
    return enriched


def result_key(solution):
    return (
        solution['coverage_rate'],
        solution['avg_satisfaction'],
        -solution['max_overload_rate'],
        -solution['total_excess'],
        -solution['build_cost'],
    )


def layout_text(config, layout):
    scale_names = {1: '小型', 2: '中型', 3: '大型'}
    return ', '.join(
        f"{config['communities'][station_idx]}-{scale_names[scale]}"
        for station_idx, scale in enumerate(layout)
        if scale > 0
    )


def station_excess_text(config, solution):
    rows = []
    for station_idx in solution['active_stations']:
        excess = solution['station_excesses'][station_idx]
        overload_rate = solution['station_overload_rates'][station_idx] * 100
        if excess <= 0:
            continue
        rows.append(
            f"{config['communities'][station_idx]} "
            f"{solution['station_loads'][station_idx]:.1f}/"
            f"{solution['station_capacities'][station_idx]:.0f} "
            f"(超 {excess:.1f}, {overload_rate:.2f}%)"
        )
    return '；'.join(rows) if rows else '无'


def scan_ratio(
    ratio,
    feasible_layouts,
    config,
    community_daily_demand,
    elderly_by_community,
    coverage_mask,
):
    valid_solutions = []
    for layout_info in feasible_layouts:
        solution = solve_layout_assignment(
            layout_info['layout'],
            layout_info['build_cost'],
            config,
            community_daily_demand,
            elderly_by_community,
            coverage_mask,
            capacity_overload_ratio=ratio,
        )
        if solution is not None:
            valid_solutions.append(enrich_solution(solution, elderly_by_community))

    valid_solutions.sort(key=result_key, reverse=True)
    best_solution = valid_solutions[0] if valid_solutions else None
    return {
        'overload_ratio': ratio,
        'valid_solution_count': len(valid_solutions),
        'full_coverage_count': sum(
            1
            for solution in valid_solutions
            if np.isclose(solution['coverage_rate'], 1.0)
        ),
        'best_solution': best_solution,
        'top_solutions': valid_solutions[:10],
    }


def build_report(scan_results, config):
    lines = [
        '# T2 严格最高满意度归属容量弹性扫描',
        '',
        '## 1. 扫描口径',
        '',
        '- 保持 1000 米有效服务半径。',
        '- 保持“仅在有效范围内选择综合满意度最高站点”。',
        '- 仅改变容量上限：`有效负荷 <= 附件容量 * (1 + 弹性比例)`。',
        '- 弹性比例扫描值：0%、1%、2%、3%、5%。',
        '',
        '## 2. 扫描结果',
        '',
        '| 容量弹性 | 可行方案数 | 全覆盖方案数 | 最优覆盖率 | 最优平均满意度 | 最优最大超载率 | 最优布局 |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | --- |',
    ]
    for result in scan_results:
        best = result['best_solution']
        if best is None:
            lines.append(
                f"| {result['overload_ratio'] * 100:.0f}% | {result['valid_solution_count']:,} | "
                f"{result['full_coverage_count']:,} | — | — | — | 无 |"
            )
            continue
        lines.append(
            f"| {result['overload_ratio'] * 100:.0f}% | {result['valid_solution_count']:,} | "
            f"{result['full_coverage_count']:,} | {best['coverage_rate'] * 100:.2f}% | "
            f"{best['avg_satisfaction']:.4f} | {best['max_overload_rate'] * 100:.2f}% | "
            f"{layout_text(config, best['layout'])} |"
        )

    first_feasible = next(
        (result for result in scan_results if result['best_solution'] is not None),
        None,
    )
    if first_feasible is not None:
        best = first_feasible['best_solution']
        lines.extend([
            '',
            '## 3. 最小可行弹性下的最优方案',
            '',
            f"- 容量弹性：{first_feasible['overload_ratio'] * 100:.0f}%。",
            f"- 布局：{layout_text(config, best['layout'])}。",
            f"- 覆盖率：{best['coverage_rate'] * 100:.2f}%。",
            f"- 平均满意度：{best['avg_satisfaction']:.4f}。",
            f"- 原附件容量上的实际超载：{station_excess_text(config, best)}。",
            '',
        ])
    return '\n'.join(lines)


def main():
    print('========== T2 严格最高满意度归属容量弹性扫描 ==========')
    feasible_layouts, config, community_daily_demand, elderly_by_community, coverage_mask = load_data()
    scan_results = []
    for ratio in OVERLOAD_RATIOS:
        print(f'\n开始扫描容量弹性 {ratio * 100:.0f}% ...')
        result = scan_ratio(
            ratio,
            feasible_layouts,
            config,
            community_daily_demand,
            elderly_by_community,
            coverage_mask,
        )
        scan_results.append(result)
        best = result['best_solution']
        if best is None:
            print(f"  可行方案 {result['valid_solution_count']:,}，全覆盖方案 {result['full_coverage_count']:,}")
            continue
        print(
            f"  可行方案 {result['valid_solution_count']:,}，全覆盖方案 {result['full_coverage_count']:,}，"
            f"最优覆盖率 {best['coverage_rate'] * 100:.2f}%，平均满意度 {best['avg_satisfaction']:.4f}"
        )
        print(
            f"  最优布局 {layout_text(config, best['layout'])}，"
            f"原附件容量最大超载率 {best['max_overload_rate'] * 100:.2f}%"
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open('wb') as output_file:
        pickle.dump({'scan_results': scan_results}, output_file)
    REPORT_PATH.write_text(build_report(scan_results, config), encoding='utf-8')

    print(f'\n扫描数据已保存: {OUTPUT_PATH}')
    print(f'扫描报告已保存: {REPORT_PATH}')
    print('====================================================')


if __name__ == '__main__':
    main()
