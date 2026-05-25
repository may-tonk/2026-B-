import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

SCALE_NAMES = {0: '不建', 1: '小型', 2: '中型', 3: '大型'}
SCALE_KEYS = {1: 'small', 2: 'medium', 3: 'large'}


def load_data():
    """Load revised T2 evaluation data and shared plotting inputs."""
    evaluation_path = Path('model/T2_03_evaluation_results.pkl')
    best_solution_path = Path('model/T2_04_best_solution.pkl')
    elasticity_scan_path = Path('model/T2_02_capacity_elasticity_scan.pkl')
    config_path = Path('model/T0_01_config.pkl')
    population_path = Path('model/T1_01_population_year5.npy')
    demand_path = Path('model/T1_03_constrained_demand_daily_by_service.npy')

    required_paths = [
        evaluation_path,
        best_solution_path,
        elasticity_scan_path,
        config_path,
        population_path,
        demand_path,
    ]
    if not all(path.exists() for path in required_paths):
        print('[错误] 缺少前置数据文件，请确保 T2_03 与 T1 阶段脚本已运行。')
        sys.exit(1)

    with evaluation_path.open('rb') as evaluation_file:
        evaluation_data = pickle.load(evaluation_file)
    if (
        evaluation_data.get('best_solution_index') is None
        or not evaluation_data.get('all_solutions')
    ):
        print('[错误] T2_03 未产生可供 T2_04 输出的最优方案，请先处理严格归属规则下的可行性问题。')
        sys.exit(1)
    with best_solution_path.open('rb') as best_file:
        best_solution = pickle.load(best_file)
    with elasticity_scan_path.open('rb') as scan_file:
        elasticity_scan_data = pickle.load(scan_file)
    with config_path.open('rb') as config_file:
        config = pickle.load(config_file)

    population_year5 = np.load(population_path)
    daily_demand_by_service = np.load(demand_path)
    return (
        evaluation_data['all_solutions'],
        best_solution,
        elasticity_scan_data['scan_results'],
        config,
        population_year5,
        daily_demand_by_service,
    )


def station_spec(config, scale):
    """Return T0 attachment-backed station parameters for a scale code."""
    return config['station_specs'][SCALE_KEYS[scale]]


def capacity_status(load, capacity, capacity_limit_ratio):
    if load <= capacity + 1e-6:
        return '未超附件容量'
    overload_rate = (load - capacity) / capacity * 100
    if load <= capacity * capacity_limit_ratio + 1e-6:
        return f'弹性内超载{overload_rate:.2f}\\%'
    return f'超弹性上限{overload_rate:.2f}\\%'


def satisfaction_value(best_solution, community_idx):
    return float(best_solution['community_satisfactions'][community_idx])


def community_station_name(best_solution, config, community_idx):
    station_idx = best_solution['assignments'].get(community_idx)
    return config['communities'][station_idx] if station_idx is not None else '—'


def assigned_communities_for_station(best_solution, config, station_idx):
    comm_names = config['communities']
    return [
        comm_names[community_idx]
        for community_idx, assigned_station in sorted(best_solution['assignments'].items())
        if assigned_station == station_idx
    ]


def compute_station_financials(best_solution, config):
    """Compute station-level finances from saved station effective service pools."""
    service_prices = np.asarray(config['service_prices'], dtype=float)
    direct_costs = np.asarray(config['direct_costs'], dtype=float)
    days_per_year = config['time_constants']['days_per_year']
    depreciation_years = config['time_constants']['depreciation_years']
    financials = {}

    for station_idx in best_solution['active_stations']:
        scale = best_solution['layout'][station_idx]
        spec = station_spec(config, scale)
        effective_daily_by_service = np.asarray(
            best_solution['station_effective_daily_by_service'][station_idx],
            dtype=float,
        )
        annual_revenue = float(np.sum(effective_daily_by_service * service_prices) * days_per_year / 10000.0)
        annual_direct_cost = float(np.sum(effective_daily_by_service * direct_costs) * days_per_year / 10000.0)
        annual_fixed_cost = float(spec['daily_fixed_cost'] * days_per_year / 10000.0)
        annual_depreciation = float(spec['build_cost'] / depreciation_years)
        financials[station_idx] = {
            'effective_daily_by_service': effective_daily_by_service,
            'annual_effective_visits': float(effective_daily_by_service.sum() * days_per_year),
            'annual_revenue': annual_revenue,
            'annual_direct_cost': annual_direct_cost,
            'annual_fixed_cost': annual_fixed_cost,
            'annual_depreciation': annual_depreciation,
            'annual_profit': (
                annual_revenue
                - annual_direct_cost
                - annual_fixed_cost
                - annual_depreciation
            ),
        }
    return financials


def write_tables(best_solution, config, population_year5, daily_demand_by_service, output_dir):
    """Generate revised six LaTeX tables for deterministic Scheme B outputs."""
    comm_names = config['communities']
    elderly_by_community = population_year5.sum(axis=1)
    community_daily_demand = daily_demand_by_service.sum(axis=1)
    station_financials = compute_station_financials(best_solution, config)

    layout_rows = []
    total_build_cost = 0.0
    total_capacity = 0.0
    capacity_limit_ratio = best_solution.get('capacity_limit_ratio', 1.0)
    for community_idx, scale in enumerate(best_solution['layout']):
        if scale == 0:
            layout_rows.append(f"{comm_names[community_idx]} & 不建 & 0 & — & — \\\\")
            continue
        spec = station_spec(config, scale)
        capacity = best_solution['station_capacities'][community_idx]
        assigned = ','.join(assigned_communities_for_station(best_solution, config, community_idx))
        total_build_cost += spec['build_cost']
        total_capacity += capacity
        layout_rows.append(
            f"{comm_names[community_idx]} & {SCALE_NAMES[scale]} & {spec['build_cost']:.1f} & "
            f"{capacity:.0f} & {assigned} \\\\"
        )
    table1 = rf"""\begin{{table}}[htbp]
\centering
\caption{{问题2.1-2.3：最优服务站布局方案}}
\begin{{tabular}}{{|c|c|c|c|c|}}
\hline
小区 & 建站规模 & 建设成本(万元) & 附件容量(人次/日) & 归属小区 \\
\hline
{chr(10).join(layout_rows)}
\hline
合计 & — & {total_build_cost:.1f} & {total_capacity:.0f} & — \\
\hline
\end{{tabular}}

\vspace{{0.2cm}}
\begin{{tabular}}{{|c|c|}}
\hline
评价指标 & 数值 \\
\hline
站点总数 & {len(best_solution['active_stations'])} 个 \\
总建设成本 & {best_solution['build_cost']:.1f} 万元 \\
覆盖率 & {best_solution['coverage_rate'] * 100:.2f}\% \\
平均满意度 & {best_solution['avg_satisfaction']:.4f} \\
年有效服务人次 & {best_solution['annual_effective_visits']:.0f} \\
\hline
\end{{tabular}}
\end{{table}}
"""

    coverage_rows = []
    satisfaction_rows = []
    for community_idx, community_name in enumerate(comm_names):
        detail = best_solution['satisfaction_details'].get(community_idx)
        elderly = elderly_by_community[community_idx]
        daily_demand = community_daily_demand[community_idx]
        if detail is None:
            coverage_rows.append(
                f"{community_name} & {elderly:.0f} & {daily_demand:.1f} & 否 & — & — & — & — \\\\"
            )
            satisfaction_rows.append(
                f"{community_name} & — & — & — & — & — & — & — \\\\"
            )
            continue
        effective_visits = best_solution['community_effective_daily_visits'][community_idx]
        coverage_rows.append(
            f"{community_name} & {elderly:.0f} & {daily_demand:.1f} & 是 & "
            f"{comm_names[detail['station_idx']]} & {detail['distance']:.0f} & "
            f"{satisfaction_value(best_solution, community_idx):.2f} & {effective_visits:.1f} \\\\"
        )
        satisfaction_rows.append(
            f"{community_name} & {comm_names[detail['station_idx']]} & {detail['distance']:.0f} & "
            f"{detail['s1']:.2f} & {detail['utilization'] * 100:.1f} & {detail['s2']:.2f} & "
            f"{detail['s3']:.2f} & {detail['composite']:.2f} \\\\"
        )
    table2 = rf"""\begin{{table}}[htbp]
\centering
\caption{{问题2.3：各小区覆盖、归属与有效服务量}}
\begin{{tabular}}{{|c|c|c|c|c|c|c|c|}}
\hline
小区 & 老人数 & 消费约束后日需求(次/日) & 是否覆盖 & 唯一归属站点 & 距离(m) & 综合满意度 & 有效人次/日 \\
\hline
{chr(10).join(coverage_rows)}
\hline
全区 & {elderly_by_community.sum():.0f} & {community_daily_demand.sum():.1f} & — & — & — & {best_solution['avg_satisfaction']:.4f} & {best_solution['community_effective_daily_visits'].sum():.1f} \\
\hline
\end{{tabular}}
\vspace{{0.2cm}}
说明：T2阶段每个被覆盖小区只保留一个最终归属站点，有效人次按日需求乘综合满意度计算。
\end{{table}}
"""

    load_rows = []
    total_load = 0.0
    for station_idx in best_solution['active_stations']:
        scale = best_solution['layout'][station_idx]
        capacity = best_solution['station_capacities'][station_idx]
        load = best_solution['station_loads'][station_idx]
        utilization = best_solution['station_utilizations'][station_idx]
        detail_s2 = next(
            (
                detail['s2']
                for detail in best_solution['satisfaction_details'].values()
                if detail['station_idx'] == station_idx
            ),
            0.0,
        )
        total_load += load
        load_rows.append(
            f"{comm_names[station_idx]} & {SCALE_NAMES[scale]} & {capacity:.0f} & "
            f"{capacity * capacity_limit_ratio:.0f} & {load:.1f} & "
            f"{utilization * 100:.1f}\\% & {detail_s2:.2f} & "
            f"{capacity_status(load, capacity, capacity_limit_ratio)} \\\\"
        )
    global_utilization = total_load / total_capacity if total_capacity > 0 else 0.0
    table3 = rf"""\begin{{table}}[htbp]
\centering
\caption{{问题2.3：最优方案站点实际有效负荷、附件容量与弹性校验}}
\begin{{tabular}}{{|c|c|c|c|c|c|c|c|}}
\hline
站点 & 规模 & 附件容量 & 弹性上限 & 实际有效负荷 & 附件利用率 & S2 & 容量状态 \\
\hline
{chr(10).join(load_rows)}
\hline
全区 & — & {total_capacity:.0f} & {total_capacity * capacity_limit_ratio:.0f} & {total_load:.1f} & {global_utilization * 100:.1f}\% & — & 弹性上限内 \\
\hline
\end{{tabular}}
\vspace{{0.2cm}}
说明：方案B中站点负荷为归属小区的实际有效日服务人次之和；本轮按附件容量的 {best_solution.get('capacity_overload_ratio', 0.0) * 100:.0f}\% 小幅弹性上限校验。
\end{{table}}
"""

    table4 = rf"""\begin{{table}}[htbp]
\centering
\caption{{问题2.3：各小区满意度分解}}
\begin{{tabular}}{{|c|c|c|c|c|c|c|c|}}
\hline
小区 & 归属站点 & 距离(m) & S1(距离) & 归属站利用率(\%) & S2(响应) & S3(价格) & 综合满意度 \\
\hline
{chr(10).join(satisfaction_rows)}
\hline
\end{{tabular}}
\vspace{{0.2cm}}
说明：综合满意度满足 $S=0.2S_1+0.3S_2+0.5S_3$，T2中 $S_3=1.00$。
\end{{table}}
"""

    financial_rows = []
    financial_sums = {
        'effective_visits': 0.0,
        'fixed': 0.0,
        'depreciation': 0.0,
        'revenue': 0.0,
        'direct_cost': 0.0,
        'profit': 0.0,
    }
    for station_idx in best_solution['active_stations']:
        scale = best_solution['layout'][station_idx]
        spec = station_spec(config, scale)
        finance = station_financials[station_idx]
        for key, finance_key in [
            ('effective_visits', 'annual_effective_visits'),
            ('fixed', 'annual_fixed_cost'),
            ('depreciation', 'annual_depreciation'),
            ('revenue', 'annual_revenue'),
            ('direct_cost', 'annual_direct_cost'),
            ('profit', 'annual_profit'),
        ]:
            financial_sums[key] += finance[finance_key]
        profit_margin = (
            finance['annual_profit'] / finance['annual_revenue'] * 100
        ) if finance['annual_revenue'] > 0 else 0.0
        financial_rows.append(
            f"{comm_names[station_idx]} & {SCALE_NAMES[scale]} & {finance['annual_effective_visits']:.0f} & "
            f"{spec['daily_fixed_cost']:.0f} & {finance['annual_fixed_cost']:.1f} & "
            f"{finance['annual_depreciation']:.2f} & {finance['annual_revenue']:.1f} & "
            f"{finance['annual_direct_cost']:.1f} & {finance['annual_profit']:.1f} & {profit_margin:.1f}\\% \\\\"
        )
    global_profit_margin = (
        financial_sums['profit'] / financial_sums['revenue'] * 100
    ) if financial_sums['revenue'] > 0 else 0.0
    table5 = rf"""\begin{{table}}[htbp]
\centering
\caption{{问题2.3：各服务站年度利润分解（金额单位：万元）}}
\begin{{tabular}}{{|c|c|c|c|c|c|c|c|c|c|}}
\hline
站点 & 规模 & 年有效人次 & 日固定(元) & 年固定 & 年折旧 & 年收入 & 年支出 & 年利润 & 利润率 \\
\hline
{chr(10).join(financial_rows)}
\hline
全区 & — & {financial_sums['effective_visits']:.0f} & — & {financial_sums['fixed']:.1f} & {financial_sums['depreciation']:.2f} & {financial_sums['revenue']:.1f} & {financial_sums['direct_cost']:.1f} & {financial_sums['profit']:.1f} & {global_profit_margin:.1f}\% \\
\hline
\end{{tabular}}
\vspace{{0.2cm}}
说明：问题2不计政府补贴，年利润=年收入-年直接支出-年固定成本-年折旧。
\end{{table}}
"""

    table6 = r"""\begin{table}[htbp]
\centering
\caption{问题2.2：修订后全局枚举与容量约束归属算法复杂度}
\begin{tabular}{|c|c|c|}
\hline
步骤 & 操作 & 复杂度说明 \\
\hline
1. 布局枚举 & 10个小区各取不建、小、中、大四种状态 & $O(4^{10})$ \\
2. 预算剪枝 & 逐布局计算建设成本 & $O(4^{10}\times 10)$ \\
3. 候选归属构造 & 对预算可行布局建立1000米内单站候选 & $O(m\times 10\times p)$ \\
4. 归属优化 & 固定S2估计下在并列最高满意度候选中搜索弹性容量可行归属 & 最坏 $O(m\times p^{10})$ \\
5. 反馈迭代 & 用有效服务负荷更新利用率与响应满意度 & $O(m\times k\times p^{10})$ \\
6. 评价排序 & 计算覆盖、满意度、利润并排序 & $O(n\log n)$ \\
\hline
\end{tabular}

\vspace{0.2cm}
说明：$m$为预算可行布局数，$p$为单个小区可达站点候选数，$k$为方案B归属反馈迭代轮数，$n$为容量可行方案数。本题只有10个小区，归属搜索可在预算剪枝后完成。
\end{table}
"""

    tables = {
        'T2_01_table_optimal_layout.tex': table1,
        'T2_02_table_coverage_by_community.tex': table2,
        'T2_03_table_station_loads.tex': table3,
        'T2_04_table_satisfaction_scores.tex': table4,
        'T2_05_table_annual_profit.tex': table5,
        'T2_06_table_algorithm_complexity.tex': table6,
    }
    for filename, content in tables.items():
        with (output_dir / filename).open('w', encoding='utf-8') as table_file:
            table_file.write(content)
    return station_financials


def plot_coverage_network(best_solution, config, population_year5, output_dir):
    comm_names = config['communities']
    num_communities = len(comm_names)
    active_stations = best_solution['active_stations']
    layout = best_solution['layout']
    x_communities = np.arange(num_communities)

    plt.figure(figsize=(11.4, 6.2))
    plt.scatter(
        x_communities,
        [0] * num_communities,
        s=[population_year5[idx].sum() * 0.55 for idx in range(num_communities)],
        c='#b9ddf3',
        edgecolor='#355070',
        zorder=4,
        label='小区(点大小=老人数量)',
    )
    plt.scatter(
        active_stations,
        [1] * len(active_stations),
        s=[best_solution['station_capacities'][idx] * 0.18 for idx in active_stations],
        c='#ef8354',
        edgecolor='#7a2e16',
        zorder=5,
        label='站点(点大小=附件容量)',
    )

    color_by_s1 = {1.00: '#0F766E', 0.90: '#A3C76D', 0.75: '#E9C46A', 0.60: '#E76F51'}
    label_offsets = {}
    for order, (community_idx, station_idx) in enumerate(best_solution['assignments'].items()):
        detail = best_solution['satisfaction_details'][community_idx]
        midpoint = (community_idx + station_idx) / 2
        offset_key = round(midpoint, 1)
        label_offsets[offset_key] = label_offsets.get(offset_key, -1) + 1
        y_offset = (label_offsets[offset_key] - 0.5) * 0.055
        plt.plot(
            [community_idx, station_idx],
            [0, 1],
            color=color_by_s1[detail['s1']],
            linewidth=1.8,
            alpha=0.8,
            zorder=1,
        )
        plt.text(
            midpoint,
            0.50 + y_offset,
            f"{detail['distance']:.0f}m",
            fontsize=7,
            ha='center',
            va='center',
            color='#333333',
            bbox=dict(boxstyle='round,pad=0.10', facecolor='white', edgecolor='none', alpha=0.72),
        )

    for community_idx, community_name in enumerate(comm_names):
        plt.text(community_idx, -0.11, community_name, ha='center', fontweight='bold')
    for station_idx in active_stations:
        plt.text(
            station_idx,
            1.11,
            f"{comm_names[station_idx]}\n{SCALE_NAMES[layout[station_idx]]}",
            ha='center',
            color='#7a2e16',
            fontweight='bold',
        )

    legend_lines = [
        plt.Line2D([0], [0], color=color, lw=2, label=f'S1={score:.2f}')
        for score, color in color_by_s1.items()
    ]
    legend_nodes = [
        plt.Line2D(
            [0], [0],
            marker='o',
            linestyle='none',
            markersize=11,
            markerfacecolor='#b9ddf3',
            markeredgecolor='#355070',
            label='小区(点大小=老人数量)',
        ),
        plt.Line2D(
            [0], [0],
            marker='o',
            linestyle='none',
            markersize=10,
            markerfacecolor='#ef8354',
            markeredgecolor='#7a2e16',
            label='站点(点大小=附件容量)',
        ),
    ]
    plt.legend(
        handles=legend_nodes + legend_lines,
        loc='upper right',
        bbox_to_anchor=(0.985, 0.985),
        ncol=1,
        fontsize=9,
        framealpha=0.92,
        borderaxespad=0.2,
        handlelength=2.2,
        handletextpad=0.8,
        labelspacing=0.55,
    )
    plt.xlim(-0.55, num_communities - 0.45)
    plt.ylim(-0.27, 1.56)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(output_dir / 'T2_01_coverage_network.png', dpi=300)
    plt.close()


def plot_capacity_elasticity_scan(scan_results, output_dir):
    ratios = [result['overload_ratio'] * 100 for result in scan_results]
    valid_counts = [result['valid_solution_count'] for result in scan_results]
    full_coverage_counts = [result['full_coverage_count'] for result in scan_results]
    first_feasible_idx = next(
        (
            idx
            for idx, result in enumerate(scan_results)
            if result['full_coverage_count'] > 0
        ),
        None,
    )

    plt.figure(figsize=(9, 6))
    x = np.arange(len(ratios))
    width = 0.34
    valid_bars = plt.bar(
        x - width / 2,
        valid_counts,
        width=width,
        color='#457b9d',
        edgecolor='black',
        label='可行方案数',
    )
    full_bars = plt.bar(
        x + width / 2,
        full_coverage_counts,
        width=width,
        color='#2a9d55',
        edgecolor='black',
        label='全覆盖方案数',
    )

    for bars in [valid_bars, full_bars]:
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + max(max(valid_counts, default=0) * 0.02, 0.3),
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontweight='bold',
            )

    if first_feasible_idx is not None:
        plt.axvline(first_feasible_idx, color='#d62828', linestyle='--', linewidth=1.8)
        plt.text(
            first_feasible_idx,
            max(valid_counts, default=0) * 0.83,
            f'最小可行弹性\n{ratios[first_feasible_idx]:.0f}%',
            color='#9d0208',
            ha='center',
            va='center',
            fontweight='bold',
            bbox={'facecolor': 'white', 'edgecolor': '#d62828', 'alpha': 0.9},
        )

    plt.xticks(x, [f'{ratio:.0f}%' for ratio in ratios])
    plt.xlabel('容量弹性比例')
    plt.ylabel('方案数量')
    plt.title('严格最高满意度归属下的容量弹性扫描')
    plt.grid(axis='y', linestyle='--', alpha=0.45)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(output_dir / 'T2_02_pareto_frontier.png', dpi=300)
    plt.close()


def plot_satisfaction_decomposition(best_solution, config, output_dir):
    comm_names = config['communities']
    weights = config['satisfaction_weights']
    x = np.arange(len(comm_names))
    s1_parts = []
    s2_parts = []
    s3_parts = []
    totals = []
    labels = []

    for community_idx, community_name in enumerate(comm_names):
        detail = best_solution['satisfaction_details'].get(community_idx)
        if detail is None:
            s1_parts.append(0.0)
            s2_parts.append(0.0)
            s3_parts.append(0.0)
            totals.append(0.0)
            labels.append(f'{community_name}\n未覆盖')
            continue
        s1_parts.append(weights['distance'] * detail['s1'])
        s2_parts.append(weights['response'] * detail['s2'])
        s3_parts.append(weights['price'] * detail['s3'])
        totals.append(detail['composite'])
        labels.append(f"{community_name}\n->{comm_names[detail['station_idx']]}")

    plt.figure(figsize=(11, 6))
    plt.bar(x, s1_parts, color='#2a9d8f', edgecolor='black', label='距离分项 0.2*S1')
    plt.bar(
        x,
        s2_parts,
        bottom=s1_parts,
        color='#457b9d',
        edgecolor='black',
        label='响应分项 0.3*S2',
    )
    bottoms = np.asarray(s1_parts) + np.asarray(s2_parts)
    plt.bar(
        x,
        s3_parts,
        bottom=bottoms,
        color='#f4a261',
        edgecolor='black',
        label='价格分项 0.5*S3',
    )
    segment_specs = [
        (np.zeros(len(x)), np.asarray(s1_parts), 'white'),
        (np.asarray(s1_parts), np.asarray(s2_parts), 'white'),
        (bottoms, np.asarray(s3_parts), '#111827'),
    ]
    for segment_bottoms, segment_values, text_color in segment_specs:
        for idx, value in enumerate(segment_values):
            if value <= 0:
                continue
            plt.text(
                idx,
                segment_bottoms[idx] + value / 2,
                f'{value:.2f}',
                ha='center',
                va='center',
                fontsize=7,
                color=text_color,
                fontweight='bold',
            )
    for idx, total in enumerate(totals):
        if total > 0:
            plt.text(
                idx,
                total + 0.02,
                f'{total:.2f}',
                ha='center',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.10', facecolor='white', edgecolor='none', alpha=0.75),
            )
    plt.xticks(x, labels)
    plt.ylabel('综合满意度')
    plt.ylim(0, 1.12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / 'T2_03_satisfaction_decomposition.png', dpi=300)
    plt.close()


def plot_station_loads(best_solution, config, output_dir):
    comm_names = config['communities']
    active_stations = best_solution['active_stations']
    colors = plt.cm.tab10.colors
    bottoms = np.zeros(len(active_stations))
    capacity_limit_ratio = best_solution.get('capacity_limit_ratio', 1.0)

    plt.figure(figsize=(11, 6))
    for community_idx, assigned_station in sorted(best_solution['assignments'].items()):
        loads = np.zeros(len(active_stations))
        station_plot_idx = active_stations.index(assigned_station)
        loads[station_plot_idx] = best_solution['community_effective_daily_visits'][community_idx]
        plt.bar(
            range(len(active_stations)),
            loads,
            bottom=bottoms,
            color=colors[community_idx % len(colors)],
            edgecolor='white',
            label=f'小区{comm_names[community_idx]}',
        )
        bottoms += loads

    for plot_idx, station_idx in enumerate(active_stations):
        capacity = best_solution['station_capacities'][station_idx]
        load = best_solution['station_loads'][station_idx]
        utilization = best_solution['station_utilizations'][station_idx] * 100
        plt.hlines(
            capacity,
            plot_idx - 0.42,
            plot_idx + 0.42,
            colors='black',
            linestyles='dashed',
            lw=2,
            label='附件容量' if plot_idx == 0 else None,
        )
        if capacity_limit_ratio > 1.0:
            plt.hlines(
                capacity * capacity_limit_ratio,
                plot_idx - 0.42,
                plot_idx + 0.42,
                colors='#c1121f',
                linestyles='dotted',
                lw=2,
                label='弹性上限' if plot_idx == 0 else None,
            )
        plt.text(
            plot_idx,
            load + capacity * 0.035,
            f'{load:.1f}\n{utilization:.1f}%',
            ha='center',
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.78),
        )

    plt.xticks(
        range(len(active_stations)),
        [f"站点{comm_names[idx]}\n{SCALE_NAMES[best_solution['layout'][idx]]}" for idx in active_stations],
    )
    plt.ylabel('实际有效日负荷(人次/日)')
    plt.grid(axis='y', linestyle='--', alpha=0.45)
    plt.ylim(0, max(best_solution['station_loads'].values()) * 1.20)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=5, title='负荷来源', frameon=False)
    plt.tight_layout()
    plt.savefig(output_dir / 'T2_04_station_loads_distribution.png', dpi=300)
    plt.close()


def plot_station_profits(best_solution, config, station_financials, output_dir):
    comm_names = config['communities']
    active_stations = best_solution['active_stations']
    profits = [station_financials[idx]['annual_profit'] for idx in active_stations]
    margins = [
        (
            station_financials[idx]['annual_profit']
            / station_financials[idx]['annual_revenue']
            * 100
        ) if station_financials[idx]['annual_revenue'] > 0 else 0.0
        for idx in active_stations
    ]
    labels = [
        f"站点{comm_names[idx]}\n{SCALE_NAMES[best_solution['layout'][idx]]}"
        for idx in active_stations
    ]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(
        range(len(active_stations)),
        profits,
        color=['#2a9d55' if profit >= 0 else '#c1121f' for profit in profits],
        edgecolor='black',
        width=0.62,
    )
    plt.axhline(0, color='black', linewidth=1.2)
    for idx, bar in enumerate(bars):
        height = bar.get_height()
        offset = max(abs(height) * 0.03, 2.0)
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + offset if height >= 0 else height - offset,
            f'{profits[idx]:.1f}万\n{margins[idx]:.1f}%',
            ha='center',
            va='bottom' if height >= 0 else 'top',
            fontweight='bold',
        )
    plt.xticks(range(len(active_stations)), labels)
    plt.ylabel('年度利润(万元)')
    plt.title('最优方案：站点年度利润')
    plt.grid(axis='y', linestyle='--', alpha=0.45)
    plt.tight_layout()
    plt.savefig(output_dir / 'T2_05_annual_profit_distribution.png', dpi=300)
    plt.close()


def write_charts(scan_results, best_solution, config, population_year5, station_financials, output_dir):
    plot_coverage_network(best_solution, config, population_year5, output_dir)
    plot_capacity_elasticity_scan(scan_results, output_dir)
    plot_satisfaction_decomposition(best_solution, config, output_dir)
    plot_station_loads(best_solution, config, output_dir)
    plot_station_profits(best_solution, config, station_financials, output_dir)


def main():
    print('========== 问题 2.3-2.4：修订后结果表和图表生成 ==========\n')
    table_dir = Path('T2_04_table')
    figure_dir = Path('T2_04_figure')
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    all_solutions, best_solution, scan_results, config, population_year5, daily_demand_by_service = load_data()
    print('【输出口径】')
    print('  - 归属展示: 每个小区唯一归属站点')
    print('  - 负荷展示: 方案B实际有效日服务人次')
    print('  - 容量展示: 附件容量 1000/2000/3000 与弹性校验上限')
    print('  - 满意度展示: S1、S2、S3 可复算综合满意度\n')

    print('【表格生成】')
    station_financials = write_tables(
        best_solution,
        config,
        population_year5,
        daily_demand_by_service,
        table_dir,
    )
    for filename in [
        'T2_01_table_optimal_layout.tex',
        'T2_02_table_coverage_by_community.tex',
        'T2_03_table_station_loads.tex',
        'T2_04_table_satisfaction_scores.tex',
        'T2_05_table_annual_profit.tex',
        'T2_06_table_algorithm_complexity.tex',
    ]:
        print(f'  [OK] {filename}')

    print('\n【图表生成】')
    write_charts(
        scan_results,
        best_solution,
        config,
        population_year5,
        station_financials,
        figure_dir,
    )
    for filename in [
        'T2_01_coverage_network.png',
        'T2_02_pareto_frontier.png',
        'T2_03_satisfaction_decomposition.png',
        'T2_04_station_loads_distribution.png',
        'T2_05_annual_profit_distribution.png',
    ]:
        print(f'  [OK] {filename}')

    global_load = sum(best_solution['station_loads'].values())
    global_capacity = sum(best_solution['station_capacities'].values())
    capacity_limit_ratio = best_solution.get('capacity_limit_ratio', 1.0)
    max_utilization_station = max(
        best_solution['active_stations'],
        key=lambda station_idx: best_solution['station_utilizations'][station_idx],
    )
    print('\n【结果摘要】')
    print(f"  最优布局: {best_solution['layout']}")
    print(f"  覆盖率: {best_solution['coverage_rate'] * 100:.2f}%")
    print(f"  平均满意度: {best_solution['avg_satisfaction']:.4f}")
    print(f'  全区有效负荷/容量: {global_load:.1f}/{global_capacity:.0f}')
    print(
        f"  容量弹性上限: {best_solution.get('capacity_overload_ratio', 0.0) * 100:.0f}% "
        f"(全区上限 {global_capacity * capacity_limit_ratio:.0f})"
    )
    print(
        f"  最高利用率站点: {config['communities'][max_utilization_station]} "
        f"({best_solution['station_utilizations'][max_utilization_station] * 100:.1f}%)"
    )
    print('\n所有表格和图表生成完毕！')
    print('================================')


if __name__ == '__main__':
    main()
