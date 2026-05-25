import os
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_pickle(path):
    with Path(path).open('rb') as file:
        return pickle.load(file)


def load_all_data():
    required = [
        'model/T0_01_config.pkl',
        'model/T2_04_best_solution.pkl',
        'model/T3_01_financial_model_config.pkl',
        'model/T3_05_optimal_price_plan.pkl',
        'model/T3_06_accessibility_metrics.pkl',
    ]
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        print('[ERROR] Missing required files:')
        for path in missing:
            print(f'  - {path}')
        sys.exit(1)

    return (
        load_pickle('model/T0_01_config.pkl'),
        load_pickle('model/T2_04_best_solution.pkl'),
        load_pickle('model/T3_01_financial_model_config.pkl'),
        load_pickle('model/T3_05_optimal_price_plan.pkl'),
        load_pickle('model/T3_06_accessibility_metrics.pkl'),
    )


def save_latex(filename, content):
    os.makedirs('T3_07_table', exist_ok=True)
    with open(Path('T3_07_table') / filename, 'w', encoding='utf-8') as file:
        file.write(content)


def fmt(value, digits=2):
    return f'{float(value):.{digits}f}'


def distance_satisfaction_score(distance, rules):
    for segment in rules['segments']:
        if segment.get('min_distance', -1) < distance <= segment.get('max_distance', float('inf')):
            return segment['score']
    return 0.0


def station_name(config, station_idx):
    return config['communities'][station_idx]


def station_scale(t2_best, station_idx):
    return {1: '小型', 2: '中型', 3: '大型'}[t2_best['layout'][station_idx]]


def generate_tables(config, t2_best, financial_config, t3_plan, accessibility_data):
    active_stations = list(financial_config['active_stations'])
    paid_services = list(financial_config['paid_services'])
    base_prices = financial_config['service_prices_base'][:len(paid_services)]
    station_prices = t3_plan['station_prices']
    station_multipliers = t3_plan.get('station_uniform_multipliers', {})
    sys_fin = t3_plan['system_financials']
    metrics = accessibility_data['metrics']

    table1 = r"""\begin{table}[h]
\centering
\caption{问题3.1：定价与补贴财务模型定义}
\begin{tabular}{|c|c|c|}
\hline
项目 & 计算口径 & 单位 \\
\hline
付费服务价格 & $p_{j,s}=r_j p^0_s$，紧急救助价格固定为0 & 元/次 \\
有效服务人次 & 需求人次 $\times$ 综合满意度 & 人次 \\
服务营收 & $\sum$ 有效付费人次 $\times$ 分站服务价格 & 万元/年 \\
政府补贴 & $\min(2\times$付费有效人次, 站点日补贴上限$)\times365$ & 万元/年 \\
年运营成本 & 日固定管理成本$\times365+$建设成本折旧 & 万元/年 \\
年利润 & 服务营收 + 政府补贴 - 变动成本 - 年运营成本 & 万元/年 \\
利润率 & 年利润 / 年运营成本 $\times100\%$ & \% \\
\hline
\end{tabular}
\end{table}"""
    save_latex('T3_01_table_financial_model_definition.tex', table1)

    header = '服务项目 & 基准价'
    for station_idx in active_stations:
        header += f" & {station_name(config, station_idx)}站价格 & {station_name(config, station_idx)}站S3"
    rows = []
    for service_idx, service in enumerate(paid_services):
        row = f"{service} & {fmt(base_prices[service_idx], 2)}"
        for station_idx in active_stations:
            row += (
                f" & {fmt(station_prices[station_idx][service], 2)}"
                f" & {fmt(t3_plan['station_s3_by_service'][station_idx][service_idx], 2)}"
            )
        rows.append(row + r" \\")
    emergency_row = '紧急救助 & 0.00' + ''.join(' & 0.00 & --' for _ in active_stations) + r" \\"
    rows.append(emergency_row)

    multiplier_note = '；'.join(
        f"{station_name(config, j)}站倍率={station_multipliers.get(j, float('nan')):.3f}"
        for j in active_stations
    )
    table2 = r"""\begin{table}[h]
\centering
\caption{问题3.2：逐站最优定价与价格满意度}
\begin{tabular}{|c|c|""" + 'cc|' * len(active_stations) + r"""}
\hline
""" + header + r""" \\
\hline
""" + '\n'.join(rows) + r"""
\hline
\end{tabular}
\vspace{0.2cm}
说明：""" + multiplier_note + r"""；紧急救助公益免费，不参与补贴。
\end{table}"""
    save_latex('T3_02_table_optimal_pricing_plan.tex', table2)

    table3_rows = []
    for station_idx in active_stations:
        fin = t3_plan['station_financials'][station_idx]
        table3_rows.append(
            f"{station_name(config, station_idx)} & {station_scale(t2_best, station_idx)}"
            f" & {fmt(fin['annual_service_revenue'])}"
            f" & {fmt(fin['annual_subsidy'])}"
            f" & {fmt(fin['annual_variable_cost'])}"
            f" & {fmt(fin['annual_operating_cost'])}"
            f" & {fmt(fin['annual_profit'])}"
            f" & {fmt(fin['profit_margin'])}\\% \\\\"
        )
    table3_rows.append(
        f"全区汇总 & -- & {fmt(sys_fin['annual_service_revenue'])}"
        f" & {fmt(sys_fin['annual_subsidy'])}"
        f" & {fmt(sys_fin['annual_variable_cost'])}"
        f" & {fmt(sys_fin['annual_operating_cost'])}"
        f" & {fmt(sys_fin['annual_profit'])}"
        f" & {fmt(sys_fin['profit_margin'])}\\% \\\\"
    )
    table3 = r"""\begin{table}[h]
\centering
\caption{问题3.2：最优定价下各服务站年度财务结果}
\begin{tabular}{|c|c|c|c|c|c|c|c|}
\hline
站点 & 规模 & 服务营收 & 政府补贴 & 变动成本 & 年运营成本 & 年利润 & 利润率 \\
\hline
""" + '\n'.join(table3_rows) + r"""
\hline
\end{tabular}
\end{table}"""
    save_latex('T3_03_table_financial_analysis.tex', table3)

    table4_rows = []
    for station_idx in active_stations:
        fin = t3_plan['station_financials'][station_idx]
        table4_rows.append(
            f"{station_name(config, station_idx)} & {station_scale(t2_best, station_idx)}"
            f" & {fmt(fin['paid_effective_daily_visits'], 1)}"
            f" & {fmt(fin['daily_subsidy_request_yuan'], 1)}"
            f" & {fmt(fin['daily_subsidy_cap_yuan'], 1)}"
            f" & {fmt(fin['daily_subsidy_actual_yuan'], 1)}"
            f" & {fmt(fin['annual_subsidy'])} \\\\"
        )
    table4 = r"""\begin{table}[h]
\centering
\caption{问题3.2：各站点日补贴请求与上限截断}
\begin{tabular}{|c|c|c|c|c|c|c|}
\hline
站点 & 规模 & 付费日有效人次 & 日补贴请求(元) & 日上限(元) & 实际日补贴(元) & 年补贴(万元) \\
\hline
""" + '\n'.join(table4_rows) + r"""
\hline
\end{tabular}
\vspace{0.2cm}
说明：实际补贴严格按 $\min(2\times$付费日有效人次, 日补贴上限$)$ 计算。
\end{table}"""
    save_latex('T3_04_table_subsidy_analysis.tex', table4)

    pop_year5 = np.load('model/T1_01_population_year5.npy')
    elderly_pop = pop_year5.sum(axis=1)
    distance_matrix = config['distance_matrix']
    distance_rules = config['satisfaction_rules']['distance']
    satisfaction_rows = []
    for community_idx, community in enumerate(config['communities']):
        assigned = t2_best['assignments'].get(community_idx, -1)
        assigned_name = station_name(config, assigned) if assigned != -1 else '--'
        if assigned != -1:
            s1 = distance_satisfaction_score(distance_matrix[community_idx, assigned], distance_rules)
            s2 = t3_plan['station_response_scores'][assigned]
        else:
            s1 = 0.0
            s2 = 0.0
        satisfaction_rows.append(
            f"{community} & {int(elderly_pop[community_idx])}"
            f" & {assigned_name}"
            f" & {fmt(s1, 2)}"
            f" & {fmt(s2, 2)}"
            f" & {fmt(t3_plan['weighted_s3_by_community'][community_idx], 2)}"
            f" & {fmt(t3_plan['satisfaction_by_community'][community_idx], 2)} \\\\"
        )
    table5 = r"""\begin{table}[h]
\centering
\caption{问题3.2：各小区满意度分解}
\small
\begin{tabular}{|c|c|c|c|c|c|c|}
\hline
小区 & 老人数 & 归属站点 & 距离满意度S1 & 响应满意度S2 & 价格满意度S3 & 综合满意度 \\
\hline
""" + '\n'.join(satisfaction_rows) + r"""
\hline
\end{tabular}
\end{table}"""
    save_latex('T3_05_table_community_satisfaction.tex', table5)

    access_rows = []
    economic_rows = []
    for key in ['healthy', 'semi_disabled', 'disabled']:
        m = metrics[key]
        access_rows.append(
            f"{m['name']} & {fmt(m['coverage_rate'], 1)}\\%"
            f" & {fmt(m['average_distance'], 1)}"
            f" & {fmt(m['s1_avg'], 4)}"
            f" & {fmt(m['s2_avg'], 4)}"
            f" & {fmt(m['s3_avg'], 4)}"
            f" & {fmt(m['satisfaction_avg'], 4)} \\\\"
        )
        economic_rows.append(
            f"{m['name']} & {fmt(m['base_payment_pc'])}"
            f" & {fmt(m['optimal_payment_pc'])}"
            f" & {fmt(m['payment_reduction_pc'])}"
            f" & {fmt(m['burden_optimal'])}\\%"
            f" & {fmt(m['fiscal_subsidy_pc'])} \\\\"
        )

    table6 = r"""\begin{table}[h]
\centering
\caption{问题3.3：三类老人空间与服务可及性}
\begin{tabular}{|c|c|c|c|c|c|c|}
\hline
老人类型 & 覆盖率 & 平均距离(m) & 距离S1 & 响应S2 & 价格S3 & 综合满意度 \\
\hline
""" + '\n'.join(access_rows) + r"""
\hline
\end{tabular}
\end{table}"""
    save_latex('T3_06_table_spatial_service_accessibility.tex', table6)
    save_latex('T3_05_table_elderly_satisfaction.tex', table6)
    save_latex('T3_06_table_spatial_accessibility.tex', table6)

    table7 = r"""\begin{table}[h]
\centering
\caption{问题3.3：三类老人经济可及性}
\begin{tabular}{|c|c|c|c|c|c|}
\hline
老人类型 & 基准月支付 & 最优月支付 & 支付降幅 & 最优负担率 & 人均财政补贴强度 \\
\hline
""" + '\n'.join(economic_rows) + r"""
\hline
\end{tabular}
\vspace{0.2cm}
说明：最优月支付按小区归属站点的真实分站价格计算；财政补贴强度不是直接现金发放。
\end{table}"""
    save_latex('T3_07_table_economic_accessibility.tex', table7)

    capacity_rows = []
    for station_idx in active_stations:
        capacity_rows.append(
            f"{station_name(config, station_idx)} & {station_scale(t2_best, station_idx)}"
            f" & {fmt(financial_config['station_capacities'][station_idx], 0)}"
            f" & {fmt(financial_config['station_elastic_capacity_limits'][station_idx], 0)}"
            f" & {fmt(t3_plan['station_loads'][station_idx], 1)}"
            f" & {fmt(t3_plan['station_utilizations'][station_idx] * 100, 2)}\\% \\\\"
        )
    table8 = r"""\begin{table}[h]
\centering
\caption{问题3.2：T2弹性容量口径继承检查}
\begin{tabular}{|c|c|c|c|c|c|}
\hline
站点 & 规模 & 附件容量 & T2弹性上限 & T3有效负荷 & 原容量利用率 \\
\hline
""" + '\n'.join(capacity_rows) + r"""
\hline
\end{tabular}
\vspace{0.2cm}
说明：T3沿用T2确定的 $1.02$ 弹性容量上限。
\end{table}"""
    save_latex('T3_08_table_capacity_check.tex', table8)
    save_latex('T3_08_table_service_accessibility.tex', table8)


def generate_charts(config, financial_config, t3_plan, accessibility_data):
    os.makedirs('T3_07_figure', exist_ok=True)
    active_stations = list(financial_config['active_stations'])
    paid_services = list(financial_config['paid_services'])
    station_prices = t3_plan['station_prices']
    metrics = accessibility_data['metrics']

    x = np.arange(len(paid_services))
    width = 0.23
    plt.figure(figsize=(10, 5.6))
    for offset, station_idx in enumerate(active_stations):
        values = [station_prices[station_idx][service] for service in paid_services]
        bars = plt.bar(x + (offset - 1) * width, values, width, label=f'{station_name(config, station_idx)}站')
        plt.bar_label(
            bars,
            labels=[f'{value:.1f}' for value in values],
            padding=3,
            fontsize=8,
            rotation=0,
        )
    plt.xticks(x, paid_services)
    plt.ylabel('价格(元/次)')
    plt.ylim(0, max(station_prices[station_idx][service] for station_idx in active_stations for service in paid_services) * 1.24)
    plt.legend(
        loc='upper right',
        bbox_to_anchor=(0.985, 0.985),
        ncol=1,
        frameon=True,
        framealpha=0.88,
        fontsize=9,
        borderpad=0.35,
        labelspacing=0.35,
        handlelength=1.6,
    )
    plt.tight_layout()
    plt.savefig('T3_07_figure/T3_01_station_prices.png', dpi=300)
    plt.savefig('T3_07_figure/T3_01_pricing_vs_satisfaction.png', dpi=300)
    plt.close()

    names = [station_name(config, idx) for idx in active_stations]
    margins = [t3_plan['station_financials'][idx]['profit_margin'] for idx in active_stations]
    profits = [t3_plan['station_financials'][idx]['annual_profit'] for idx in active_stations]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    bars = ax1.bar(names, margins, color=['#4C78A8', '#59A14F', '#F28E2B'], alpha=0.85)
    ax1.axhline(0, color='gray', linewidth=1)
    ax1.axhline(8, color='red', linestyle='--', linewidth=1, label='8%上限')
    ax1.set_ylabel('利润率(%)')
    ax1.set_ylim(0, max(8.8, max(margins) + 1))
    for bar, value in zip(bars, margins):
        ax1.text(bar.get_x() + bar.get_width() / 2, value + 0.15, f'{value:.2f}%', ha='center')
    ax2 = ax1.twinx()
    ax2.plot(names, profits, color='#B07AA1', marker='o', label='年利润')
    ax2.set_ylabel('年利润(万元)')
    fig.suptitle('逐站利润率与年度利润')
    fig.tight_layout()
    plt.savefig('T3_07_figure/T3_02_station_margin_profit.png', dpi=300)
    plt.close()

    labels = ['覆盖率', '距离便利', '需求释放', '支付可及', '综合满意']
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    for key, color in zip(['healthy', 'semi_disabled', 'disabled'], ['#4C78A8', '#F28E2B', '#59A14F']):
        m = metrics[key]
        values = [
            m['coverage_rate'],
            max(0.0, (1000.0 - m['average_distance']) / 1000.0 * 100.0),
            m['demand_release_rate'],
            max(0.0, 100.0 - m['burden_optimal']),
            m['satisfaction_avg'] * 100.0,
        ]
        values += values[:1]
        ax.plot(angles, values, label=m['name'], color=color, linewidth=2)
        ax.fill(angles, values, color=color, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 100)
    ax.set_title('三类老人可及性雷达图')
    ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.1))
    plt.tight_layout()
    plt.savefig('T3_07_figure/T3_03_accessibility_radar_chart.png', dpi=300)
    plt.close()

    x = np.arange(3)
    keys = ['healthy', 'semi_disabled', 'disabled']
    labels_zh = [metrics[key]['name'] for key in keys]
    s1 = np.asarray([metrics[key]['s1_avg'] * 0.2 for key in keys])
    s2 = np.asarray([metrics[key]['s2_avg'] * 0.3 for key in keys])
    s3 = np.asarray([metrics[key]['s3_avg'] * 0.5 for key in keys])
    plt.figure(figsize=(8, 5))
    plt.bar(x, s1, label='距离S1贡献', color='#4C78A8')
    plt.bar(x, s2, bottom=s1, label='响应S2贡献', color='#F28E2B')
    plt.bar(x, s3, bottom=s1 + s2, label='价格S3贡献', color='#59A14F')
    totals = s1 + s2 + s3
    segment_specs = [
        (np.zeros(len(x)), s1, 'white'),
        (s1, s2, 'white'),
        (s1 + s2, s3, 'white'),
    ]
    for bottoms, values, text_color in segment_specs:
        for idx, value in enumerate(values):
            if value <= 0:
                continue
            plt.text(
                idx,
                bottoms[idx] + value / 2,
                f'{value:.2f}',
                ha='center',
                va='center',
                fontsize=8,
                color=text_color,
                fontweight='bold',
            )
    for idx, total in enumerate(totals):
        plt.text(
            idx,
            total + 0.022,
            f'{total:.4f}',
            ha='center',
            fontsize=9,
            bbox=dict(boxstyle='round,pad=0.10', facecolor='white', edgecolor='none', alpha=0.75),
        )
    plt.xticks(x, labels_zh)
    plt.ylim(0, 1.12)
    plt.ylabel('综合满意度贡献')
    plt.legend(
        loc='upper right',
        bbox_to_anchor=(0.99, 0.99),
        frameon=True,
        framealpha=0.88,
        fontsize=8,
        borderpad=0.4,
        labelspacing=0.35,
        handlelength=1.6,
    )
    plt.tight_layout()
    plt.savefig('T3_07_figure/T3_02_elderly_satisfaction_decomposition.png', dpi=300)
    plt.close()


def main():
    print('========== T3_07：生成阶段三结果表和图表 ==========\n')
    config, t2_best, financial_config, t3_plan, accessibility_data = load_all_data()
    generate_tables(config, t2_best, financial_config, t3_plan, accessibility_data)
    generate_charts(config, financial_config, t3_plan, accessibility_data)
    print('已生成 T3_07_table/*.tex 和 T3_07_figure/*.png')
    print('================================')


if __name__ == '__main__':
    main()
