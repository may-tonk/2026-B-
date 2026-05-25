import pickle
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from T1_01_population_forecast import (
    FORECAST_HISTORY_PATH,
    T0_CONFIG_PATH,
    YEAR5_POPULATION_PATH,
    load_t0_config,
)
from T1_02_theoretical_demand import (
    ELDERLY_TYPE_KEYS,
    ELDERLY_TYPE_LABELS,
    THEORETICAL_DEMAND_BY_TYPE_PATH,
    THEORETICAL_DEMAND_PATH,
)
from T1_03_constrained_demand import (
    CONSTRAINED_MONTHLY_DEMAND_BY_TYPE_PATH,
    CONSTRAINED_MONTHLY_DEMAND_PATH,
    REDUCTION_COEFFICIENT_PATH,
)


PROJECT_ROOT = Path(__file__).resolve().parent
TABLE_DIR = PROJECT_ROOT / 'T1_04_table'
FIGURE_DIR = PROJECT_ROOT / 'T1_04_figure'


def display_int(value):
    return str(int(np.rint(float(value))))


def display_ratio(value):
    return f'{float(value):.3f}'


def save_text(path, content):
    path.write_text(content, encoding='utf-8')
    print(f'OK {path.name}')


def latex_table(caption, headers, rows):
    col_spec = '|'.join(['c'] * len(headers))
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        f'\\caption{{{caption}}}',
        f'\\begin{{tabular}}{{|{col_spec}|}}',
        r'\hline',
        ' & '.join(headers) + r' \\',
        r'\hline'
    ]
    lines.extend(' & '.join(map(str, row)) + r' \\' for row in rows)
    lines.extend([
        r'\hline',
        r'\end{tabular}',
        r'\end{table}',
        ''
    ])
    return '\n'.join(lines)


def service_rows(communities, service_names, demand_matrix):
    rows = []
    for idx, community in enumerate(communities):
        values = [display_int(value) for value in demand_matrix[idx]]
        rows.append([community, *values, display_int(demand_matrix[idx].sum())])
    totals = demand_matrix.sum(axis=0)
    rows.append(['全区', *[display_int(value) for value in totals], display_int(totals.sum())])
    return rows


def type_detail_rows(communities, demand_by_type):
    rows = []
    for community_idx, community in enumerate(communities):
        for key in ELDERLY_TYPE_KEYS:
            demand = demand_by_type[key][community_idx]
            rows.append([
                community,
                ELDERLY_TYPE_LABELS[key],
                *[display_int(value) for value in demand],
                display_int(demand.sum())
            ])
    return rows


def build_population_history_table(config, forecast_history):
    rows = []
    for record in forecast_history:
        population = record['population_matrix']
        for idx, community in enumerate(config['communities']):
            row = population[idx]
            rows.append([
                record['year'],
                community,
                *[display_int(value) for value in row],
                display_int(row.sum())
            ])
    return latex_table(
        '问题1.1：第1年至第5年末各小区老人数量预测',
        ['年份', '小区', '自理', '半失能', '失能', '合计'],
        rows
    )


def build_population_comparison_table(config, population_year5):
    rows = []
    initial = config['population_matrix']
    for idx, community in enumerate(config['communities']):
        rows.append([
            community,
            *[display_int(value) for value in initial[idx]],
            *[display_int(value) for value in population_year5[idx]],
            display_int(population_year5[idx].sum())
        ])

    rows.append([
        '全区',
        *[display_int(value) for value in initial.sum(axis=0)],
        *[display_int(value) for value in population_year5.sum(axis=0)],
        display_int(population_year5.sum())
    ])
    return latex_table(
        '问题1.1：第0年与第5年老人结构对比',
        ['小区', '自理(年0)', '半失能(年0)', '失能(年0)', '自理(年5)', '半失能(年5)', '失能(年5)', '合计(年5)'],
        rows
    )


def build_reduction_table(config, reduction_coefficient):
    rows = []
    for idx, community in enumerate(config['communities']):
        rows.append([
            community,
            *[display_ratio(value) for value in reduction_coefficient[idx]]
        ])
    return latex_table(
        '问题1.3：各小区各类老人消费约束削减系数',
        ['小区', '自理', '半失能', '失能'],
        rows
    )


def build_demand_comparison_table(config, theoretical_demand, constrained_demand):
    rows = []
    for idx, community in enumerate(config['communities']):
        theoretical_total = theoretical_demand[idx].sum()
        constrained_total = constrained_demand[idx].sum()
        reduction = theoretical_total - constrained_total
        reduction_rate = 0.0 if theoretical_total == 0 else reduction / theoretical_total * 100
        rows.append([
            community,
            display_int(theoretical_total),
            display_int(constrained_total),
            display_int(reduction),
            f'{reduction_rate:.2f}'
        ])

    theoretical_total = theoretical_demand.sum()
    constrained_total = constrained_demand.sum()
    reduction = theoretical_total - constrained_total
    reduction_rate = 0.0 if theoretical_total == 0 else reduction / theoretical_total * 100
    rows.append([
        '全区',
        display_int(theoretical_total),
        display_int(constrained_total),
        display_int(reduction),
        f'{reduction_rate:.2f}'
    ])
    return latex_table(
        '问题1：理论需求与消费约束后需求对比',
        ['小区', '理论需求(次/月)', '约束后需求(次/月)', '削减量(次/月)', '削减率(\\%)'],
        rows
    )


def save_population_png(config, forecast_history):
    """Write the year-0 through year-5 population structure chart as PNG."""
    colors = ['#2563EB', '#0F766E', '#E11D48']
    labels = ['自理', '半失能', '失能']
    year_labels = ['第0年', *[f"第{record['year']}年" for record in forecast_history]]
    yearly_totals = [
        config['population_matrix'].sum(axis=0).astype(float),
        *[record['population_matrix'].sum(axis=0).astype(float) for record in forecast_history]
    ]

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10.2, 5.4), dpi=180)
    x_positions = np.arange(len(year_labels))
    stacks = np.vstack(yearly_totals)
    bottoms = np.zeros(len(year_labels), dtype=float)

    for idx, label in enumerate(labels):
        values = stacks[:, idx]
        bars = ax.bar(
            x_positions,
            values,
            bottom=bottoms,
            width=0.62,
            color=colors[idx],
            label=label
        )
        ax.bar_label(
            bars,
            labels=[display_int(value) for value in values],
            label_type='center',
            color='white',
            fontsize=9
        )
        bottoms += values

    ax.set_title('T1 第0年至第5年老人结构变化', fontsize=14, pad=14)
    ax.set_ylabel('全区老人数量')
    ax.set_xticks(x_positions, year_labels)
    ax.set_ylim(0, bottoms.max() * 1.16)
    ax.grid(axis='y', color='#D1D5DB', linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.10))
    fig.tight_layout()

    figure_path = FIGURE_DIR / 'T1_01_population_structure_change.png'
    fig.savefig(figure_path, bbox_inches='tight')
    plt.close(fig)
    print(f'OK {figure_path.name}')
    return figure_path


def save_demand_structure_png(config, theoretical_demand, constrained_demand):
    """Write the year-5 service demand structure chart as PNG."""
    theoretical_totals = theoretical_demand.sum(axis=0)
    constrained_totals = constrained_demand.sum(axis=0)

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10.8, 5.8), dpi=180)
    x_positions = np.arange(len(config['service_names']))
    width = 0.36
    theoretical_bars = ax.bar(
        x_positions - width / 2,
        theoretical_totals,
        width=width,
        color='#2563EB',
        label='理论月需求'
    )
    constrained_bars = ax.bar(
        x_positions + width / 2,
        constrained_totals,
        width=width,
        color='#F59E0B',
        label='消费约束后月需求'
    )

    ax.bar_label(theoretical_bars, labels=[display_int(value) for value in theoretical_totals], fontsize=8, padding=5)
    ax.bar_label(constrained_bars, labels=[display_int(value) for value in constrained_totals], fontsize=8, padding=5)
    ax.set_ylabel('全区月需求量')
    ax.set_xticks(x_positions, config['service_names'])
    ax.set_ylim(0, max(theoretical_totals.max(), constrained_totals.max()) * 1.26)
    ax.grid(axis='y', color='#D1D5DB', linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()

    figure_path = FIGURE_DIR / 'T1_02_demand_structure_comparison.png'
    fig.savefig(figure_path, bbox_inches='tight')
    plt.close(fig)
    print(f'OK {figure_path.name}')
    return figure_path


def save_constraint_effect_png(config, theoretical_demand, constrained_demand):
    """Write the community-level consumption constraint effect chart as PNG."""
    theoretical_totals = theoretical_demand.sum(axis=1)
    constrained_totals = constrained_demand.sum(axis=1)
    reduction_rates = np.divide(
        theoretical_totals - constrained_totals,
        theoretical_totals,
        out=np.zeros_like(theoretical_totals, dtype=float),
        where=theoretical_totals > 0
    ) * 100

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(11.8, 6.0), dpi=180)
    x_positions = np.arange(len(config['communities']))
    width = 0.36
    theoretical_bars = ax.bar(
        x_positions - width / 2,
        theoretical_totals,
        width=width,
        color='#0F766E',
        label='理论月需求'
    )
    constrained_bars = ax.bar(
        x_positions + width / 2,
        constrained_totals,
        width=width,
        color='#E11D48',
        label='消费约束后月需求'
    )
    ax.bar_label(theoretical_bars, labels=[display_int(value) for value in theoretical_totals], fontsize=7, padding=2)
    ax.bar_label(constrained_bars, labels=[display_int(value) for value in constrained_totals], fontsize=7, padding=2)
    ax.set_title('T1 消费约束对各小区需求的削减效果', fontsize=14, pad=14)
    ax.set_ylabel('月需求总量')
    ax.set_xticks(x_positions, config['communities'], rotation=25, ha='right')
    ax.set_ylim(0, theoretical_totals.max() * 1.24)
    ax.grid(axis='y', color='#D1D5DB', linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)

    rate_axis = ax.twinx()
    rate_axis.plot(
        x_positions,
        reduction_rates,
        color='#7C3AED',
        marker='o',
        linewidth=2.0,
        label='削减率'
    )
    rate_axis.set_ylabel('削减率 (%)')
    rate_axis.set_ylim(0, max(5.0, reduction_rates.max() * 1.35))
    for x_position, reduction_rate in zip(x_positions, reduction_rates):
        rate_axis.annotate(
            f'{reduction_rate:.1f}%',
            (x_position, reduction_rate),
            textcoords='offset points',
            xytext=(0, 7),
            ha='center',
            fontsize=8,
            color='#5B21B6'
        )

    handles, labels = ax.get_legend_handles_labels()
    rate_handles, rate_labels = rate_axis.get_legend_handles_labels()
    ax.legend(
        handles + rate_handles,
        labels + rate_labels,
        frameon=False,
        ncol=3,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.16)
    )
    fig.tight_layout()

    figure_path = FIGURE_DIR / 'T1_03_consumption_constraint_effect.png'
    fig.savefig(figure_path, bbox_inches='tight')
    plt.close(fig)
    print(f'OK {figure_path.name}')
    return figure_path


def configure_plot_fonts():
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False


def population_year_series(config, forecast_history):
    year_labels = ['第0年', *[f"第{record['year']}年" for record in forecast_history]]
    populations = [
        config['population_matrix'].sum(axis=0).astype(float),
        *[record['population_matrix'].sum(axis=0).astype(float) for record in forecast_history]
    ]
    return year_labels, np.vstack(populations)


def save_population_trend_png(config, forecast_history):
    """Write the year-0 through year-5 population trend chart as PNG."""
    configure_plot_fonts()
    colors = ['#2563EB', '#0F766E', '#E11D48']
    labels = ['自理', '半失能', '失能']
    year_labels, stacks = population_year_series(config, forecast_history)
    x_positions = np.arange(len(year_labels))

    fig, ax = plt.subplots(figsize=(11.8, 6.3), dpi=180)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    label_bbox = dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.72)
    for idx, label in enumerate(labels):
        values = stacks[:, idx]
        ax.plot(x_positions, values, color=colors[idx], marker='o', linewidth=2.2, label=label)
        for point_idx, (x_position, value) in enumerate(zip(x_positions, values)):
            offset = (-10, 13) if idx == 1 and point_idx == 0 else (0, 9)
            ax.annotate(
                display_int(value),
                (x_position, value),
                textcoords='offset points',
                xytext=offset,
                ha='center',
                fontsize=8,
                color=colors[idx],
                bbox=label_bbox
            )

    total_axis = ax.twinx()
    total_values = stacks.sum(axis=1)
    total_axis.plot(
        x_positions,
        total_values,
        color='#111827',
        marker='D',
        linewidth=2.0,
        linestyle='--',
        label='总老人数'
    )
    for point_idx, (x_position, value) in enumerate(zip(x_positions, total_values)):
        offset = (12, -27) if point_idx == 0 else (0, -18)
        total_axis.annotate(
            display_int(value),
            (x_position, value),
            textcoords='offset points',
            xytext=offset,
            ha='center',
            fontsize=8,
            color='#111827',
            bbox=label_bbox
        )

    ax.set_ylabel('分类型老人数')
    total_axis.set_ylabel('总老人数')
    ax.set_xticks(x_positions, year_labels)
    ax.set_xlim(x_positions.min() - 0.25, x_positions.max() + 0.25)
    ax.set_ylim(0, stacks[:, 0].max() * 1.20)
    total_axis.set_ylim(total_values.min() * 0.92, total_values.max() * 1.08)
    ax.grid(axis='y', color='#D1D5DB', linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    handles, legend_labels = ax.get_legend_handles_labels()
    total_handles, total_labels = total_axis.get_legend_handles_labels()
    ax.legend(
        handles + total_handles,
        legend_labels + total_labels,
        frameon=False,
        ncol=4,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.13)
    )
    fig.tight_layout()

    figure_path = FIGURE_DIR / 'T1_01_population_forecast_trend.png'
    fig.savefig(figure_path, bbox_inches='tight')
    plt.close(fig)
    print(f'OK {figure_path.name}')
    return figure_path


def save_elderly_structure_comparison_png(config, population_year5):
    """Write the year-0 and year-5 elderly structure comparison chart as PNG."""
    configure_plot_fonts()
    colors = ['#2563EB', '#0F766E', '#E11D48']
    labels = ['自理', '半失能', '失能']
    stacks = np.vstack([
        config['population_matrix'].sum(axis=0).astype(float),
        population_year5.sum(axis=0).astype(float)
    ])
    percentages = stacks / stacks.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8.8, 5.6), dpi=180)
    x_positions = np.arange(2)
    bottoms = np.zeros(2, dtype=float)
    for idx, label in enumerate(labels):
        bars = ax.bar(
            x_positions,
            percentages[:, idx],
            bottom=bottoms,
            width=0.56,
            color=colors[idx],
            label=label
        )
        ax.bar_label(
            bars,
            labels=[f'{value:.1f}%' for value in percentages[:, idx]],
            label_type='center',
            color='white',
            fontsize=10
        )
        bottoms += percentages[:, idx]

    ax.set_title('T1 第0年与第5年老人结构对比', fontsize=14, pad=14)
    ax.set_ylabel('占比 (%)')
    ax.set_xticks(x_positions, ['第0年', '第5年'])
    ax.set_ylim(0, 100)
    ax.grid(axis='y', color='#D1D5DB', linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()

    figure_path = FIGURE_DIR / 'T1_02_elderly_structure_comparison.png'
    fig.savefig(figure_path, bbox_inches='tight')
    plt.close(fig)
    print(f'OK {figure_path.name}')
    return figure_path


def save_population_by_community_png(config, population_year5):
    """Write the year-0 and year-5 population comparison by community as PNG."""
    configure_plot_fonts()
    population_year0 = config['population_matrix'].sum(axis=1).astype(float)
    population_year5_totals = population_year5.sum(axis=1).astype(float)
    x_positions = np.arange(len(config['communities']))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11.8, 6.0), dpi=180)
    year0_bars = ax.bar(
        x_positions - width / 2,
        population_year0,
        width=width,
        color='#0F766E',
        label='第0年'
    )
    year5_bars = ax.bar(
        x_positions + width / 2,
        population_year5_totals,
        width=width,
        color='#F59E0B',
        label='第5年'
    )
    ax.bar_label(year0_bars, labels=[display_int(value) for value in population_year0], fontsize=7, padding=4)
    ax.bar_label(year5_bars, labels=[display_int(value) for value in population_year5_totals], fontsize=7, padding=4)

    ax.set_ylabel('老人数')
    ax.set_xticks(x_positions, config['communities'])
    ax.set_ylim(0, population_year5_totals.max() * 1.20)
    ax.grid(axis='y', color='#D1D5DB', linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()

    figure_path = FIGURE_DIR / 'T1_03_population_by_community.png'
    fig.savefig(figure_path, bbox_inches='tight')
    plt.close(fig)
    print(f'OK {figure_path.name}')
    return figure_path


def save_demand_before_after_constraint_png(config, theoretical_demand, constrained_demand):
    """Write the year-5 service demand comparison before and after constraint."""
    configure_plot_fonts()
    theoretical_totals = theoretical_demand.sum(axis=0)
    constrained_totals = constrained_demand.sum(axis=0)
    reduction_rates = np.divide(
        theoretical_totals - constrained_totals,
        theoretical_totals,
        out=np.zeros_like(theoretical_totals, dtype=float),
        where=theoretical_totals > 0
    ) * 100
    x_positions = np.arange(len(config['service_names']))
    width = 0.36

    fig, ax = plt.subplots(figsize=(11.2, 6.0), dpi=180)
    theoretical_bars = ax.bar(
        x_positions - width / 2,
        theoretical_totals,
        width=width,
        color='#2563EB',
        label='理论月需求'
    )
    constrained_bars = ax.bar(
        x_positions + width / 2,
        constrained_totals,
        width=width,
        color='#E11D48',
        label='消费约束后月需求'
    )
    ax.bar_label(theoretical_bars, labels=[display_int(value) for value in theoretical_totals], fontsize=8, padding=5)
    ax.bar_label(constrained_bars, labels=[display_int(value) for value in constrained_totals], fontsize=8, padding=5)
    ax.set_ylabel('全区月需求量')
    ax.set_xticks(x_positions, config['service_names'])
    ax.set_ylim(0, theoretical_totals.max() * 1.30)
    ax.grid(axis='y', color='#D1D5DB', linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)

    rate_axis = ax.twinx()
    rate_axis.plot(
        x_positions,
        reduction_rates,
        color='#7C3AED',
        marker='o',
        linewidth=2.0,
        label='削减率'
    )
    rate_axis.set_ylabel('削减率 (%)')
    rate_axis.set_ylim(0, max(5.0, reduction_rates.max() * 1.40))
    rate_offsets = [(0, 12), (0, 12), (0, 10), (0, 10), (0, 10), (0, 12)]
    for x_position, reduction_rate, offset in zip(x_positions, reduction_rates, rate_offsets):
        rate_axis.annotate(
            f'{reduction_rate:.1f}%',
            (x_position, reduction_rate),
            textcoords='offset points',
            xytext=offset,
            ha='center',
            fontsize=8,
            color='#5B21B6',
            bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.72)
        )

    handles, legend_labels = ax.get_legend_handles_labels()
    rate_handles, rate_labels = rate_axis.get_legend_handles_labels()
    ax.legend(
        handles + rate_handles,
        legend_labels + rate_labels,
        frameon=False,
        ncol=3,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.14)
    )
    fig.tight_layout()

    figure_path = FIGURE_DIR / 'T1_04_demand_before_after_constraint.png'
    fig.savefig(figure_path, bbox_inches='tight')
    plt.close(fig)
    print(f'OK {figure_path.name}')
    return figure_path


def save_constraint_impact_by_elderly_type_png(config, reduction_coefficient):
    """Write the consumption constraint impact chart split by elderly type."""
    configure_plot_fonts()
    labels = ['自理', '半失能', '失能']
    x_positions = np.arange(len(config['communities']))
    color_map = plt.get_cmap('RdYlGn')

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.8), dpi=180, sharey=True)
    for idx, (axis, label) in enumerate(zip(axes, labels)):
        values = reduction_coefficient[:, idx]
        bars = axis.bar(
            x_positions,
            values,
            color=color_map(values),
            width=0.68,
            edgecolor='#374151',
            linewidth=0.4
        )
        axis.set_title(label, fontsize=12, pad=10)
        axis.set_xticks(x_positions, config['communities'], rotation=25, ha='right')
        axis.set_ylim(0, 1.08)
        axis.grid(axis='y', color='#D1D5DB', linewidth=0.8, alpha=0.7)
        axis.set_axisbelow(True)
        for bar, value in zip(bars, values):
            if value < 1 - 1e-9:
                axis.annotate(
                    f'削减{(1 - value) * 100:.1f}%',
                    (bar.get_x() + bar.get_width() / 2, value),
                    textcoords='offset points',
                    xytext=(0, 6),
                    ha='center',
                    fontsize=7,
                    color='#991B1B'
                )
        if idx == 0:
            axis.set_ylabel('削减系数')

    fig.suptitle('T1 消费约束影响分析：按老人类型', fontsize=14, y=1.02)
    fig.tight_layout()

    figure_path = FIGURE_DIR / 'T1_05_constraint_impact_by_elderly_type.png'
    fig.savefig(figure_path, bbox_inches='tight')
    plt.close(fig)
    print(f'OK {figure_path.name}')
    return figure_path


def main():
    config = load_t0_config(T0_CONFIG_PATH)
    population_year5 = np.load(YEAR5_POPULATION_PATH)
    theoretical_demand = np.load(THEORETICAL_DEMAND_PATH)
    constrained_demand = np.load(CONSTRAINED_MONTHLY_DEMAND_PATH)
    reduction_coefficient = np.load(REDUCTION_COEFFICIENT_PATH)

    with FORECAST_HISTORY_PATH.open('rb') as history_file:
        forecast_history = pickle.load(history_file)
    with THEORETICAL_DEMAND_BY_TYPE_PATH.open('rb') as theoretical_file:
        theoretical_by_type = pickle.load(theoretical_file)
    with CONSTRAINED_MONTHLY_DEMAND_BY_TYPE_PATH.open('rb') as constrained_file:
        constrained_by_type = pickle.load(constrained_file)

    TABLE_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)

    service_headers = ['小区', *config['service_names'], '合计']
    type_headers = ['小区', '老人类型', *config['service_names'], '合计']
    outputs = {
        'T1_01_population_forecast_year1_to_year5.tex': build_population_history_table(config, forecast_history),
        'T1_02_population_year0_vs_year5.tex': build_population_comparison_table(config, population_year5),
        'T1_03_theoretical_demand_by_service.tex': latex_table(
            '问题1.2：第5年末各小区理论月需求',
            service_headers,
            service_rows(config['communities'], config['service_names'], theoretical_demand)
        ),
        'T1_04_theoretical_demand_by_elderly_type.tex': latex_table(
            '问题1.2：第5年末理论月需求老人类型拆分',
            type_headers,
            type_detail_rows(config['communities'], theoretical_by_type)
        ),
        'T1_05_reduction_coefficients.tex': build_reduction_table(config, reduction_coefficient),
        'T1_06_constrained_demand_by_service.tex': latex_table(
            '问题1.3：第5年末消费约束后月需求',
            service_headers,
            service_rows(config['communities'], config['service_names'], constrained_demand)
        ),
        'T1_07_constrained_demand_by_elderly_type.tex': latex_table(
            '问题1.3：第5年末消费约束后月需求老人类型拆分',
            type_headers,
            type_detail_rows(config['communities'], constrained_by_type)
        ),
        'T1_08_demand_comparison.tex': build_demand_comparison_table(
            config,
            theoretical_demand,
            constrained_demand
        )
    }

    print('========== T1 问题 1 表格与图表生成 ==========')
    for filename, content in outputs.items():
        save_text(TABLE_DIR / filename, content)
    figure_paths = [
        save_population_trend_png(config, forecast_history),
        save_demand_structure_png(config, theoretical_demand, constrained_demand),
        save_elderly_structure_comparison_png(config, population_year5),
        save_population_by_community_png(config, population_year5),
        save_demand_before_after_constraint_png(config, theoretical_demand, constrained_demand),
        save_constraint_impact_by_elderly_type_png(config, reduction_coefficient)
    ]
    print('图表输出:')
    for figure_path in figure_paths:
        print(f'  {figure_path}')
    print('============================================')


if __name__ == '__main__':
    main()
