import os
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
import numpy as np
import pandas as pd


MODEL_DIR = Path("model")
TABLE_DIR = Path("T4_05_table")
FIGURE_DIR = Path("T4_05_figure")
SCENARIOS = ["Baseline", "Scenario_A", "Scenario_B", "Scenario_C"]
SCENARIO_LABELS = {
    "Baseline": "基准场景",
    "Scenario_A": "场景A",
    "Scenario_B": "场景B",
    "Scenario_C": "场景C",
}


plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def save_latex(df, filename, caption):
    TABLE_DIR.mkdir(exist_ok=True)
    path = TABLE_DIR / filename
    columns = list(df.columns)
    align = "|" + "|".join(["c"] * len(columns)) + "|"
    rows = [
        " & ".join(str(value) for value in row) + r" \\"
        for row in df.astype(str).itertuples(index=False, name=None)
    ]
    latex = (
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\begin{{tabular}}{{{align}}}\n"
        "\\hline\n"
        + " & ".join(columns) + r" \\"
        + "\n\\hline\n"
        + "\n".join(rows)
        + "\n\\hline\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )
    path.write_text(latex, encoding="utf-8")


def fmt(value, digits=2):
    if value is None:
        return "--"
    if isinstance(value, float) and np.isnan(value):
        return "--"
    return f"{value:.{digits}f}"


def scale_name(scale):
    return {0: "--", 1: "小型", 2: "中型", 3: "大型"}.get(int(scale), "--")


def station_name(config, idx):
    return config["communities"][idx]


def generate_tables(config, data, report):
    metrics = {name: data[name] for name in SCENARIOS}
    scenario_results = data["scenario_results"]

    table1 = pd.DataFrame([
        ["基准场景", "7.0%", "4.5%", "10.0%", "不变", "120"],
        ["场景A", "8.0%", "5.5%", "9.5%", "不变", "120"],
        ["场景B", "7.0%", "4.5%", "10.0%", "+20%", "120"],
        ["场景C", "7.0%", "4.5%", "10.0%", "不变", "140"],
    ], columns=["场景", "老人增长率", "自理转半失能", "半失能转失能", "日固定管理成本", "建设预算(万元)"])
    save_latex(table1, "T4_01_table_scenario_parameters.tex", "问题4.1 场景参数设置")

    all_stations = sorted(set().union(*(set(metrics[name]["active_stations"]) for name in SCENARIOS)))
    layout_rows = []
    for idx in all_stations:
        row = {"站点": station_name(config, idx)}
        for name in SCENARIOS:
            layout = metrics[name]["layout"]
            row[SCENARIO_LABELS[name]] = scale_name(layout[idx]) if idx < len(layout) else "--"
        layout_rows.append(row)
    save_latex(pd.DataFrame(layout_rows), "T4_02_table_station_layout.tex", "问题4.2 各场景站点位置与规模对比")

    price_rows = []
    for service in config["service_names"]:
        row = {"服务项目": service}
        for name in SCENARIOS:
            row[SCENARIO_LABELS[name]] = f"{metrics[name]['service_prices'].get(service, 0.0):.2f}"
        price_rows.append(row)
    save_latex(pd.DataFrame(price_rows), "T4_03_table_pricing_comparison.tex", "问题4.2 各场景平均服务定价对比")

    financial_rows = []
    financial_items = [
        ("年服务收入(万元)", "annual_service_revenue", 2),
        ("年政府补贴(万元)", "annual_subsidy", 2),
        ("年直接成本(万元)", "annual_variable_cost", 2),
        ("年运营成本(万元)", "annual_operating_cost", 2),
        ("年利润(万元)", "annual_profit", 4),
        ("全区利润率(%)", "profit_margin", 4),
        ("最低站点利润率(%)", "min_station_profit_margin", 4),
        ("最高站点利润率(%)", "max_station_profit_margin", 4),
    ]
    for label, key, digits in financial_items:
        row = {"指标": label}
        for name in SCENARIOS:
            row[SCENARIO_LABELS[name]] = fmt(metrics[name][key], digits)
        financial_rows.append(row)
    feasible_row = {"指标": "逐站利润率约束"}
    for name in SCENARIOS:
        feasible_row[SCENARIO_LABELS[name]] = "满足" if metrics[name].get("station_margin_feasible") else "不满足"
    financial_rows.append(feasible_row)
    save_latex(pd.DataFrame(financial_rows), "T4_04_table_financial_comparison.tex", "问题4.2 财务与补贴结果对比")

    coverage_rows = []
    coverage_items = [
        ("覆盖率(%)", "coverage_rate", 100.0, 2),
        ("Q2覆盖内满意度", "q2_avg_satisfaction_covered", 1.0, 4),
        ("Q3覆盖内满意度", "avg_satisfaction_covered", 1.0, 4),
        ("全区域有效满意度", "avg_satisfaction_system", 1.0, 4),
        ("未覆盖老人(人)", "uncovered_elderly", 1.0, 0),
        ("未满足年需求(万人次)", "uncovered_annual_constrained_demand", 1.0, 2),
    ]
    for label, key, multiplier, digits in coverage_items:
        row = {"指标": label}
        for name in SCENARIOS:
            row[SCENARIO_LABELS[name]] = fmt(metrics[name][key] * multiplier, digits)
        coverage_rows.append(row)
    save_latex(pd.DataFrame(coverage_rows), "T4_05_table_coverage_satisfaction.tex", "问题4.2 覆盖率、满意度与需求缺口对比")

    capacity_rows = []
    for name in SCENARIOS:
        q2 = scenario_results[name]["q2"]
        for station_idx, audit in q2.get("capacity_audit", {}).items():
            layout = q2["layout"]
            capacity_rows.append({
                "场景": SCENARIO_LABELS[name],
                "站点": station_name(config, station_idx),
                "规模": scale_name(layout[station_idx]),
                "附件容量": fmt(audit["nominal_capacity"], 0),
                "2%弹性上限": fmt(audit["elastic_limit"], 0),
                "有效日负荷": fmt(audit["effective_load"], 1),
                "附件容量利用率(%)": fmt(audit["utilization_rate"] * 100, 2),
                "附件容量超载率(%)": fmt(audit["overload_rate"] * 100, 2),
                "弹性可行": "是" if audit["elastic_feasible"] else "否",
            })
    save_latex(pd.DataFrame(capacity_rows), "T4_06_table_capacity_audit.tex", "各场景标准容量审计")

    coeff_rows = []
    coeffs = data.get("sensitivity_coefficients", {})
    for label, values in coeffs.items():
        coeff_rows.append({
            "指标": label,
            "场景A": fmt(values.get("Scenario_A"), 3),
            "场景B": fmt(values.get("Scenario_B"), 3),
            "场景C": fmt(values.get("Scenario_C"), 3),
        })
    coverage_pp = data.get("coverage_change_pp", {})
    coeff_rows.append({
        "指标": "覆盖率变化(百分点)",
        "场景A": fmt(coverage_pp.get("Scenario_A"), 2),
        "场景B": fmt(coverage_pp.get("Scenario_B"), 2),
        "场景C": fmt(coverage_pp.get("Scenario_C"), 2),
    })
    save_latex(pd.DataFrame(coeff_rows), "T4_07_table_sensitivity_coefficients.tex", "问题4.2 参数敏感度系数")

    uncertainty_rows = []
    for idx, factor in enumerate(report.get("uncertainty_factors", []), 1):
        uncertainty_rows.append({
            "序号": idx,
            "不确定因素": factor["factor"],
            "影响变量": "、".join(factor["affected_variables"]),
            "应对策略": factor["strategy"],
        })
    save_latex(pd.DataFrame(uncertainty_rows), "T4_08_table_uncertainty_factors.tex", "问题4.3 实际推广不确定因素与应对策略")


def generate_charts(config, data, report):
    FIGURE_DIR.mkdir(exist_ok=True)
    metrics = {name: data[name] for name in SCENARIOS}

    labels = [SCENARIO_LABELS[name] for name in SCENARIOS]
    colors = ["#4C78A8", "#E45756", "#59A14F", "#F28E2B"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=220)
    pop_values = [metrics[name]["pop_year5"] for name in SCENARIOS]
    demand_values = [metrics[name]["demand_daily"] for name in SCENARIOS]
    bars = axes[0].bar(labels, pop_values, color=colors)
    axes[0].set_ylabel("第5年老人数")
    axes[0].set_title("第5年老人数")
    axes[0].set_ylim(min(pop_values) * 0.96, max(pop_values) * 1.02)
    for bar, value in zip(bars, pop_values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.0f}", ha="center", va="bottom", fontsize=9)
    bars = axes[1].bar(labels, demand_values, color=colors)
    axes[1].set_ylabel("消费约束后日需求(人次/日)")
    axes[1].set_title("消费约束后日需求")
    axes[1].set_ylim(min(demand_values) * 0.96, max(demand_values) * 1.02)
    for bar, value in zip(bars, demand_values):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.0f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("人口压力与需求规模对比", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "T4_01_population_trend_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    layout_matrix = np.array([metrics[name]["layout"] for name in SCENARIOS], dtype=int)
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=220)
    cmap = ListedColormap(["#F7F7F7", "#A1D99B", "#6BAED6", "#08519C"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    im = ax.imshow(layout_matrix, aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks(range(len(SCENARIOS)), labels)
    ax.set_xticks(range(config["n_communities"]), config["communities"])
    cell_labels = {0: "未", 1: "小", 2: "中", 3: "大"}
    for i in range(layout_matrix.shape[0]):
        for j in range(layout_matrix.shape[1]):
            value = int(layout_matrix[i, j])
            text_color = "white" if value == 3 else "black"
            ax.text(j, i, cell_labels[value], ha="center", va="center", color=text_color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.ax.set_yticklabels(["未建", "小型", "中型", "大型"])
    ax.set_title("各场景站点布局与规模对比")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "T4_02_station_layout_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    paid_services = config["service_names"][:5]
    price_matrix = np.array([
        [metrics[name]["service_prices"][service] for service in paid_services]
        for name in SCENARIOS
    ])
    baseline_prices = price_matrix[0]
    relative_change = (price_matrix - baseline_prices) / np.maximum(np.abs(baseline_prices), 1e-9) * 100
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=220)
    max_abs_change = max(1.0, float(np.max(np.abs(relative_change))))
    im = ax.imshow(
        relative_change,
        aspect="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0, vmin=-max_abs_change, vmax=max_abs_change),
    )
    ax.set_yticks(range(len(SCENARIOS)), labels)
    ax.set_xticks(range(len(paid_services)), paid_services, rotation=20)
    fig.colorbar(im, ax=ax, label="相对基准变化(%)")
    ax.set_title("各场景服务定价相对基准变化")
    for i in range(relative_change.shape[0]):
        for j in range(relative_change.shape[1]):
            ax.text(j, i, f"{relative_change[i, j]:+.2f}%", ha="center", va="center", color="black", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "T4_03_pricing_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    x = np.arange(len(SCENARIOS))
    subsidies = [metrics[name]["annual_subsidy"] for name in SCENARIOS]
    profits = [metrics[name]["annual_profit"] for name in SCENARIOS]
    margins = [metrics[name]["profit_margin"] for name in SCENARIOS]
    min_station_margins = [metrics[name].get("min_station_profit_margin", margins[idx]) for idx, name in enumerate(SCENARIOS)]
    max_station_margins = [metrics[name].get("max_station_profit_margin", margins[idx]) for idx, name in enumerate(SCENARIOS)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=220)
    bars = axes[0].bar(labels, subsidies, color="#4C78A8")
    axes[0].set_ylabel("年政府补贴(万元)")
    axes[0].set_title("政府补贴规模")
    axes[0].set_ylim(0, max(subsidies) * 1.18 if max(subsidies) > 0 else 1)
    for bar, value in zip(bars, subsidies):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    bars = axes[1].bar(labels, profits, color="#F28E2B")
    axes[1].set_ylabel("年利润(万元)")
    axes[1].set_title("利润与利润率")
    profit_upper = max(max(profits) * 1.45, 0.001)
    axes[1].set_ylim(0, profit_upper)
    axes_margin = axes[1].twinx()
    axes_margin.plot(labels, margins, color="#E45756", marker="o", linewidth=2, label="全区利润率")
    axes_margin.vlines(
        labels,
        min_station_margins,
        max_station_margins,
        color="#374151",
        linewidth=2,
        label="逐站利润率范围",
    )
    axes_margin.set_ylabel("利润率(%)")
    margin_upper = max(max(max_station_margins) * 1.60, 0.0003)
    axes_margin.set_ylim(0, margin_upper)
    for bar, value in zip(bars, profits):
        axes[1].text(bar.get_x() + bar.get_width() / 2, max(value, 0), f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    for idx, value in enumerate(margins):
        axes_margin.text(idx, value, f"{value:.4f}%", ha="center", va="bottom", fontsize=8, color="#E45756")
    axes_margin.legend(loc="upper right", fontsize=8)
    fig.suptitle("财务可持续性对比", y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "T4_04_financial_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    base = metrics["Baseline"]
    coverage_pp = data.get("coverage_change_pp", {})

    def pct_from_base(name, key):
        denom = abs(base[key])
        return 0.0 if denom < 1e-12 else (metrics[name][key] - base[key]) / denom * 100.0

    def average_price_change(name):
        changes = []
        for service, base_price in base["service_prices"].items():
            if service == config["service_names"][5] or abs(base_price) < 1e-12:
                continue
            current = metrics[name]["service_prices"].get(service, 0.0)
            changes.append((current - base_price) / abs(base_price) * 100.0)
        return float(np.mean(changes)) if changes else 0.0

    scenario_names = SCENARIOS[1:]
    sensitivity_df = pd.DataFrame({
        "老人规模变化(%)": [pct_from_base(name, "pop_year5") for name in scenario_names],
        "覆盖率变化(百分点)": [coverage_pp.get(name, 0.0) for name in scenario_names],
        "未满足年需求(万人次)": [metrics[name]["uncovered_annual_constrained_demand"] for name in scenario_names],
        "运营成本变化(%)": [pct_from_base(name, "annual_operating_cost") for name in scenario_names],
        "预算余量增加(万元)": [
            (metrics[name]["budget_limit"] - metrics[name]["build_cost"]) - (base["budget_limit"] - base["build_cost"])
            for name in scenario_names
        ],
        "平均定价变化(%)": [average_price_change(name) for name in scenario_names],
    }, index=["场景A", "场景B", "场景C"])

    heatmap_data = sensitivity_df.T
    raw_values = heatmap_data.values.astype(float)
    row_scales = np.max(np.abs(raw_values), axis=1)
    row_scales[row_scales < 1e-12] = 1.0
    color_values = raw_values / row_scales[:, None]

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=220)
    im = ax.imshow(color_values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(heatmap_data.columns)), heatmap_data.columns)
    ax.set_yticks(np.arange(len(heatmap_data.index)), heatmap_data.index)
    ax.tick_params(axis="x", top=False, bottom=True, labeltop=False, labelbottom=True)

    for row_idx in range(raw_values.shape[0]):
        for col_idx in range(raw_values.shape[1]):
            value = raw_values[row_idx, col_idx]
            label = "0.00" if abs(value) < 0.005 else f"{value:.2f}"
            text_color = "white" if abs(color_values[row_idx, col_idx]) > 0.55 else "#111827"
            ax.text(
                col_idx,
                row_idx,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
                fontweight="bold",
            )

    ax.set_xticks(np.arange(-0.5, len(heatmap_data.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(heatmap_data.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.set_label("行内相对变化强度")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "T4_05_sensitivity_bar_chart.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 6), dpi=220)
    offsets = {
        "Baseline": (0.08, 0.004),
        "Scenario_A": (0.08, 0.004),
        "Scenario_B": (0.08, -0.006),
        "Scenario_C": (-0.70, 0.006),
    }
    for color, name in zip(colors, SCENARIOS):
        m = metrics[name]
        unmet = m["uncovered_annual_constrained_demand"]
        size = 120 + max(unmet, 0.0) * 30
        ax.scatter(
            m["coverage_rate"] * 100,
            m["avg_satisfaction_system"],
            s=size,
            color=color,
            alpha=0.78,
            edgecolor="black",
            label=SCENARIO_LABELS[name],
        )
        dx, dy = offsets[name]
        short_label = "基准" if name == "Baseline" else name.replace("Scenario_", "")
        ax.text(m["coverage_rate"] * 100 + dx, m["avg_satisfaction_system"] + dy, short_label, fontsize=9)
    ax.set_xlabel("覆盖率(%)")
    ax.set_ylabel("全区域有效满意度")
    ax.set_title("覆盖率-满意度-需求缺口散点图")
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(title="气泡大小=未满足年需求", loc="lower right")
    ax.set_xlim(min(m["coverage_rate"] for m in metrics.values()) * 100 - 1.5, 101)
    ax.set_ylim(max(0.0, min(m["avg_satisfaction_system"] for m in metrics.values()) - 0.025), max(m["avg_satisfaction_system"] for m in metrics.values()) + 0.02)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "T4_06_coverage_satisfaction_gap_scatter.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    print("========== T4 步骤5：输出论文表格与图表 ==========")
    metrics_path = MODEL_DIR / "T4_03_sensitivity_metrics.pkl"
    report_path = MODEL_DIR / "T4_04_robustness_report.pkl"
    if not metrics_path.exists() or not report_path.exists():
        print("[错误] 缺少 T4_03 或 T4_04 产物，请先运行前置脚本。")
        return

    TABLE_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(exist_ok=True)
    config = load_pickle(MODEL_DIR / "T0_01_config.pkl")
    data = load_pickle(metrics_path)
    report = load_pickle(report_path)

    generate_tables(config, data, report)
    generate_charts(config, data, report)

    print("已生成表格：")
    for path in sorted(TABLE_DIR.glob("T4_*.tex")):
        print(f"  - {path}")
    print("已生成图表：")
    for path in sorted(FIGURE_DIR.glob("T4_*.png")):
        print(f"  - {path}")
    print("T4_05 完成。")


if __name__ == "__main__":
    main()
