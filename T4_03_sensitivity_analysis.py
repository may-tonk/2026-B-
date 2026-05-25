import pickle
import sys
from pathlib import Path

import numpy as np


MODEL_DIR = Path("model")
SCENARIO_FILES = {
    "Baseline": MODEL_DIR / "T4_02_scenario_baseline_results.pkl",
    "Scenario_A": MODEL_DIR / "T4_02_scenario_A_results.pkl",
    "Scenario_B": MODEL_DIR / "T4_02_scenario_B_results.pkl",
    "Scenario_C": MODEL_DIR / "T4_02_scenario_C_results.pkl",
}


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def station_names(config, active_stations):
    return [config["communities"][idx] for idx in active_stations]


def station_scales(layout, active_stations):
    scale_map = {1: "小型", 2: "中型", 3: "大型"}
    return [scale_map.get(int(layout[idx]), "--") for idx in active_stations]


def average_prices_by_service(config, q3):
    station_prices = q3.get("station_prices", {})
    service_names = config["service_names"]
    if not station_prices:
        return {service: 0.0 for service in service_names}
    result = {}
    for service in service_names:
        values = [float(prices.get(service, 0.0)) for prices in station_prices.values()]
        result[service] = float(np.mean(values)) if values else 0.0
    return result


def station_margin_metrics(q3):
    station_financials = q3.get("station_financials", {})
    margins = {
        station_idx: float(financials.get("profit_margin", 0.0))
        for station_idx, financials in station_financials.items()
    }
    if not margins:
        margins = {
            station_idx: float(margin)
            for station_idx, margin in q3.get("station_profit_margins", {}).items()
        }
    if not margins:
        return {
            "station_profit_margins": {},
            "min_station_profit_margin": float(q3.get("min_station_profit_margin", 0.0)),
            "max_station_profit_margin": float(q3.get("max_station_profit_margin", 0.0)),
            "station_margin_feasible": bool(q3.get("station_margin_feasible", False)),
        }
    min_margin = min(margins.values())
    max_margin = max(margins.values())
    return {
        "station_profit_margins": margins,
        "min_station_profit_margin": float(q3.get("min_station_profit_margin", min_margin)),
        "max_station_profit_margin": float(q3.get("max_station_profit_margin", max_margin)),
        "station_margin_feasible": bool(
            q3.get("station_margin_feasible", min_margin >= 0.0 and max_margin <= 8.0)
        ),
    }


def extract_metrics(config, result):
    q1 = result["q1"]
    q2 = result["q2"]
    q3 = result["q3"]
    sys_fin = q3.get("system_financials", {})
    unmet = q2.get("unmet_demand", {})
    capacity_audit = q2.get("capacity_audit", q2.get("standard_capacity_audit", {}))
    margin_metrics = station_margin_metrics(q3)

    active = list(q2.get("active_stations", []))
    max_utilization = max(
        (audit.get("utilization_rate", 0.0) for audit in capacity_audit.values()),
        default=0.0,
    )
    max_overload = max(
        (audit.get("overload_rate", 0.0) for audit in capacity_audit.values()),
        default=0.0,
    )

    return {
        "scenario": result.get("scenario", ""),
        "description": result.get("description", ""),
        "pop_year5": float(np.asarray(q1["population_year5"]).sum()),
        "demand_daily": float(np.asarray(q1["constrained_demand_daily"]).sum()),
        "demand_monthly": float(np.asarray(q1["constrained_demand_daily"]).sum() * 30.0),
        "station_count": len(active),
        "active_stations": active,
        "station_locations": station_names(config, active),
        "station_scales": station_scales(q2.get("layout", []), active),
        "layout": q2.get("layout", []),
        "budget_limit": float(result.get("budget_limit", 0.0)),
        "build_cost": float(q2.get("build_cost", 0.0)),
        "coverage_rate": float(q2.get("coverage_rate", 0.0)),
        "q2_avg_satisfaction_covered": float(q2.get("avg_satisfaction_covered", 0.0)),
        "q2_avg_satisfaction_system": float(q2.get("avg_satisfaction_system", 0.0)),
        "avg_satisfaction_covered": float(q3.get("avg_satisfaction_covered", 0.0)),
        "avg_satisfaction_system": float(q3.get("avg_satisfaction_system", 0.0)),
        "service_prices": average_prices_by_service(config, q3),
        "annual_subsidy": float(q3.get("annual_subsidy", sys_fin.get("annual_subsidy", 0.0))),
        "annual_profit": float(q3.get("annual_profit", sys_fin.get("annual_profit", 0.0))),
        "profit_margin": float(q3.get("profit_margin", sys_fin.get("profit_margin", 0.0))),
        "min_station_profit_margin": margin_metrics["min_station_profit_margin"],
        "max_station_profit_margin": margin_metrics["max_station_profit_margin"],
        "station_margin_feasible": margin_metrics["station_margin_feasible"],
        "station_profit_margins": margin_metrics["station_profit_margins"],
        "annual_service_revenue": float(sys_fin.get("annual_service_revenue", 0.0)),
        "annual_variable_cost": float(sys_fin.get("annual_variable_cost", 0.0)),
        "annual_operating_cost": float(sys_fin.get("annual_operating_cost", 0.0)),
        "uncovered_communities": q2.get("uncovered_communities", []),
        "uncovered_community_names": unmet.get("uncovered_communities", []),
        "uncovered_elderly": float(q2.get("uncovered_elderly", 0.0)),
        "uncovered_annual_constrained_demand": float(unmet.get("uncovered_annual_constrained_demand", 0.0)),
        "coverage_gap_pct": float(unmet.get("coverage_gap_pct", (1.0 - q2.get("coverage_rate", 0.0)) * 100.0)),
        "max_capacity_utilization": float(max_utilization),
        "max_standard_overload_rate": float(max_overload),
        "degradation_triggered": bool(q2.get("degradation_triggered", False)),
        "pricing_search_mode": q3.get("pricing_search_mode", ""),
        "candidate_count": q3.get("candidate_count", 0),
        "feasible_count": q3.get("feasible_count", 0),
    }


def pct_change(base, current):
    if abs(base) < 1e-12:
        return np.nan
    return (current - base) / abs(base) * 100.0


def sensitivity(base, current, input_delta_pct):
    change = pct_change(base, current)
    if np.isnan(change) or abs(input_delta_pct) < 1e-12:
        return np.nan
    return change / input_delta_pct


def build_sensitivity_coefficients(metrics):
    base = metrics["Baseline"]
    input_deltas = {
        "Scenario_A": 14.29,
        "Scenario_B": 20.00,
        "Scenario_C": 16.67,
    }
    metric_map = {
        "第5年老人总数": "pop_year5",
        "消费约束后月需求": "demand_monthly",
        "年政府补贴总额": "annual_subsidy",
        "年利润": "annual_profit",
        "最低站点利润率": "min_station_profit_margin",
        "最高站点利润率": "max_station_profit_margin",
        "覆盖内满意度": "avg_satisfaction_covered",
        "全区域有效满意度": "avg_satisfaction_system",
        "未满足年需求": "uncovered_annual_constrained_demand",
        "最大容量利用率": "max_capacity_utilization",
    }
    coeffs = {}
    changes = {}
    for label, key in metric_map.items():
        coeffs[label] = {}
        changes[label] = {}
        for scenario, input_delta in input_deltas.items():
            coeffs[label][scenario] = sensitivity(base[key], metrics[scenario][key], input_delta)
            changes[label][scenario] = pct_change(base[key], metrics[scenario][key])
    coverage_pp = {
        scenario: (metrics[scenario]["coverage_rate"] - base["coverage_rate"]) * 100.0
        for scenario in input_deltas
    }
    return coeffs, changes, coverage_pp


def print_summary(config, metrics, coeffs, coverage_pp):
    print("\n【PDF 4.2 指标对比】")
    print(f"{'指标':<18} {'Baseline':>14} {'A':>14} {'B':>14} {'C':>14}")
    print("-" * 78)
    rows = [
        ("站点数量", "station_count", "{:.0f}"),
        ("建设成本(万元)", "build_cost", "{:.1f}"),
        ("覆盖率(%)", "coverage_rate", "{:.2f}", 100.0),
        ("覆盖内满意度", "avg_satisfaction_covered", "{:.4f}"),
        ("全区域满意度", "avg_satisfaction_system", "{:.4f}"),
        ("年政府补贴(万元)", "annual_subsidy", "{:.2f}"),
        ("年利润(万元)", "annual_profit", "{:.4f}"),
        ("全区利润率(%)", "profit_margin", "{:.4f}"),
        ("最低站点利润率(%)", "min_station_profit_margin", "{:.4f}"),
        ("最高站点利润率(%)", "max_station_profit_margin", "{:.4f}"),
        ("未覆盖老人", "uncovered_elderly", "{:.0f}"),
        ("未满足年需求", "uncovered_annual_constrained_demand", "{:.2f}"),
    ]
    for row in rows:
        label, key, fmt = row[:3]
        multiplier = row[3] if len(row) > 3 else 1.0
        values = [metrics[name][key] * multiplier for name in ["Baseline", "Scenario_A", "Scenario_B", "Scenario_C"]]
        print(f"{label:<18} {fmt.format(values[0]):>14} {fmt.format(values[1]):>14} {fmt.format(values[2]):>14} {fmt.format(values[3]):>14}")

    feasibility = [
        "是" if metrics[name]["station_margin_feasible"] else "否"
        for name in ["Baseline", "Scenario_A", "Scenario_B", "Scenario_C"]
    ]
    print(f"{'逐站利润率可行':<18} {feasibility[0]:>14} {feasibility[1]:>14} {feasibility[2]:>14} {feasibility[3]:>14}")

    print("\n【站点位置与规模】")
    for name in ["Baseline", "Scenario_A", "Scenario_B", "Scenario_C"]:
        m = metrics[name]
        stations = [
            f"{loc}({scale})"
            for loc, scale in zip(m["station_locations"], m["station_scales"])
        ]
        print(f"{name}: {', '.join(stations)}")

    print("\n【平均服务定价】")
    print(f"{'服务':<12} {'Baseline':>12} {'A':>12} {'B':>12} {'C':>12}")
    for service in config["service_names"]:
        values = [metrics[name]["service_prices"][service] for name in ["Baseline", "Scenario_A", "Scenario_B", "Scenario_C"]]
        print(f"{service:<12} {values[0]:>10.2f}元 {values[1]:>10.2f}元 {values[2]:>10.2f}元 {values[3]:>10.2f}元")

    print("\n【敏感度系数】")
    for label, values in coeffs.items():
        parts = []
        for scenario in ["Scenario_A", "Scenario_B", "Scenario_C"]:
            value = values[scenario]
            parts.append("N/A" if np.isnan(value) else f"{value:+.2f}")
        print(f"{label:<18} A={parts[0]:>8} B={parts[1]:>8} C={parts[2]:>8}")
    print("\n覆盖率变化使用百分点报告：")
    for scenario, value in coverage_pp.items():
        print(f"  {scenario}: {value:+.2f} percentage points")


def main():
    print("========== T4 步骤3：敏感度分析与方案比较 ==========")
    missing = [str(path) for path in SCENARIO_FILES.values() if not path.exists()]
    if missing:
        print("[错误] 缺少 T4_02 结果文件：")
        for path in missing:
            print(f"  - {path}")
        sys.exit(1)

    config = load_pickle(MODEL_DIR / "T0_01_config.pkl")
    scenario_results = {name: load_pickle(path) for name, path in SCENARIO_FILES.items()}
    metrics = {name: extract_metrics(config, result) for name, result in scenario_results.items()}
    coeffs, changes, coverage_pp = build_sensitivity_coefficients(metrics)

    output = {
        **metrics,
        "scenario_results": scenario_results,
        "sensitivity_coefficients": coeffs,
        "relative_changes_pct": changes,
        "coverage_change_pp": coverage_pp,
    }
    with open(MODEL_DIR / "T4_03_sensitivity_metrics.pkl", "wb") as f:
        pickle.dump(output, f)

    print_summary(config, metrics, coeffs, coverage_pp)
    print("\nT4_03 完成：已保存 model/T4_03_sensitivity_metrics.pkl")


if __name__ == "__main__":
    main()
