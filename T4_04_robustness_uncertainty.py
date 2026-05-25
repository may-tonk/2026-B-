import pickle
from pathlib import Path

import numpy as np


MODEL_DIR = Path("model")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def jaccard_similarity(left, right):
    left = set(left)
    right = set(right)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def rating_from_threshold(value, high, medium, reverse=False):
    if reverse:
        if value <= high:
            return "H"
        if value <= medium:
            return "M"
        return "L"
    if value >= high:
        return "H"
    if value >= medium:
        return "M"
    return "L"


def price_vector(metrics):
    prices = metrics.get("service_prices", {})
    return np.array([prices.get(name, 0.0) for name in sorted(prices)], dtype=float)


def price_change_rate(base, current):
    base_vec = price_vector(base)
    cur_vec = price_vector(current)
    denom = np.maximum(np.abs(base_vec), 1e-9)
    return float(np.mean(np.abs(cur_vec - base_vec) / denom))


def evaluate_by_dimension(metrics):
    base = metrics["Baseline"]
    scenarios = ["Scenario_A", "Scenario_B", "Scenario_C"]

    jaccards = {
        scenario: jaccard_similarity(base["active_stations"], metrics[scenario]["active_stations"])
        for scenario in scenarios
    }
    coverage_changes = {
        scenario: (metrics[scenario]["coverage_rate"] - base["coverage_rate"]) * 100.0
        for scenario in scenarios
    }
    system_sat_changes = {
        scenario: metrics[scenario]["avg_satisfaction_system"] - base["avg_satisfaction_system"]
        for scenario in scenarios
    }
    price_changes = {
        scenario: price_change_rate(base, metrics[scenario])
        for scenario in scenarios
    }
    min_station_margins = {
        scenario: metrics[scenario].get("min_station_profit_margin", metrics[scenario]["profit_margin"])
        for scenario in scenarios
    }
    max_station_margins = {
        scenario: metrics[scenario].get("max_station_profit_margin", metrics[scenario]["profit_margin"])
        for scenario in scenarios
    }
    station_margin_feasible = {
        scenario: bool(metrics[scenario].get("station_margin_feasible", False))
        for scenario in scenarios
    }
    max_overloads = {
        scenario: metrics[scenario]["max_standard_overload_rate"]
        for scenario in scenarios
    }
    unmet_demands = {
        scenario: metrics[scenario]["uncovered_annual_constrained_demand"]
        for scenario in scenarios
    }

    avg_jaccard = float(np.mean(list(jaccards.values())))
    min_coverage = min(metrics[scenario]["coverage_rate"] for scenario in scenarios)
    min_system_sat = min(metrics[scenario]["avg_satisfaction_system"] for scenario in scenarios)
    max_price_change = max(price_changes.values())
    min_station_margin = min(min_station_margins.values())
    max_station_margin = max(max_station_margins.values())
    all_station_margins_feasible = (
        all(station_margin_feasible.values())
        and min_station_margin >= 0.0
        and max_station_margin <= 8.0
    )
    max_unmet = max(unmet_demands.values())
    max_standard_overload = max(max_overloads.values())

    by_dimension = {
        "站点结构": {
            "rating": rating_from_threshold(avg_jaccard, 0.80, 0.60),
            "evidence": f"平均Jaccard相似度={avg_jaccard:.3f}; A/B/C={jaccards['Scenario_A']:.3f}/{jaccards['Scenario_B']:.3f}/{jaccards['Scenario_C']:.3f}",
        },
        "覆盖能力": {
            "rating": rating_from_threshold(min_coverage, 0.95, 0.90),
            "evidence": f"最低覆盖率={min_coverage*100:.2f}%; 覆盖率变化A/B/C={coverage_changes['Scenario_A']:+.2f}/{coverage_changes['Scenario_B']:+.2f}/{coverage_changes['Scenario_C']:+.2f} percentage points",
        },
        "满意度": {
            "rating": rating_from_threshold(min_system_sat, 0.85, 0.75),
            "evidence": f"最低全区域有效满意度={min_system_sat:.4f}; A/B/C变化={system_sat_changes['Scenario_A']:+.4f}/{system_sat_changes['Scenario_B']:+.4f}/{system_sat_changes['Scenario_C']:+.4f}",
        },
        "定价方案": {
            "rating": rating_from_threshold(max_price_change, 0.05, 0.12, reverse=True),
            "evidence": f"最大平均价格相对变化={max_price_change*100:.2f}%",
        },
        "财务可持续性": {
            "rating": "H" if all_station_margins_feasible and min_station_margin > 0.5 else ("M" if all_station_margins_feasible else "L"),
            "evidence": (
                f"逐站利润率范围=[{min_station_margin:.4f}%, {max_station_margin:.4f}%]，"
                f"逐站0%-8%约束{'全部满足' if all_station_margins_feasible else '存在不满足'}"
            ),
        },
        "需求缺口控制": {
            "rating": "H" if max_unmet <= 1e-9 else ("M" if max_unmet <= 25.0 else "L"),
            "evidence": f"最大未满足消费约束后年需求={max_unmet:.2f}万人次",
        },
        "容量审计": {
            "rating": "H" if max_standard_overload <= 0.02 + 1e-9 else ("M" if max_standard_overload <= 0.05 else "L"),
            "evidence": f"最大相对附件标准容量超载率={max_standard_overload*100:.2f}%，主口径为2%弹性容量",
        },
    }
    return {
        "by_dimension": by_dimension,
        "jaccard_similarity": jaccards,
        "coverage_changes": coverage_changes,
        "system_satisfaction_changes": system_sat_changes,
        "price_change_rates": price_changes,
        "min_station_profit_margins": min_station_margins,
        "max_station_profit_margins": max_station_margins,
        "station_margin_feasible": station_margin_feasible,
        "unmet_demands": unmet_demands,
        "max_standard_overload_rates": max_overloads,
    }


def evaluate_budget_fragility(metrics):
    base = metrics["Baseline"]
    scenario_a = metrics["Scenario_A"]
    scenario_c = metrics["Scenario_C"]
    coverage_loss_a = max(0.0, base["coverage_rate"] - scenario_a["coverage_rate"])
    coverage_gain_c = max(0.0, scenario_c["coverage_rate"] - base["coverage_rate"])

    if coverage_loss_a > 1e-9:
        fragility_rating = "中高脆弱" if coverage_loss_a >= 0.05 else "中脆弱"
        conclusion = (
            f"场景A覆盖率下降{coverage_loss_a*100:.2f}个百分点，说明120万元预算在老龄化压力下存在脆弱性。"
        )
    else:
        fragility_rating = "低脆弱"
        conclusion = "场景A覆盖率未下降，说明120万元预算在该扰动幅度下仍较稳健。"

    if coverage_gain_c <= 1e-9:
        budget_note = "场景C未带来覆盖率提升，不能仅用140万元场景外推预算缺口。"
        budget_needed = np.nan
    else:
        budget_needed = coverage_loss_a * 100.0 * (20.0 / (coverage_gain_c * 100.0))
        budget_note = f"按场景C边际覆盖改善粗略估计，恢复覆盖率约需追加{budget_needed:.1f}万元；该数值仅作辅助参考。"

    recommendations = [
        "将建设预算与60+老人增长率、失能率变化挂钩，定期重算站点布局。",
        "在人口压力场景下优先评估未覆盖小区的可达性与临近站点扩容可能。",
        "保持2%弹性容量作为短期排班调度缓冲，同时在论文中报告标准容量审计。",
        "若覆盖率下降，应优先追加站点建设或调整站点规模，而不是强行把超载布局作为可行方案。",
    ]
    return {
        "coverage_base": base["coverage_rate"],
        "coverage_scenario_A": scenario_a["coverage_rate"],
        "coverage_scenario_C": scenario_c["coverage_rate"],
        "coverage_loss_under_population_stress": coverage_loss_a,
        "coverage_gain_under_budget_relaxation": coverage_gain_c,
        "fragility_rating": fragility_rating,
        "budget_needed_to_restore": float(budget_needed) if not np.isnan(budget_needed) else np.nan,
        "conclusion": conclusion,
        "budget_note": budget_note,
        "policy_recommendation": recommendations,
    }


def build_uncertainty_factors():
    return [
        {
            "factor": "护理人员供给不足与人工成本非线性上涨",
            "affected_variables": ["日固定管理成本", "服务能力", "响应满意度"],
            "strategy": "建立人工成本压力测试，并预留人员补贴或灵活排班方案。",
        },
        {
            "factor": "老人服务需求存在季节性、节假日和突发健康事件波动",
            "affected_variables": ["日服务需求", "容量利用率", "未满足需求"],
            "strategy": "按月或周滚动更新需求预测，保留短期弹性服务资源。",
        },
        {
            "factor": "实际通行时间与表格距离不完全一致",
            "affected_variables": ["距离满意度", "覆盖关系", "站点归属"],
            "strategy": "用步行时间或路网距离替代表格直线距离，并定期校准覆盖矩阵。",
        },
        {
            "factor": "政府补贴标准或单站补贴上限调整",
            "affected_variables": ["政府补贴", "利润率", "最优定价"],
            "strategy": "把补贴金额和补贴上限作为政策参数，定期重算Q3定价。",
        },
        {
            "factor": "服务质量差异影响满意度和复购行为",
            "affected_variables": ["综合满意度", "实际有效服务人次", "收入"],
            "strategy": "引入服务质量评分和投诉率指标，扩展满意度模型。",
        },
    ]


def main():
    print("========== T4 步骤4：鲁棒性评价与不确定性分析 ==========")
    metrics_path = MODEL_DIR / "T4_03_sensitivity_metrics.pkl"
    if not metrics_path.exists():
        print("[错误] 缺少 T4_03_sensitivity_metrics.pkl，请先运行 T4_03。")
        return

    data = load_pickle(metrics_path)
    metrics = {name: data[name] for name in ["Baseline", "Scenario_A", "Scenario_B", "Scenario_C"]}
    robustness = evaluate_by_dimension(metrics)
    budget_fragility = evaluate_budget_fragility(metrics)
    uncertainty_factors = build_uncertainty_factors()

    report = {
        "robustness_evaluation": robustness,
        "budget_fragility": budget_fragility,
        "uncertainty_factors": uncertainty_factors,
    }
    with open(MODEL_DIR / "T4_04_robustness_report.pkl", "wb") as f:
        pickle.dump(report, f)

    print("\n【鲁棒性分维度评价】")
    for dimension, result in robustness["by_dimension"].items():
        print(f"{dimension}: {result['rating']} | {result['evidence']}")

    print("\n【预算脆弱性】")
    print(f"评级: {budget_fragility['fragility_rating']}")
    print(budget_fragility["conclusion"])
    print(budget_fragility["budget_note"])

    print("\n【实际推广不确定因素】")
    for idx, factor in enumerate(uncertainty_factors, 1):
        print(f"{idx}. {factor['factor']} -> {factor['strategy']}")

    print("\nT4_04 完成：已保存 model/T4_04_robustness_report.pkl")


if __name__ == "__main__":
    main()
