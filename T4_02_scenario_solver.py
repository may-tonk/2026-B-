import copy
import pickle
import sys
from pathlib import Path

import numpy as np

from T4_02_q1_demand import run_q1_pipeline
from T4_02_q2_layout import run_q2_pipeline
from T4_02_q3_pricing import run_q3_pipeline


MODEL_DIR = Path("model")


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def dump_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def scale_name(scale_code):
    return {0: "--", 1: "小型", 2: "中型", 3: "大型"}.get(int(scale_code), "--")


def station_label(config, station_idx):
    return config["communities"][station_idx]


def process_scenario(scenario_id, config, base_result, description, budget_limit):
    print(f"\n========== 场景 {scenario_id}: {description} ==========")
    pop_year5, constrained_daily, forecast_history = run_q1_pipeline(config)
    print(f"Q1: 第5年老人总数 {pop_year5.sum():.2f} 人，消费约束后日需求 {constrained_daily.sum():.2f} 人次/日。")

    q2, counts = run_q2_pipeline(config, pop_year5, constrained_daily, budget_limit)
    if q2 is None:
        raise RuntimeError(f"场景 {scenario_id} 未找到容量可行的 Q2 布局。")
    print(
        "Q2: 从 {budget:,} 个预算可行布局中得到 {capacity:,} 个容量可行布局，"
        "最优覆盖率 {coverage:.2f}%，站点 {stations}，建设成本 {cost:.1f} 万元。".format(
            budget=counts["budget_feasible_layouts"],
            capacity=counts["capacity_feasible_layouts"],
            coverage=q2["coverage_rate"] * 100,
            stations=[station_label(config, j) + scale_name(q2["layout"][j]) for j in q2["active_stations"]],
            cost=q2["build_cost"],
        )
    )
    print(
        f"    降级机制: {'触发' if q2['degradation_triggered'] else '未触发'}；"
        f"未覆盖小区 {q2['unmet_demand']['uncovered_communities']}。"
    )

    q3 = run_q3_pipeline(config, pop_year5, constrained_daily, q2)
    if q3 is None:
        raise RuntimeError(f"场景 {scenario_id} 未找到 Q3 定价结果。")
    station_margin_range = (
        q3.get("min_station_profit_margin", 0.0),
        q3.get("max_station_profit_margin", 0.0),
    )
    print(
        "Q3: {mode} 搜索 {cand:,} 个候选，逐站利润率可行 {feas:,} 个；"
        "年补贴 {subsidy:.2f} 万元，年利润 {profit:.4f} 万元，"
        "全区利润率 {margin:.4f}%，逐站利润率范围 [{min_margin:.4f}%, {max_margin:.4f}%]。".format(
            mode=q3["pricing_search_mode"],
            cand=q3.get("candidate_count", 0) or 0,
            feas=q3.get("feasible_count", 0) or 0,
            subsidy=q3.get("annual_subsidy", 0.0),
            profit=q3.get("annual_profit", 0.0),
            margin=q3.get("profit_margin", 0.0),
            min_margin=station_margin_range[0],
            max_margin=station_margin_range[1],
        )
    )

    return {
        "scenario": scenario_id,
        "description": description,
        "q1_recomputed": True,
        "budget_limit": float(budget_limit),
        "q1": {
            "population_year5": pop_year5,
            "constrained_demand_daily": constrained_daily,
            "forecast_history": forecast_history,
        },
        "q2": q2,
        "q3": q3,
    }


def build_scenario_configs(base_config):
    cfg_a = copy.deepcopy(base_config)
    cfg_a["transition_params"]["elder_growth_rate"] = 0.08
    cfg_a["transition_matrix"] = np.array([
        [1.0 - 0.055, 0.055, 0.0],
        [0.0, 1.0 - 0.095, 0.095],
        [0.0, 0.0, 1.0],
    ])

    cfg_b = copy.deepcopy(base_config)
    for scale in ["small", "medium", "large"]:
        cfg_b["station_specs"][scale]["daily_fixed_cost"] *= 1.20

    cfg_c = copy.deepcopy(base_config)
    return {
        "A": (cfg_a, "人口压力场景：增长率8%，自理->半失能5.5%，半失能->失能9.5%", 120.0),
        "B": (cfg_b, "成本上升场景：日固定管理成本增加20%", 120.0),
        "C": (cfg_c, "预算宽松场景：建设预算提高至140万元", 140.0),
    }


def main():
    print("========== T4 步骤2：基准场景与扰动场景统一口径重算 ==========")
    required = [
        MODEL_DIR / "T0_01_config.pkl",
        MODEL_DIR / "T3_02_candidate_price_plans.pkl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print("[错误] 缺少前置正式结果：")
        for path in missing:
            print(f"  - {path}")
        sys.exit(1)

    base_config = load_pickle(MODEL_DIR / "T0_01_config.pkl")
    baseline = process_scenario("baseline", base_config, None, "Baseline scenario: original parameters, recompute Q2 and Q3 with the unified T4 pipeline", 120.0)
    dump_pickle(MODEL_DIR / "T4_02_scenario_baseline_results.pkl", baseline)
    print(
        "Baseline: 使用T4统一口径重算Q2/Q3，站点 {stations}，覆盖率 {coverage:.2f}%，"
        "Q3全区利润率 {margin:.4f}%，逐站利润率范围 [{min_margin:.4f}%, {max_margin:.4f}%]。".format(
            stations=[station_label(base_config, j) + scale_name(baseline["q2"]["layout"][j]) for j in baseline["q2"]["active_stations"]],
            coverage=baseline["q2"]["coverage_rate"] * 100,
            margin=baseline["q3"]["profit_margin"],
            min_margin=baseline["q3"].get("min_station_profit_margin", 0.0),
            max_margin=baseline["q3"].get("max_station_profit_margin", 0.0),
        )
    )

    scenario_configs = build_scenario_configs(base_config)
    results = {}
    for scenario_id, (scenario_config, description, budget_limit) in scenario_configs.items():
        results[scenario_id] = process_scenario(
            scenario_id,
            scenario_config,
            baseline,
            description,
            budget_limit,
        )
        dump_pickle(MODEL_DIR / f"T4_02_scenario_{scenario_id}_results.pkl", results[scenario_id])

    print("\nT4_02 完成：已保存 baseline/A/B/C 统一结构结果。")
    for scenario_id, result in {"Baseline": baseline, **results}.items():
        print(
            f"  {scenario_id}: 覆盖率 {result['q2']['coverage_rate']*100:.2f}%，"
            f"覆盖内满意度 {result['q2']['avg_satisfaction_covered']:.4f}，"
            f"Q3全区利润率 {result['q3']['profit_margin']:.4f}%，"
            f"逐站利润率范围 [{result['q3'].get('min_station_profit_margin', 0.0):.4f}%, "
            f"{result['q3'].get('max_station_profit_margin', 0.0):.4f}%]"
        )


if __name__ == "__main__":
    main()
