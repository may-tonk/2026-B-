import os
import sys
import copy
import pickle
import numpy as np
from pathlib import Path


def main():
    print("========== 问题 4.1：灵敏度分析场景配置生成与验证 ==========\n")

    config_path = Path('model/T0_01_config.pkl')
    if not config_path.exists():
        print("[错误] 找不到 T0_01_config.pkl，请确认前置数据正常。")
        sys.exit(1)

    with open(config_path, 'rb') as f:
        base_config = pickle.load(f)

    scenarios = {}
    scenarios['baseline'] = copy.deepcopy(base_config)

    print("【场景参数基础设置】")
    print("成功加载基准 (Baseline) 配置。即将生成三个扰动场景...\n")

    # 场景 A：人口压力场景（调整版参数）
    # 老人增长率 7% -> 8%；自理到半失能 4.5% -> 5.5%；半失能到失能 10% -> 9.5%
    config_A = copy.deepcopy(base_config)
    config_A['transition_params']['elder_growth_rate'] = 0.08
    
    p_self_to_semi = 0.055
    p_semi_to_disabled = 0.095
    death_rate = base_config.get('transition_params', {}).get('death_rate', 0.05)
    
    config_A['transition_matrix'] = np.array([
        [1.0 - p_self_to_semi, p_self_to_semi, 0.0],
        [0.0, 1.0 - p_semi_to_disabled, p_semi_to_disabled],
        [0.0, 0.0, 1.0]
    ])
    scenarios['scenario_A'] = config_A

    # 场景 B：成本上升场景（调整版参数）
    # 所有站点日固定管理成本增加 20%
    config_B = copy.deepcopy(base_config)
    for scale, spec in config_B['station_specs'].items():
        spec['daily_fixed_cost'] = round(spec['daily_fixed_cost'] * 1.20, 2)
    scenarios['scenario_B'] = config_B

    # 场景 C：预算宽松场景（调整版参数）
    # 建设预算 120 万元 -> 140 万元
    config_C = copy.deepcopy(base_config)
    config_C['budget_limit'] = 140.0
    scenarios['scenario_C'] = config_C

    print("【扰动场景参数核对】\n")
    print(
        f"{'参数名称':<25} | {'基准 (Baseline)':<18} | {'场景 A (人口压力)':<18} | {'场景 B (成本上升)':<18} | {'场景 C (预算宽松)':<18}")
    print("-" * 110)

    # 对比：新增老年人口比例
    base_growth = f"{base_config.get('transition_params', {}).get('elder_growth_rate', 0.07) * 100:.1f}%"
    a_growth = f"{config_A.get('transition_params', {}).get('elder_growth_rate', 0) * 100:.1f}%"
    print(f"{'新增老年人口比例':<29} | {base_growth:<20} | {a_growth:<21} | {base_growth:<21} | {base_growth:<21}")

    # 对比：自理->半失能转移率
    base_p1 = f"{base_config['transition_matrix'][0, 1]:.3f}"
    a_p1 = f"{config_A['transition_matrix'][0, 1]:.3f}"
    print(f"{'转移概率: 自理->半失能':<25} | {base_p1:<20} | {a_p1:<21} | {base_p1:<21} | {base_p1:<21}")

    # 对比：半失能->失能转移率
    base_p2 = f"{base_config['transition_matrix'][1, 2]:.3f}"
    a_p2 = f"{config_A['transition_matrix'][1, 2]:.3f}"
    print(f"{'转移概率: 半失能->失能':<25} | {base_p2:<20} | {a_p2:<21} | {base_p2:<21} | {base_p2:<21}")

    # 对比：小型站日固定成本
    base_small_cost = f"{base_config['station_specs']['small']['daily_fixed_cost']:.0f} 元/日"
    b_small_cost = f"{config_B['station_specs']['small']['daily_fixed_cost']:.0f} 元/日"
    print(
        f"{'日固定管理成本(小型站)':<26} | {base_small_cost:<18} | {base_small_cost:<19} | {b_small_cost:<19} | {base_small_cost:<19}")

    # 对比：大型站日固定成本
    base_large_cost = f"{base_config['station_specs']['large']['daily_fixed_cost']:.0f} 元/日"
    b_large_cost = f"{config_B['station_specs']['large']['daily_fixed_cost']:.0f} 元/日"
    print(
        f"{'日固定管理成本(大型站)':<26} | {base_large_cost:<18} | {base_large_cost:<19} | {b_large_cost:<19} | {base_large_cost:<19}")

    # 对比：建设成本 (确认不变)
    base_small_build = f"{base_config['station_specs']['small']['build_cost']} 万元"
    print(
        f"{'建设成本(小型站-确认不变)':<24} | {base_small_build:<18} | {base_small_build:<19} | {base_small_build:<19} | {base_small_build:<19}")

    # 对比：建设预算
    base_budget = "120 万元"
    c_budget = "140 万元"
    print(f"{'建设预算':<29} | {base_budget:<20} | {base_budget:<21} | {base_budget:<21} | {c_budget:<21}")

    print("\n【执行规划提示】")
    print("  - 场景 A（人口压力）：增长率从7%升至8%，自理→半失能从4.5%升至5.5%，半失能→失能从10%降至9.5%")
    print("    将导致第5年老人总数增加，考验预算上限和容量限制")
    print("  - 场景 B（成本上升）：建设成本不变，日固定管理成本+20%，将压缩利润率，可能导致定价上涨")
    print("  - 场景 C（预算宽松）：建设预算从120万增至140万，可建设更多/更大站点")

    os.makedirs('model', exist_ok=True)
    with open('model/T4_01_scenario_configs.pkl', 'wb') as f:
        pickle.dump(scenarios, f)

    print("\n场景配置生成与验证完成！")
    print("结果已保存至: model/T4_01_scenario_configs.pkl")
    print("================================\n")


if __name__ == '__main__':
    main()
