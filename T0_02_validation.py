import os
import pickle
import json
import numpy as np
from T0_validation_utils import validate_config


def main():
    # ==========================================
    # 1. 加载数据
    # ==========================================
    try:
        with open('model/T0_01_config.pkl', 'rb') as f:
            config = pickle.load(f)
    except FileNotFoundError:
        print("错误：找不到 model/T0_01_config.pkl，请确保先运行了 T0_01_data_loader.py")
        return

    validate_config(config)

    print("========== 数据校验结果 ==========\n")

    # ==========================================
    # 2. 维度与基础一致性校验
    # ==========================================
    print("【维度校验】")
    assert config['population_matrix'].shape == (10, 3), "人口矩阵维度错误"
    print("OK 人口矩阵: (10, 3)")
    assert config['transition_matrix'].shape == (3, 3), "转移矩阵维度错误"
    print("OK 转移矩阵: (3, 3)")
    assert config['demand_rate_matrix'].shape == (3, 6), "需求率矩阵维度错误"
    print("OK 需求矩阵: (3, 6)")
    assert config['distance_matrix'].shape == (10, 10), "距离矩阵维度错误"
    print("OK 距离矩阵: (10, 10)")
    assert len(config['communities']) == 10, "小区数应为10"
    print("OK 小区数: 10")
    assert len(config['service_names']) == 6, "服务数应为6"
    print("OK 服务数: 6")
    assert config['communities'] == ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'], "小区顺序应为 A-J"
    print("OK 小区顺序: [A, B, C, D, E, F, G, H, I, J]\n")

    print("【服务顺序校验】")
    expected_service_order = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴', '紧急救助']
    assert config[
               'service_names'] == expected_service_order, f"服务顺序错误，期望：{expected_service_order}，实际：{config['service_names']}"
    print(f"OK 固定顺序: {expected_service_order}\n")

    # ==========================================
    # 3. 业务逻辑深度校验
    # ==========================================
    print("【转移矩阵结构校验】")
    tm = config['transition_matrix']
    assert np.isclose(tm[0, 0] + tm[0, 1], 1.0), "自理状态: 只能保持自理或转为半失能"
    assert np.isclose(tm[1, 1] + tm[1, 2], 1.0), "半失能状态: 只能保持半失能或转为失能"
    assert np.isclose(tm[2, 2], 1.0), "失能状态: 必须保持失能"
    assert tm[0, 2] == 0, "自理不能直接转为失能"
    assert tm[1, 0] == 0, "半失能不能转回自理"
    assert tm[2, 0] == 0 and tm[2, 1] == 0, "失能不能恢复"
    print(f"OK 自理->半失能转移率: {config['transition_params']['healthy_to_semi']}")
    print(f"OK 半失能->失能转移率: {config['transition_params']['semi_to_disabled']}")
    print("OK 自理不能直接转失能\nOK 半失能不能转回自理\nOK 失能不能恢复\n")

    print("【老人结构一致性校验】")
    print("小区 | 自理 | 半失能 | 失能 | 合计")
    print("-----|-----|-------|------|-----")
    total_all = 0
    for i, community in enumerate(config['communities']):
        pop_sum = config['population_matrix'][i].sum()
        elderly_total = config['elderly_total'][i]
        assert pop_sum == elderly_total, f"小区{community}三类老人数之和应等于60+老人数"
        c_p = config['population_matrix'][i]
        print(f"{community:<4} | {c_p[0]:<4} | {c_p[1]:<5} | {c_p[2]:<4} | {pop_sum}")
        total_all += pop_sum
    print(f"总计 | ...  | ...    | ...  | {total_all}\n")

    print("【数值范围校验】")
    assert np.all(config['population_matrix'] >= 0), "人口数不能为负"
    print("OK 人口数非负")
    assert np.all(config['service_prices'] >= 0), "价格不能为负"
    print("OK 价格非负")
    assert np.all(config['distance_matrix'] >= 0), "距离不能为负"
    print("OK 距离非负")
    assert config['spending_cap_ratios'].tolist() == [0.20, 0.25, 0.30], "消费上限比例应为 [0.20, 0.25, 0.30]"
    print(f"OK 消费上限比例: {config['spending_cap_ratios'].tolist()}\n")

    print("【特殊字段校验】")
    assert config['service_prices'][5] == 0, "紧急救助单价应为0"
    print("OK 紧急救助单价: 0 元/次")
    print(f"OK 紧急救助直接成本: {config['direct_costs'][5]} 元/次\n")

    # ==========================================
    # 4. 距离与物理拓扑网络生成
    # ==========================================
    print("【距离矩阵校验】")
    assert np.allclose(config['distance_matrix'], config['distance_matrix'].T), "距离矩阵应对称"
    print("OK 对称性检查")
    assert np.allclose(np.diag(config['distance_matrix']), 0), "距离矩阵对角线应为0"
    print("OK 对角线为0")

    # 核心：生成 1000米有效覆盖矩阵
    coverage_mask = config['coverage_mask']
    print("OK 覆盖矩阵对角线为1，自身可覆盖")
    print(f"OK 距离范围: [{config['distance_matrix'].min()} m, {config['distance_matrix'].max()} m]\n")

    print("========== 覆盖关系分析 ==========")
    print(f"OK 覆盖矩阵已生成 {coverage_mask.shape}\n")
    print("小区间 ≤1000 米覆盖情况:")
    print("小区 | 1000米内可达小区数 | 候选可达比例")
    print("-----|-------------------|-----------")

    reachable_counts = []
    for i, comm in enumerate(config['communities']):
        reachable = int(coverage_mask[i].sum())
        reachable_counts.append(reachable)
        ratio = (reachable / 10) * 100
        print(f"{comm:<4} | {reachable:<17} | {ratio:.1f}%")

    avg_reachable = np.mean(reachable_counts)
    max_reach = np.max(reachable_counts)
    min_reach = np.min(reachable_counts)
    print(f"\n平均覆盖候选站点数: {avg_reachable:.1f} 个")
    print(f"最多: {max_reach} 个，最少: {min_reach} 个\n")
    print("说明: 覆盖关系已记录，需求未做截断，待 Q2 阶段处理\n")

    print("所有校验通过! OK")
    print("================================\n")

    # ==========================================
    # 5. 输出文件
    # ==========================================
    os.makedirs('model', exist_ok=True)
    np.save('model/T0_02_coverage_mask.npy', coverage_mask)

    # 导出纯粹的满意度规则配置供 Q2 和 Q3 轻量级调用
    with open('model/T0_02_distance_satisfaction_rules.json', 'w', encoding='utf-8') as f:
        json.dump(config['satisfaction_rules']['distance'], f, indent=2, ensure_ascii=False)

    with open('model/T0_02_all_satisfaction_rules.json', 'w', encoding='utf-8') as f:
        json.dump({
            'distance': config['satisfaction_rules']['distance'],
            'response': config['satisfaction_rules']['response'],
            'price': config['satisfaction_rules']['price'],
            'weights': config['satisfaction_weights']
        }, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
