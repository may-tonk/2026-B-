import os
import sys
import time
import pickle
import itertools
import numpy as np
from pathlib import Path


def load_data():
    """加载 T0 和 T1 的标准化数据"""
    config_path = Path('model/T0_01_config.pkl')
    coverage_path = Path('model/T0_02_coverage_mask.npy')
    demand_path = Path('model/T1_03_constrained_demand_daily_by_service.npy')

    if not config_path.exists() or not coverage_path.exists() or not demand_path.exists():
        print("[错误] 找不到前置依赖数据，请确保 T0 和 T1 阶段已完整运行。")
        sys.exit(1)

    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    coverage_mask = np.load(coverage_path)
    daily_demand = np.load(demand_path)

    return config, coverage_mask, daily_demand


def main():
    print("========== 问题 2.1：站点布局全局枚举与预算剪枝 ==========\n")

    # 1. 准备数据与参数
    config, coverage_mask, daily_demand = load_data()
    n_communities = config['n_communities']
    budget_limit = 120.0  # 万元

    # 构建成本映射表
    # 0: 不建, 1: 小型, 2: 中型, 3: 大型
    costs = np.array([
        0.0,
        config['station_specs']['small']['build_cost'],
        config['station_specs']['medium']['build_cost'],
        config['station_specs']['large']['build_cost']
    ])

    print("【枚举参数确认】")
    print(f"小区数: {n_communities} (A-J)")
    print("每小区规模选择数: 4 (不建、小型、中型、大型)")
    print(f"总布局数上界: 4^{n_communities} = {4 ** n_communities:,}")
    print("\n【站点建设成本参考】")
    print("不建: 0 万元")
    for scale_idx, scale_name, zh_name in [(1, 'small', '小型'), (2, 'medium', '中型'), (3, 'large', '大型')]:
        spec = config['station_specs'][scale_name]
        print(
            f"{zh_name}: {spec['build_cost']} 万元, 日容量 {spec['daily_capacity']} 人次/日, 日固定成本 {spec['daily_fixed_cost']} 元")

    print("\n【预算约束】")
    print(f"总建设预算上限: {budget_limit} 万元\n")

    # 2. 向量化全局枚举与剪枝
    start_time = time.time()

    print("【枚举进度】")
    print(f"已生成总布局数: {4 ** n_communities:,}")
    print("开始预算筛选...")

    # 利用 itertools.product 和 numpy 快速生成百万矩阵
    # shape: (1048576, 10)
    all_layouts = np.array(list(itertools.product([0, 1, 2, 3], repeat=n_communities)), dtype=np.int8)

    # 利用索引快速映射成本并求和
    # shape: (1048576,)
    build_costs = costs[all_layouts].sum(axis=1)

    # 预算剪枝过滤
    feasible_indices = np.where(build_costs <= budget_limit)[0]
    feasible_layouts = all_layouts[feasible_indices]
    feasible_costs = build_costs[feasible_indices]

    n_total = len(all_layouts)
    n_feasible = len(feasible_layouts)
    n_infeasible = n_total - n_feasible
    avg_cost = feasible_costs.mean() if n_feasible > 0 else 0

    # 统计分布 (划分成本区间寻找最多的一档)
    if n_feasible > 0:
        bins = np.arange(0, 130, 10)
        hist, bin_edges = np.histogram(feasible_costs, bins=bins)
        max_bin_idx = np.argmax(hist)
        freq_range_str = f"{bin_edges[max_bin_idx]}-{bin_edges[max_bin_idx + 1]}"
    else:
        freq_range_str = "0"

    print("\n【预算剪枝结果】")
    print(f"[OK] 预算超支的布局数: {n_infeasible:,} (已过滤)")
    print(f"[OK] 预算可行的布局数: {n_feasible:,} (保留)")
    print(f"  其中：{freq_range_str}万 成本区间的布局数最多")
    print(f"  平均建设成本: {avg_cost:.2f} 万元\n")

    # 3. 覆盖关系预核验
    # 只有当一个小区距离某个激活的站点 <= 1000m 时，才算被覆盖
    # feasible_layouts > 0 生成布尔矩阵 (n_feasible, 10) 表示是否有站点
    # 覆盖矩阵 shape (10, 10)，转置后点乘核验
    has_station_mask = (feasible_layouts > 0)
    # covered_matrix: (n_feasible, 10)
    covered_matrix = (has_station_mask @ coverage_mask.T) > 0
    # 检查是否每一行（10个小区）都为 True
    all_covered = covered_matrix.all(axis=1)

    full_coverage_count = all_covered.sum()
    partial_coverage_count = n_feasible - full_coverage_count

    print("【覆盖关系初步分析】")
    print("可行布局中：")
    print(f"  - 全覆盖布局数（所有小区都在1000m内至少有一个站点）: {full_coverage_count:,}")
    print(f"  - 部分覆盖布局数: {partial_coverage_count:,}")
    print("  说明: 本步仅记录，Q2.3 可选择目标\n")

    # 4. 数据打包并持久化
    # 按照 SOP 格式构建 list of dict
    # 为了减少对象拷贝，使用字典映射规避硬编码冗余
    scale_names = {1: 'small', 2: 'medium', 3: 'large'}
    output_list = []

    for i in range(n_feasible):
        layout = feasible_layouts[i].tolist()
        b_cost = float(feasible_costs[i])

        # 构建该布局的 active station 详细规格 (仅提取有站点的)
        s_specs = {}
        for comm_idx, scale_val in enumerate(layout):
            if scale_val > 0:
                s_specs[comm_idx] = config['station_specs'][scale_names[scale_val]]

        output_list.append({
            'layout': layout,
            'build_cost': b_cost,
            'station_specs': s_specs
        })

    os.makedirs('model', exist_ok=True)
    with open('model/T2_01_feasible_layouts.pkl', 'wb') as f:
        pickle.dump(output_list, f)

    end_time = time.time()
    elapsed = end_time - start_time
    # 预估后续 T2_02 处理耗时：假设每个布局需要约 0.0001~0.0005 秒
    est_satisfaction_time = n_feasible * 0.0002

    print("【时间成本估计】")
    print(f"全量枚举与剪枝耗时: {elapsed:.2f} 秒")
    print(f"预算剪枝后待评价的布局数: {n_feasible:,}")
    print(f"后续满意度计算估计耗时: 约 {est_satisfaction_time:.1f} 秒\n")

    print("预算剪枝完成！")
    print("可行布局已保存，准备进入满意度评价阶段。")
    print("================================================")


if __name__ == '__main__':
    main()