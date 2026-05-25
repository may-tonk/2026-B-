import os
import sys
import pickle
import numpy as np
from pathlib import Path


def get_price_level(price, base_price):
    """判断价格所属的溢价等级"""
    ratio = price / base_price
    if ratio <= 1.0 + 1e-4:
        return "平价"
    elif ratio <= 1.1 + 1e-4:
        return "微溢价"
    elif ratio <= 1.2 + 1e-4:
        return "中溢价"
    else:
        return "高溢价"


def main():
    print("========== 问题 3.2：最优定价方案选择 (分站差异化定价) ==========\n")

    # 1. 加载数据
    eval_path = Path('model/T3_03_price_evaluation_results.pkl')
    feasible_path = Path('model/T3_04_feasible_price_plans.pkl')
    cfg_path = Path('model/T3_01_financial_model_config.pkl')
    config_path = Path('model/T0_01_config.pkl')
    t2_path = Path('model/T2_04_best_solution.pkl')
    cand_path = Path('model/T3_02_candidate_price_plans.pkl')

    if not all(p.exists() for p in [feasible_path, eval_path, cfg_path, config_path, t2_path, cand_path]):
        print("[错误] 找不到前置文件，请确认 T0~T3_04 均已成功执行。")
        sys.exit(1)

    with open(eval_path, 'rb') as f:
        all_evaluated = pickle.load(f)
    with open(feasible_path, 'rb') as f:
        feasible_plans = pickle.load(f)
    with open(cfg_path, 'rb') as f:
        fin_cfg = pickle.load(f)
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    with open(t2_path, 'rb') as f:
        best_sol = pickle.load(f)
    with open(cand_path, 'rb') as f:
        cand_data = pickle.load(f)

    active_stations = best_sol['active_stations']
    station_scales = {j: best_sol['layout'][j] for j in active_stations}
    scale_zh = {1: '小型', 2: '中型', 3: '大型'}
    station_templates = cand_data['station_templates']

    print("【最优方案选取规则】")
    print("目标函数：在满足所有约束的可行方案中，选择加权平均综合满意度最高的方案\n")
    print("选取过程：")
    print(f"  可行方案数：{len(feasible_plans):,} 个")

    if len(feasible_plans) == 0:
        print("[严重警告] 可行方案数为 0！无法选择最优方案。请检查 T3_04 的过滤条件。")
        sys.exit(1)

    # 2. 选择最优方案 (匹配分站定价的数据结构)
    # 排序规则: 满意度降序 -> 老人支付(服务营收)升序 -> 政府补贴升序 -> 利润率接近保本(0)升序 -> ID保证稳定排序
    optimal_plan = sorted(
        feasible_plans,
        key=lambda p: (
            -p['weighted_avg_satisfaction'],
            p['system_financials']['annual_service_revenue'],
            p['system_financials']['annual_subsidy'],
            abs(p['system_financials']['profit_margin'] - 0.0),
            p['plan_id']
        )
    )[0]

    print(
        f"  最优方案（满意度最高）：方案编号 {optimal_plan['plan_id']}，满意度 {optimal_plan['weighted_avg_satisfaction']:.4f}\n")

    # 3. 打印最优分站定价方案详情
    print("【最优分站定价方案详情】\n")
    service_names = config['service_names']
    base_prices = fin_cfg['service_prices_base']

    sum_premium_rate = 0.0
    station_count = len(active_stations)

    for st in active_stations:
        st_name = config['communities'][st]
        tid = optimal_plan['station_template_ids'][st]
        template = next(t for t in station_templates if t['template_id'] == tid)

        print(f"站点 {st_name} ({scale_zh[station_scales[st]]}) - 采用定价模板: {template['label']}")
        print(f"{'服务项目':<10}| {'定价(元)':<8}| {'基准价(元)':<9}| {'溢价率(%)':<8}| {'价格等级':<10}| {'S3':<8}")
        print("-" * 65)

        s3_tuple = optimal_plan['station_s3_by_service'][st]
        st_premium_sum = 0.0

        for i, svc in enumerate(service_names[:5]):
            price = template['prices'][svc]
            base = base_prices[i]
            rate = (price - base) / base * 100
            st_premium_sum += rate
            level = get_price_level(price, base)
            s3 = s3_tuple[i]
            print(f"{svc:<10}| {price:<10.1f}| {base:<11.1f}| {rate:<10.1f}| {level:<10}| {s3:<8.2f}")

        sum_premium_rate += (st_premium_sum / 5)
        print(f"{'紧急救助':<10}| {'0(免费)':<9}| {'0':<11}| {'—':<10}| {'公益免费':<10}| {'—':<8}\n")

    avg_premium_rate = sum_premium_rate / station_count

    # 4. 年度财务指标
    print("【年度系统财务指标】\n")
    sys_fin = optimal_plan['system_financials']

    print("收入端：")
    total_visits = sum(optimal_plan['station_loads'].values()) * 365
    print(f"  年有效服务总人次：{total_visits:,.0f} 人次")
    print(f"  年服务营收（5项付费）：{sys_fin['annual_service_revenue']:.2f} 万元")
    print(f"  年政府补贴收入：{sys_fin['annual_subsidy']:.2f} 万元")
    print(f"  年总营收：{sys_fin['annual_total_revenue']:.2f} 万元\n")

    print("成本端：")
    annual_operating_cost = fin_cfg['annual_operating_cost']
    print(f"  年固定成本（运营成本总额）：{annual_operating_cost:.2f} 万元")
    print(f"    - 日均固定管理成本合计：{fin_cfg['annual_fixed_management_cost']:.2f} 万元/年")
    print(f"    - 年度折旧成本：{fin_cfg['annual_depreciation']:.2f} 万元/年")
    print(f"\n  年变动成本：{sys_fin['annual_variable_cost']:.2f} 万元")
    print("    - 由实际有效服务人次 × 单次直接成本计算")
    print(f"\n  年总成本：{sys_fin['annual_total_cost']:.2f} 万元\n")

    print("利润端：")
    print(f"  年利润：{sys_fin['annual_profit']:.2f} 万元")
    print(f"  利润率：{sys_fin['profit_margin']:.2f}%（严格符合 0%-8% 约束）\n")

    # 5. 各站点补贴情况
    print("【各站点的精算补贴情况】")
    print(
        f"{'站点':<4} | {'规模':<4} | {'日均付费人次':<12} | {'日补贴请求(元)':<12} | {'日上限(元)':<10} | {'实际补贴(元)':<10} | {'年补贴(万)':<10}")
    print("-" * 85)

    total_req = 0
    total_actual = 0

    daily_demand = np.load('model/T1_03_constrained_demand_daily_by_service.npy')
    assignments = best_sol['assignments']
    assigned_station = np.full(len(config['communities']), -1)
    S1_fixed = np.zeros(len(config['communities']))
    dist_matrix = config['distance_matrix']

    # 重构分配与S1
    for i in range(len(config['communities'])):
        if best_sol['community_satisfactions'][i] > 0:
            best_j = assignments[i]
            assigned_station[i] = best_j
            dist = dist_matrix[i, best_j]
            if dist <= 300:
                S1_fixed[i] = 1.00
            elif dist <= 500:
                S1_fixed[i] = 0.90
            elif dist <= 650:
                S1_fixed[i] = 0.75
            elif dist <= 1000:
                S1_fixed[i] = 0.60
            else:
                S1_fixed[i] = 0.0

    for st in active_stations:
        st_name = config['communities'][st]
        scale = scale_zh[station_scales[st]]
        station_fin = optimal_plan['station_financials'][st]

        daily_paid_visits = station_fin['paid_effective_daily_visits']
        req_yuan = station_fin['daily_subsidy_request_yuan']
        actual_yuan = station_fin['daily_subsidy_actual_yuan']
        annual_sub = station_fin['annual_subsidy']

        total_req += req_yuan
        total_actual += actual_yuan

        print(
            f"{st_name:<5} | {scale:<5} | {daily_paid_visits:<14.1f} | {req_yuan:<16.1f} | {station_fin['daily_subsidy_cap_yuan']:<12.1f} | {actual_yuan:<14.1f} | {annual_sub:<10.2f}")

    if abs(total_req - total_actual) < 1e-4:
        print("\n说明：各站点的补贴都在日上限内，全部能够获得\n")
    else:
        print("\n说明：部分站点的补贴请求已触碰日上限，实际补贴已严格按上限截断\n")

    # 6. 小区满意度细分
    print("【小区满意度细分】")
    print(
        f"{'小区':<4} | {'老人数':<6} | {'归属站点':<8} | {'距离满意度':<10} | {'响应满意度':<10} | {'价格满意度':<10} | {'综合满意度':<8}")
    print("-" * 85)

    pop_matrix = np.load('model/T1_01_population_year5.npy')
    elderly_cnt = np.sum(pop_matrix, axis=1)

    for i, comm in enumerate(config['communities']):
        if assigned_station[i] == -1:
            print(f"{comm:<5} | {int(elderly_cnt[i]):<8} | {'—':<10} | {'—':<12} | {'—':<12} | {'—':<12} | —（未覆盖）")
        else:
            st = assigned_station[i]
            st_name = config['communities'][st]
            s1 = S1_fixed[i]
            s2 = optimal_plan['station_response_scores'][st]
            s3 = optimal_plan['weighted_s3_by_community'][i]
            s_comp = optimal_plan['satisfaction_by_community'][i]
            print(
                f"{comm:<5} | {int(elderly_cnt[i]):<8} | {st_name:<10} | {s1:<14.2f} | {s2:<14.2f} | {s3:<14.2f} | {s_comp:<8.2f}")

    print(
        f"\n全区(加权平均) | {'—':<8} | {'—':<10} | {'—':<14} | {'—':<14} | {'—':<14} | {optimal_plan['weighted_avg_satisfaction']:.4f}\n")

    # 7. 对比基准方案（所有站点全1.0倍基准价）
    # 找到全基准价模板
    baseline_template_id = None
    for template in station_templates:
        if all(abs(m - 1.0) < 0.001 for m in template['multipliers']):
            baseline_template_id = template['template_id']
            break
    
    # 找到所有站点都用全基准价模板的方案
    baseline_plan = None
    for plan in all_evaluated:
        all_baseline = True
        for station_idx, template_id in plan['station_template_ids'].items():
            if template_id != baseline_template_id:
                all_baseline = False
                break
        if all_baseline:
            baseline_plan = plan
            break
    
    if baseline_plan is None:
        print("[警告] 未找到全基准价方案，使用第一个方案代替！")
        baseline_plan = all_evaluated[0]

    print("【与基准方案（全基准价）的对比】\n")
    print(f"{'指标':<18} | {'基准方案(全基准价)':<16} | {'最优方案(分站最优)':<16} | {'变化量':<12} | {'变化率(%)':<8}")
    print("-" * 89)

    base_sat = baseline_plan['weighted_avg_satisfaction']
    opt_sat = optimal_plan['weighted_avg_satisfaction']
    d_sat = opt_sat - base_sat
    r_sat = (d_sat / base_sat * 100) if base_sat > 0 else 0

    base_visits = sum(baseline_plan['station_loads'].values()) * 365
    opt_visits = total_visits
    d_visits = opt_visits - base_visits
    r_visits = (d_visits / base_visits * 100) if base_visits > 0 else 0

    base_rev = baseline_plan['system_financials']['annual_service_revenue']
    opt_rev = sys_fin['annual_service_revenue']
    d_rev = opt_rev - base_rev
    r_rev = (d_rev / base_rev * 100) if base_rev > 0 else 0

    base_sub = baseline_plan['system_financials']['annual_subsidy']
    opt_sub = sys_fin['annual_subsidy']
    d_sub = opt_sub - base_sub
    r_sub = (d_sub / base_sub * 100) if base_sub > 0 else 0

    base_prof = baseline_plan['system_financials']['annual_profit']
    opt_prof = sys_fin['annual_profit']
    d_prof = opt_prof - base_prof
    r_prof = (d_prof / abs(base_prof) * 100) if base_prof != 0 else 0

    base_margin = baseline_plan['system_financials']['profit_margin']
    opt_margin = sys_fin['profit_margin']
    d_margin = opt_margin - base_margin

    print(
        f"{'定价溢价率（均值）':<16} | {'0%':<20} | {avg_premium_rate:<19.1f}% | {avg_premium_rate:<11.1f}%p | {'—':<8}")
    print(f"{'综合满意度':<21} | {base_sat:<20.4f} | {opt_sat:<20.4f} | {d_sat:<13.4f} | {r_sat:<8.1f}%")
    print(
        f"{'年有效服务人次':<20} | {base_visits:<16,.0f}人次 | {opt_visits:<16,.0f}人次 | {d_visits:<11,.0f}人次 | {r_visits:<8.1f}%")
    print(f"{'年服务营收':<22} | {base_rev:<18.2f}万元 | {opt_rev:<18.2f}万元 | {d_rev:<11.2f}万元 | {r_rev:<8.1f}%")
    print(f"{'年政府补贴':<22} | {base_sub:<18.2f}万元 | {opt_sub:<18.2f}万元 | {d_sub:<11.2f}万元 | {r_sub:<8.1f}%")
    print(f"{'年利润':<24} | {base_prof:<18.2f}万元 | {opt_prof:<18.2f}万元 | {d_prof:<11.2f}万元 | {r_prof:<8.1f}%")
    print(f"{'利润率':<25} | {base_margin:<19.2f}% | {opt_margin:<19.2f}% | {d_margin:<11.2f}%p | {'—':<8}\n")

    print("结论：")
    dir_str = "结构性调价" if avg_premium_rate > 0 else "全面降价"
    sat_str = "提升" if d_sat > 0 else "保持" if d_sat == 0 else "适度让步"
    print(f"  相比全基准价方案，最优分站定价方案通过 {dir_str} 实现了满意度的 {sat_str}，")
    print(f"  并在 {opt_margin:.2f}% 的利润率水平严格保本（将基准方案 {base_margin:.2f}% 的过高利润率压回微利区间）。")

    print("\n最优分站定价方案选择完成！")
    print("================================\n")

    # 8. 保存最优方案（同时保持与T3_06、T3_07的兼容性）
    optimal_plan['is_feasible'] = True
    optimal_plan['station_prices'] = {}
    optimal_plan['station_template_labels'] = {}
    for st in active_stations:
        tid = optimal_plan['station_template_ids'][st]
        template = next(t for t in station_templates if t['template_id'] == tid)
        optimal_plan['station_prices'][st] = dict(template['prices'])
        optimal_plan['station_template_labels'][st] = template['label']
    # 复制 system_financials 到 sys_fin 以保持兼容性
    optimal_plan['sys_fin'] = optimal_plan['system_financials'].copy()
    # 添加 avg_premium_rate
    optimal_plan['sys_fin']['avg_premium_rate'] = avg_premium_rate
    # 添加 T3_07 需要的字段
    if 'annual_fixed_management_cost' not in optimal_plan['sys_fin']:
        optimal_plan['sys_fin']['annual_fixed_management_cost'] = fin_cfg.get('annual_fixed_management_cost', 0.0)
    if 'annual_depreciation' not in optimal_plan['sys_fin']:
        optimal_plan['sys_fin']['annual_depreciation'] = fin_cfg.get('annual_depreciation', 0.0)
    if 'annual_effective_visits' not in optimal_plan['sys_fin']:
        optimal_plan['sys_fin']['annual_effective_visits'] = total_visits
    
    os.makedirs('model', exist_ok=True)
    with open('model/T3_05_optimal_price_plan.pkl', 'wb') as f:
        pickle.dump(optimal_plan, f)


if __name__ == '__main__':
    main()
