import os
import sys
import pickle
from pathlib import Path


SCALE_NAME_BY_CODE = {1: 'small', 2: 'medium', 3: 'large'}
SCALE_NAME_ZH_BY_CODE = {1: '小型', 2: '中型', 3: '大型'}
REQUIRED_T2_BEST_FIELDS = {
    'active_stations',
    'layout',
    'assignments',
    'build_cost',
    'station_capacities',
    'station_loads',
    'station_utilizations',
    'satisfaction_details',
}


def validate_t2_best_solution(best_solution):
    """Check the T2 best-solution interface required by T3."""
    missing_fields = sorted(REQUIRED_T2_BEST_FIELDS - set(best_solution))
    if missing_fields:
        raise KeyError(f'T2 最优方案缺少 T3 必需字段: {missing_fields}')

    active_stations = list(best_solution['active_stations'])
    if not active_stations:
        raise ValueError('T2 最优方案没有活动站点，无法建立 T3 财务模型')

    layout = best_solution['layout']
    invalid_scales = {
        layout[station_idx]
        for station_idx in active_stations
        if layout[station_idx] not in SCALE_NAME_BY_CODE
    }
    if invalid_scales:
        raise ValueError(f'T2 最优方案包含未知站点规模编码: {sorted(invalid_scales)}')


class FinancialModel:
    def __init__(self, best_solution, config):
        validate_t2_best_solution(best_solution)

        # 从 T2 加载站点配置
        self.stations = list(best_solution['active_stations'])
        self.layout = list(best_solution['layout'])
        self.assignments = dict(best_solution['assignments'])
        self.station_scales = {j: best_solution['layout'][j] for j in self.stations}
        self.total_build_cost = best_solution['build_cost']  # 万元
        self.station_capacities = {
            j: float(best_solution['station_capacities'][j])
            for j in self.stations
        }
        self.station_loads_t2 = {
            j: float(best_solution['station_loads'][j])
            for j in self.stations
        }
        self.station_utilizations_t2 = {
            j: float(best_solution['station_utilizations'][j])
            for j in self.stations
        }
        self.capacity_overload_ratio = float(best_solution.get('capacity_overload_ratio', 0.0))
        self.capacity_limit_ratio = float(
            best_solution.get('capacity_limit_ratio', 1.0 + self.capacity_overload_ratio)
        )
        self.station_elastic_capacity_limits = {
            j: self.station_capacities[j] * self.capacity_limit_ratio
            for j in self.stations
        }
        self.station_overload_rates_t2 = {
            j: max(0.0, self.station_loads_t2[j] - self.station_capacities[j])
            / self.station_capacities[j]
            for j in self.stations
        }
        self.satisfaction_details_t2 = dict(best_solution['satisfaction_details'])

        # 从 T0 加载基础参数
        self.service_names = config['service_names']
        self.paid_services = self.service_names[:5]  # 5项付费服务
        self.service_prices_base = config['service_prices']
        self.direct_costs = config['direct_costs']
        self.days_per_year = config['time_constants']['days_per_year']
        self.depreciation_years = config['time_constants']['depreciation_years']

        # 初始化字典
        self.daily_fixed_costs_yuan = {}
        self.daily_capacity = {}
        self.daily_subsidy_caps_yuan = {}
        self.station_build_costs_wan = {}
        self.station_annual_depreciation_wan = {}

        for station_idx in self.stations:
            scale_num = self.station_scales[station_idx]
            spec_name = SCALE_NAME_BY_CODE[scale_num]
            spec = config['station_specs'][spec_name]

            self.daily_fixed_costs_yuan[station_idx] = spec['daily_fixed_cost']
            self.daily_capacity[station_idx] = self.station_capacities[station_idx]
            self.daily_subsidy_caps_yuan[station_idx] = spec['subsidy_cap_day']
            self.station_build_costs_wan[station_idx] = spec['build_cost']
            self.station_annual_depreciation_wan[station_idx] = (
                spec['build_cost'] / self.depreciation_years
            )

        # 计算年固定成本（单位统一转为万元）
        self.annual_depreciation = sum(self.station_annual_depreciation_wan.values())
        self.annual_fixed_management_cost = (
            sum(self.daily_fixed_costs_yuan.values()) * self.days_per_year / 10000.0
        )

        # 年运营成本总额 = 年固定管理成本 + 年折旧成本
        self.annual_operating_cost = self.annual_fixed_management_cost + self.annual_depreciation

    def calculate_annual_variable_cost(self, effective_visits):
        """计算年变动成本（万元），含紧急救助的直接成本"""
        annual_variable_cost = 0.0
        for i, svc_name in enumerate(self.service_names):
            if svc_name in effective_visits:
                visits = effective_visits[svc_name]
                cost_per_visit = self.direct_costs[i]
                annual_variable_cost += (visits * cost_per_visit) / 10000.0
        return annual_variable_cost

    def calculate_annual_service_revenue(self, effective_visits, price_plan):
        """计算年服务营收（不含补贴，万元）。紧急救助价格固定为0。"""
        annual_revenue = 0.0
        for svc_name in self.paid_services:
            if svc_name in effective_visits:
                visits = effective_visits[svc_name]
                price = price_plan.get(svc_name, 0)
                annual_revenue += (visits * price) / 10000.0
        return annual_revenue

    def calculate_annual_subsidy(self, paid_effective_visits_daily_by_station):
        """计算年政府补贴（按各站点的日上限分别截断，万元）"""
        annual_subsidy = 0.0
        for station in self.stations:
            daily_paid_visits = paid_effective_visits_daily_by_station.get(station, 0)
            daily_request_yuan = daily_paid_visits * 2.0
            daily_cap_yuan = self.daily_subsidy_caps_yuan[station]

            # 日上限截断核心逻辑
            daily_actual_yuan = min(daily_request_yuan, daily_cap_yuan)
            annual_subsidy += (daily_actual_yuan * self.days_per_year) / 10000.0

        return annual_subsidy

    def calculate_profit_and_margin(self, annual_service_revenue, annual_subsidy, annual_variable_cost):
        """计算年利润和严格定义的利润率"""
        service_profit = annual_service_revenue - annual_variable_cost
        annual_total_revenue = annual_service_revenue + annual_subsidy
        annual_total_cost = self.annual_operating_cost + annual_variable_cost
        annual_profit = annual_total_revenue - annual_total_cost

        # 题面利润率公式: (服务总利润 + 年政府补贴 - 年运营成本总额) / 年运营成本总额 * 100%
        if self.annual_operating_cost > 0:
            profit_margin = (
                                        service_profit + annual_subsidy - self.annual_operating_cost) / self.annual_operating_cost * 100.0
        else:
            profit_margin = 0.0

        return {
            'total_revenue': annual_total_revenue,
            'variable_cost': annual_variable_cost,
            'operating_cost': self.annual_operating_cost,
            'total_cost': annual_total_cost,
            'profit': annual_profit,
            'profit_margin': profit_margin
        }


def main():
    print("========== 问题 3.1：财务模型建立 ==========\n")

    config_path = Path('model/T0_01_config.pkl')
    best_solution_path = Path('model/T2_04_best_solution.pkl')

    if not config_path.exists() or not best_solution_path.exists():
        print("[错误] 找不到 T0_01_config.pkl 或 T2_04_best_solution.pkl，请确保先执行了前置步骤。")
        sys.exit(1)

    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    with open(best_solution_path, 'rb') as f:
        best_sol = pickle.load(f)

    # 1. 初始化财务模型
    fm = FinancialModel(best_sol, config)

    # 2. 控制台输出
    print("【财务模型参数定义】\n")
    print("年固定成本（不随服务量变化）：")
    daily_fixed_total = sum(fm.daily_fixed_costs_yuan.values())
    print(f"  日均固定管理成本合计：{daily_fixed_total:.0f} 元/日")
    print(
        f"  → 年固定管理成本：{fm.annual_fixed_management_cost:.2f} 万元"
        f"（= {daily_fixed_total:.0f} × {fm.days_per_year} / 10000）\n"
    )
    print(f"  建设成本一次性折旧：")
    print(f"  → 一次性建设成本：{fm.total_build_cost:.1f} 万元（来自T2最优方案）")
    print(
        f"  → 年均折旧成本：{fm.annual_depreciation:.2f} 万元"
        f"（= {fm.total_build_cost:.1f} / {fm.depreciation_years}）\n"
    )
    print(f"  年固定成本总额（运营成本总额）：{fm.annual_operating_cost:.2f} 万元\n")

    print("年变动成本（随服务量变化）：")
    print("  年变动成本 = Σ(该服务实际有效年人次 × 单次直接成本)")
    print("  单位：万元")
    print("  \n  直接成本（元/次）：")
    for i, svc_name in enumerate(config['service_names']):
        tag = "（公益免费）" if svc_name == "紧急救助" else ""
        print(f"    {svc_name}：{config['direct_costs'][i]} 元 {tag}".strip())

    print("\n年总成本：")
    print("  年总成本 = 年固定成本 + 年变动成本\n")

    print("年总营收：")
    print("  年服务营收 = Σ(5项付费服务的实际有效年人次 × 定价)")
    print("  年政府补贴 = Σ(各站点按日上限约束的年补贴)")
    print("  年总营收 = 年服务营收 + 年政府补贴\n")

    print("年利润与利润率：")
    print("  年利润 = 年总营收 - 年总成本")
    print("  利润率 = (服务总利润 + 政府补贴 - 年运营成本总额) / 年运营成本总额 × 100%")
    print("  约束：0% ≤ 利润率 ≤ 8%\n")

    print("【站点配置（来自T2最优方案）】")
    for j in fm.stations:
        s_num = fm.station_scales[j]
        elastic_capacity = fm.station_elastic_capacity_limits[j]
        t2_load = fm.station_loads_t2[j]
        overload_rate = fm.station_overload_rates_t2[j] * 100
        print(
            f"  小区 {config['communities'][j]}：{SCALE_NAME_ZH_BY_CODE[s_num]}站，"
            f"日固定成本 {fm.daily_fixed_costs_yuan[j]} 元，附件日容量 {fm.daily_capacity[j]} 人次，"
            f"弹性上限 {elastic_capacity:.1f} 人次，T2 有效负荷 {t2_load:.1f} 人次，"
            f"原容量超载率 {overload_rate:.2f}%，日补贴上限 {fm.daily_subsidy_caps_yuan[j]} 元"
        )
    print(
        f"  容量口径：继承 T2 的 {fm.capacity_overload_ratio * 100:.0f}% 小幅弹性，"
        f"后续分站定价评价需按附件容量 × {fm.capacity_limit_ratio:.2f} 校验。\n"
    )

    print("\n【政府补贴规则】")
    print("补贴标准：5项付费服务实际有效服务人次 × 2 元/人次")
    print("日补贴上限：")
    print("  小型站：1000 元/日\n  中型站：1800 元/日\n  大型站：2600 元/日")
    print("  \n补贴计算方法：")
    print("  付费服务日有效人次 = 该站5项付费服务实际有效日人次之和")
    print("  日补贴请求 = 付费服务日有效人次 × 2 元")
    print("  日实际补贴 = min(日补贴请求, 日补贴上限)")
    print("  年补贴收入 = 日实际补贴 × 365\n")

    print("【优化模型设定】")
    print("决策变量：p_{j,s}（服务站 j 对付费服务 s 的定价，元/次）")
    print("目标函数：max 综合满意度（约束下）")
    print("约束条件：")
    print("  1. 利润率约束：0% ≤ 利润率(P) ≤ 8%")
    print("  2. 补贴按日上限截断：actual_subsidy = min(request, cap)")
    print("  3. 容量口径：各站有效负荷不超过 T2 选定弹性容量上限")
    print("  4. 价格区间约束：0.8×基准价 ≤ p_{j,s} ≤ 1.35×基准价")
    print("  5. 消费能力约束：（来自T1的削减系数已隐式体现）\n")

    print("求解算法准备：分站候选定价模板 + 约束评价")
    print("  说明：分站定价变量增多，后续 T3_02 将生成可计算的站点价格模板组合。")
    print("  求解过程：构造候选 → 逐一评价 → 过滤违约 → 比较目标函数 → 输出最优\n")

    # 3. 持久化财务配置
    model_config = {
        't2_best_solution_path': str(best_solution_path),
        'active_stations': fm.stations,
        'layout': fm.layout,
        'assignments': fm.assignments,
        'station_scales': fm.station_scales,
        'station_scale_names': {
            station_idx: SCALE_NAME_BY_CODE[scale]
            for station_idx, scale in fm.station_scales.items()
        },
        'station_scale_names_zh': {
            station_idx: SCALE_NAME_ZH_BY_CODE[scale]
            for station_idx, scale in fm.station_scales.items()
        },
        'annual_fixed_management_cost': fm.annual_fixed_management_cost,
        'annual_operating_cost': fm.annual_operating_cost,
        'daily_fixed_costs_yuan': fm.daily_fixed_costs_yuan,
        'daily_capacity': fm.daily_capacity,
        'station_capacities': fm.station_capacities,
        'station_elastic_capacity_limits': fm.station_elastic_capacity_limits,
        'capacity_overload_ratio': fm.capacity_overload_ratio,
        'capacity_limit_ratio': fm.capacity_limit_ratio,
        'station_loads_t2': fm.station_loads_t2,
        'station_utilizations_t2': fm.station_utilizations_t2,
        'station_overload_rates_t2': fm.station_overload_rates_t2,
        'satisfaction_details_t2': fm.satisfaction_details_t2,
        'service_names': fm.service_names,
        'paid_services': fm.paid_services,
        'service_prices_base': fm.service_prices_base,
        'direct_costs': fm.direct_costs,
        'total_build_cost': fm.total_build_cost,
        'station_build_costs_wan': fm.station_build_costs_wan,
        'annual_depreciation': fm.annual_depreciation,
        'station_annual_depreciation_wan': fm.station_annual_depreciation_wan,
        'subsidy_per_visit': 2,
        'daily_subsidy_caps_yuan': fm.daily_subsidy_caps_yuan,
        'profit_margin_lower': 0.0,
        'profit_margin_upper': 0.08,
        'profit_margin_lower_pct': 0.0,
        'profit_margin_upper_pct': 8.0,
        'days_per_year': fm.days_per_year,
        'depreciation_years': fm.depreciation_years,
    }

    os.makedirs('model', exist_ok=True)
    with open('model/T3_01_financial_model_config.pkl', 'wb') as f:
        pickle.dump(model_config, f)

    print("财务模型建立完成！")
    print("结果已保存至: model/T3_01_financial_model_config.pkl")
    print("================================\n")


if __name__ == '__main__':
    main()
