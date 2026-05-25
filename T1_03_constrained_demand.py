import pickle

import numpy as np

from T1_01_population_forecast import (
    MODEL_DIR,
    T0_CONFIG_PATH,
    YEAR5_POPULATION_PATH,
    load_t0_config,
)
from T1_02_theoretical_demand import ELDERLY_TYPE_KEYS, THEORETICAL_DEMAND_PATH


SPENDING_LIMIT_PATH = MODEL_DIR / 'T1_03_spending_limit_per_person.npy'
THEORETICAL_EXPENSE_PATH = MODEL_DIR / 'T1_03_theoretical_expense_per_person.npy'
REDUCTION_COEFFICIENT_PATH = MODEL_DIR / 'T1_03_reduction_coefficient.npy'
CONSTRAINED_MONTHLY_DEMAND_PATH = MODEL_DIR / 'T1_03_constrained_demand_monthly_by_service.npy'
CONSTRAINED_MONTHLY_DEMAND_BY_TYPE_PATH = MODEL_DIR / 'T1_03_constrained_demand_monthly_by_elderly_type.pkl'
CONSTRAINED_DAILY_DEMAND_PATH = MODEL_DIR / 'T1_03_constrained_demand_daily_by_service.npy'


def compute_spending_limit_per_person(monthly_income_per_capita, spending_cap_ratios):
    """Compute per-person monthly spending limits for communities and elderly types."""
    income = np.asarray(monthly_income_per_capita, dtype=float)
    caps = np.asarray(spending_cap_ratios, dtype=float)
    if income.shape != (10,):
        raise ValueError('各小区人均月收入应为 (10,)')
    if caps.shape != (3,):
        raise ValueError('消费上限比例应为 (3,)')
    return income[:, None] * caps[None, :]


def compute_theoretical_expense_per_person(demand_rate_matrix, service_prices):
    """Compute each elderly type's per-person theoretical monthly expense."""
    demand_rates = np.asarray(demand_rate_matrix, dtype=float)
    prices = np.asarray(service_prices, dtype=float)
    if demand_rates.shape != (3, 6):
        raise ValueError('需求率矩阵应为 (3, 6)')
    if prices.shape != (6,):
        raise ValueError('服务价格向量应为 (6,)')
    return demand_rates @ prices


def compute_reduction_coefficient(spending_limit_per_person, theoretical_expense_per_person):
    """Compute lambda values for communities and elderly types."""
    spending_limits = np.asarray(spending_limit_per_person, dtype=float)
    expenses = np.asarray(theoretical_expense_per_person, dtype=float)
    if spending_limits.shape != (10, 3):
        raise ValueError('单人消费上限矩阵应为 (10, 3)')
    if expenses.shape != (3,):
        raise ValueError('单人理论费用向量应为 (3,)')

    expense_row = expenses[None, :]
    ratio = np.divide(
        spending_limits,
        expense_row,
        out=np.ones_like(spending_limits, dtype=float),
        where=expense_row > 0
    )
    coefficient = np.minimum(1.0, ratio)
    if np.any((coefficient < 0) | (coefficient > 1)):
        raise ValueError('削减系数超出 [0, 1]')
    return coefficient


def apply_consumption_constraint(population_matrix_year5, demand_rate_matrix, reduction_coefficient):
    """Apply proportional demand reduction and keep splits by elderly type."""
    population = np.asarray(population_matrix_year5, dtype=float)
    demand_rates = np.asarray(demand_rate_matrix, dtype=float)
    coefficient = np.asarray(reduction_coefficient, dtype=float)
    if population.shape != (10, 3):
        raise ValueError('第 5 年人口矩阵应为 (10, 3)')
    if demand_rates.shape != (3, 6):
        raise ValueError('需求率矩阵应为 (3, 6)')
    if coefficient.shape != (10, 3):
        raise ValueError('削减系数矩阵应为 (10, 3)')

    constrained_by_type = {}
    for idx, key in enumerate(ELDERLY_TYPE_KEYS):
        constrained_rate = demand_rates[idx, :][None, :] * coefficient[:, idx:idx + 1]
        constrained_by_type[key] = population[:, idx:idx + 1] * constrained_rate

    constrained_by_type['total'] = sum(
        constrained_by_type[key] for key in ELDERLY_TYPE_KEYS
    )
    return constrained_by_type


def convert_monthly_to_daily(monthly_demand, days_per_month):
    """Convert monthly demand to daily demand for Q2 capacity comparisons."""
    if days_per_month <= 0:
        raise ValueError('days_per_month 必须为正数')
    return np.asarray(monthly_demand, dtype=float) / days_per_month


def save_constrained_demand_outputs(
    spending_limit_per_person,
    theoretical_expense_per_person,
    reduction_coefficient,
    constrained_monthly_by_type,
    constrained_daily_demand
):
    MODEL_DIR.mkdir(exist_ok=True)
    np.save(SPENDING_LIMIT_PATH, spending_limit_per_person)
    np.save(THEORETICAL_EXPENSE_PATH, theoretical_expense_per_person)
    np.save(REDUCTION_COEFFICIENT_PATH, reduction_coefficient)
    np.save(CONSTRAINED_MONTHLY_DEMAND_PATH, constrained_monthly_by_type['total'])
    with CONSTRAINED_MONTHLY_DEMAND_BY_TYPE_PATH.open('wb') as demand_file:
        pickle.dump(constrained_monthly_by_type, demand_file)
    np.save(CONSTRAINED_DAILY_DEMAND_PATH, constrained_daily_demand)


def validate_constrained_demand(
    constrained_monthly_demand,
    theoretical_monthly_demand,
    demand_rate_matrix,
    service_prices,
    spending_limit_per_person,
    reduction_coefficient
):
    if np.any(constrained_monthly_demand - theoretical_monthly_demand > 1e-8):
        raise ValueError('消费约束后需求超过理论需求')

    payable_expense = (
        np.asarray(demand_rate_matrix, dtype=float) @ np.asarray(service_prices, dtype=float)
    )[None, :] * reduction_coefficient
    if np.any(payable_expense - spending_limit_per_person > 1e-8):
        raise ValueError('消费约束后单人费用超过消费上限')


def main():
    config = load_t0_config(T0_CONFIG_PATH)
    population_year5 = np.load(YEAR5_POPULATION_PATH)
    theoretical_monthly_demand = np.load(THEORETICAL_DEMAND_PATH)

    spending_limits = compute_spending_limit_per_person(
        config['monthly_income_per_capita'],
        config['spending_cap_ratios']
    )
    theoretical_expenses = compute_theoretical_expense_per_person(
        config['demand_rate_matrix'],
        config['service_prices']
    )
    coefficient = compute_reduction_coefficient(spending_limits, theoretical_expenses)
    constrained_by_type = apply_consumption_constraint(
        population_year5,
        config['demand_rate_matrix'],
        coefficient
    )
    constrained_daily_demand = convert_monthly_to_daily(
        constrained_by_type['total'],
        config['time_constants']['days_per_month']
    )
    validate_constrained_demand(
        constrained_by_type['total'],
        theoretical_monthly_demand,
        config['demand_rate_matrix'],
        config['service_prices'],
        spending_limits,
        coefficient
    )
    save_constrained_demand_outputs(
        spending_limits,
        theoretical_expenses,
        coefficient,
        constrained_by_type,
        constrained_daily_demand
    )

    binding = coefficient < 1 - 1e-9
    print('========== T1 问题 1.3：消费约束后需求 ==========')
    print(f'单人消费上限矩阵: {spending_limits.shape}')
    print(f'单人理论费用: {theoretical_expenses.round(3).tolist()} 元/月')
    print(f'削减系数矩阵: {coefficient.shape}')
    print(f'触发削减的小区-老人类型组合数: {int(binding.sum())}')
    print(f"消费约束后月需求总量: {constrained_by_type['total'].sum():.2f} 次/月")
    print(f'消费约束后日需求矩阵: {constrained_daily_demand.shape}')
    print(f'已保存: {CONSTRAINED_MONTHLY_DEMAND_PATH}')
    print(f'已保存: {CONSTRAINED_DAILY_DEMAND_PATH}')
    print('================================================')


if __name__ == '__main__':
    main()
