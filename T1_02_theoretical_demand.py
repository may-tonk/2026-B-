import pickle
from pathlib import Path

import numpy as np

from T1_01_population_forecast import (
    MODEL_DIR,
    T0_CONFIG_PATH,
    YEAR5_POPULATION_PATH,
    load_t0_config,
)


THEORETICAL_DEMAND_PATH = MODEL_DIR / 'T1_02_theoretical_demand_by_service.npy'
THEORETICAL_DEMAND_BY_TYPE_PATH = MODEL_DIR / 'T1_02_theoretical_demand_by_elderly_type.pkl'
ELDERLY_TYPE_KEYS = ['healthy', 'semi_disabled', 'disabled']
ELDERLY_TYPE_LABELS = {
    'healthy': '自理',
    'semi_disabled': '半失能',
    'disabled': '失能'
}


def compute_theoretical_demand(population_matrix_year5, demand_rate_matrix):
    """Compute total theoretical monthly demand by community and service."""
    population = np.asarray(population_matrix_year5, dtype=float)
    demand_rates = np.asarray(demand_rate_matrix, dtype=float)

    if population.shape != (10, 3):
        raise ValueError('第 5 年人口矩阵应为 (10, 3)')
    if demand_rates.shape != (3, 6):
        raise ValueError('需求率矩阵应为 (3, 6)')

    demand = population @ demand_rates
    if np.any(demand < 0):
        raise ValueError('理论需求出现负数')
    return demand


def compute_theoretical_demand_by_elderly_type(population_matrix_year5, demand_rate_matrix):
    """Compute theoretical monthly demand split by elderly type."""
    population = np.asarray(population_matrix_year5, dtype=float)
    demand_rates = np.asarray(demand_rate_matrix, dtype=float)
    demand_by_type = {}

    for idx, key in enumerate(ELDERLY_TYPE_KEYS):
        demand_by_type[key] = np.outer(population[:, idx], demand_rates[idx, :])

    demand_by_type['total'] = sum(demand_by_type[key] for key in ELDERLY_TYPE_KEYS)
    return demand_by_type


def save_theoretical_demand_outputs(theoretical_demand, demand_by_type):
    MODEL_DIR.mkdir(exist_ok=True)
    np.save(THEORETICAL_DEMAND_PATH, theoretical_demand)
    with THEORETICAL_DEMAND_BY_TYPE_PATH.open('wb') as demand_file:
        pickle.dump(demand_by_type, demand_file)


def main():
    config = load_t0_config(T0_CONFIG_PATH)
    population_year5 = np.load(YEAR5_POPULATION_PATH)
    theoretical_demand = compute_theoretical_demand(
        population_year5,
        config['demand_rate_matrix']
    )
    demand_by_type = compute_theoretical_demand_by_elderly_type(
        population_year5,
        config['demand_rate_matrix']
    )
    save_theoretical_demand_outputs(theoretical_demand, demand_by_type)

    print('========== T1 问题 1.2：理论月需求 ==========')
    print(f'第 5 年人口矩阵: {population_year5.shape}')
    print(f"需求率矩阵: {config['demand_rate_matrix'].shape}")
    print(f'理论需求矩阵: {theoretical_demand.shape}')
    print(f'全区理论月需求总量: {theoretical_demand.sum():.2f} 次/月')
    for key in ELDERLY_TYPE_KEYS:
        print(f"{ELDERLY_TYPE_LABELS[key]}理论月需求: {demand_by_type[key].sum():.2f} 次/月")
    print(f'已保存: {THEORETICAL_DEMAND_PATH}')
    print(f'已保存: {THEORETICAL_DEMAND_BY_TYPE_PATH}')
    print('===========================================')


if __name__ == '__main__':
    main()
