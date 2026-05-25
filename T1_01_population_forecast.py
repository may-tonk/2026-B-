import pickle
from pathlib import Path

import numpy as np

from T0_validation_utils import validate_config


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / 'model'
T0_CONFIG_PATH = MODEL_DIR / 'T0_01_config.pkl'
FORECAST_HISTORY_PATH = MODEL_DIR / 'T1_01_forecast_history.pkl'
YEAR5_POPULATION_PATH = MODEL_DIR / 'T1_01_population_year5.npy'
ELDERLY_TYPE_NAMES = ['自理', '半失能', '失能']


def load_t0_config(config_path=T0_CONFIG_PATH):
    """Load and validate the standardized T0 configuration."""
    path = Path(config_path)
    with path.open('rb') as config_file:
        config = pickle.load(config_file)
    validate_config(config)
    return config


def forecast_population_year(population_current, transition_matrix, death_rate, growth_rate):
    """Forecast the next year-end population matrix for all communities."""
    population = np.asarray(population_current, dtype=float)
    transition = np.asarray(transition_matrix, dtype=float)

    if population.ndim != 2 or population.shape[1] != 3:
        raise ValueError('population_current 应为 (n_communities, 3) 矩阵')
    if transition.shape != (3, 3):
        raise ValueError('transition_matrix 应为 (3, 3) 矩阵')

    transitioned = (1 - death_rate) * population @ transition
    new_elderly = np.zeros_like(population, dtype=float)
    new_elderly[:, 0] = growth_rate * population.sum(axis=1)
    population_next = transitioned + new_elderly

    if np.any(population_next < 0):
        raise ValueError('人口预测出现负数')
    return population_next


def forecast_population(initial_population, transition_matrix, death_rate, growth_rate, years):
    """Return year-end population records for the requested forecast horizon."""
    if years < 1:
        raise ValueError('years 必须为正整数')

    population_current = np.asarray(initial_population, dtype=float).copy()
    history = []
    for year in range(1, years + 1):
        population_current = forecast_population_year(
            population_current,
            transition_matrix,
            death_rate,
            growth_rate
        )
        history.append({
            'year': year,
            'population_matrix': population_current.copy(),
            'total_by_community': population_current.sum(axis=1),
            'total_by_type': population_current.sum(axis=0),
            'total_all': float(population_current.sum())
        })
    return history


def forecast_population_5years(config):
    """Forecast Q1 population history and return the year-5 population matrix."""
    history = forecast_population(
        config['population_matrix'],
        config['transition_matrix'],
        config['transition_params']['death_rate'],
        config['transition_params']['elder_growth_rate'],
        years=5
    )
    return history, history[-1]['population_matrix']


def save_forecast_outputs(forecast_history, population_year5):
    MODEL_DIR.mkdir(exist_ok=True)
    with FORECAST_HISTORY_PATH.open('wb') as history_file:
        pickle.dump(forecast_history, history_file)
    np.save(YEAR5_POPULATION_PATH, population_year5)


def _format_type_totals(values):
    rounded = np.rint(values).astype(int)
    return ', '.join(
        f'{name}={rounded[idx]}'
        for idx, name in enumerate(ELDERLY_TYPE_NAMES)
    )


def main():
    config = load_t0_config()
    forecast_history, population_year5 = forecast_population_5years(config)
    save_forecast_outputs(forecast_history, population_year5)

    initial_population = config['population_matrix'].astype(float)
    initial_total = float(initial_population.sum())
    year5_total = float(population_year5.sum())

    print('========== T1 问题 1.1：老人数量递推预测 ==========')
    print(f'T0 配置: {T0_CONFIG_PATH}')
    print(f"死亡率: {config['transition_params']['death_rate']:.3f}")
    print(f"新增老人比例: {config['transition_params']['elder_growth_rate']:.3f}")
    print(f'初始人口矩阵: {initial_population.shape}，dtype={initial_population.dtype}')
    print(f'第 0 年末全区老人数: {initial_total:.2f}')
    for record in forecast_history:
        print(f"第 {record['year']} 年末全区老人数: {record['total_all']:.2f}")
    print(f'第 0 年类型合计: {_format_type_totals(initial_population.sum(axis=0))}')
    print(f'第 5 年类型合计: {_format_type_totals(population_year5.sum(axis=0))}')
    print(f'已保存: {FORECAST_HISTORY_PATH}')
    print(f'已保存: {YEAR5_POPULATION_PATH}')
    print('=================================================')


if __name__ == '__main__':
    main()
