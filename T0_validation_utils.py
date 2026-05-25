import numpy as np


EXPECTED_COMMUNITIES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
EXPECTED_SERVICES = ['助餐', '日间照料', '上门护理', '康复理疗', '助浴', '紧急救助']


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_closed_segments(segments, lower_key, upper_key, final_upper, label):
    _require(segments, f'{label}满意度规则不能为空')
    _require(lower_key not in segments[0], f'{label}满意度首段不应设置下界')
    _require(upper_key in segments[0], f'{label}满意度首段缺少上界')

    previous_upper = segments[0][upper_key]
    for segment in segments[1:]:
        _require(lower_key in segment and upper_key in segment, f'{label}满意度中间段边界不完整')
        _require(np.isclose(segment[lower_key], previous_upper), f'{label}满意度分段边界不连续')
        _require(segment[upper_key] > segment[lower_key], f'{label}满意度分段上界应大于下界')
        previous_upper = segment[upper_key]

    _require(np.isclose(previous_upper, final_upper), f'{label}满意度最终上界错误')


def _validate_open_ended_segments(segments, lower_key, upper_key, label):
    _require(len(segments) >= 2, f'{label}满意度规则段数不足')
    _require(lower_key not in segments[0], f'{label}满意度首段不应设置下界')
    _require(upper_key in segments[0], f'{label}满意度首段缺少上界')
    _require(lower_key in segments[-1], f'{label}满意度末段缺少下界')
    _require(upper_key not in segments[-1], f'{label}满意度末段应为开区间')

    previous_upper = segments[0][upper_key]
    for segment in segments[1:-1]:
        _require(lower_key in segment and upper_key in segment, f'{label}满意度中间段边界不完整')
        _require(np.isclose(segment[lower_key], previous_upper), f'{label}满意度分段边界不连续')
        _require(segment[upper_key] > segment[lower_key], f'{label}满意度分段上界应大于下界')
        previous_upper = segment[upper_key]

    _require(np.isclose(segments[-1][lower_key], previous_upper), f'{label}满意度末段边界不连续')


def validate_config(config):
    """Validate the standardized T0 config before downstream use."""
    _require(config['population_matrix'].shape == (10, 3), '人口矩阵维度错误')
    _require(config['transition_matrix'].shape == (3, 3), '转移矩阵维度错误')
    _require(config['demand_rate_matrix'].shape == (3, 6), '需求率矩阵维度错误')
    _require(config['distance_matrix'].shape == (10, 10), '距离矩阵维度错误')
    _require(config['coverage_mask'].shape == (10, 10), '覆盖矩阵维度错误')
    _require(config['communities'] == EXPECTED_COMMUNITIES, '小区顺序应为 A-J')
    _require(config['service_names'] == EXPECTED_SERVICES, '服务顺序错误')

    population_sums = config['population_matrix'].sum(axis=1)
    _require(
        np.array_equal(population_sums, config['elderly_total']),
        '各小区三类老人数之和应等于 60+ 老人数'
    )
    _require(np.all(config['population_matrix'] >= 0), '人口数不能为负')
    _require(np.all(config['service_prices'] >= 0), '价格不能为负')
    _require(np.all(config['distance_matrix'] >= 0), '距离不能为负')
    _require(
        np.allclose(config['spending_cap_ratios'], [0.20, 0.25, 0.30]),
        '消费上限比例应为 [0.20, 0.25, 0.30]'
    )
    _require(config['service_prices'][5] == 0, '紧急救助单价应为 0')

    tm = config['transition_matrix']
    _require(np.isclose(tm[0, 0] + tm[0, 1], 1.0), '自理状态转移行和错误')
    _require(np.isclose(tm[1, 1] + tm[1, 2], 1.0), '半失能状态转移行和错误')
    _require(np.isclose(tm[2, 2], 1.0), '失能状态转移行和错误')
    _require(tm[0, 2] == 0, '自理不能直接转为失能')
    _require(tm[1, 0] == 0, '半失能不能转回自理')
    _require(tm[2, 0] == 0 and tm[2, 1] == 0, '失能不能恢复')

    _require(np.allclose(config['distance_matrix'], config['distance_matrix'].T), '距离矩阵应对称')
    _require(np.allclose(np.diag(config['distance_matrix']), 0), '距离矩阵对角线应为 0')
    _require(np.diag(config['coverage_mask']).all(), '覆盖矩阵对角线应为 1')

    expected_coverage = (config['distance_matrix'] <= 1000).astype(int)
    _require(np.array_equal(config['coverage_mask'], expected_coverage), '覆盖矩阵与距离阈值不一致')

    rules = config['satisfaction_rules']
    _require(len(rules['distance']['segments']) == 4, '距离满意度规则应为 4 段')
    _require(len(rules['response']['segments']) == 5, '响应满意度规则应为 5 段')
    _require(len(rules['price']['segments']) == 4, '价格满意度规则应为 4 段')
    _validate_closed_segments(
        rules['distance']['segments'],
        'min_distance',
        'max_distance',
        1000,
        '距离'
    )
    _validate_closed_segments(
        rules['response']['segments'],
        'min_utilization',
        'max_utilization',
        1.0,
        '响应'
    )
    _validate_open_ended_segments(
        rules['price']['segments'],
        'min_ratio',
        'max_ratio',
        '价格'
    )
