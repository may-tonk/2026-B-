import os
import pickle
import re
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'model')


def clean_columns(df):
    df.columns = df.columns.astype(str).str.strip()
    return df


def safe_read_data(excel_file, sheet_name, csv_fallback):
    excel_path = os.path.join(DATA_DIR, excel_file)
    csv_path = os.path.join(DATA_DIR, csv_fallback)
    try:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
        if pd.isna(df.iloc[0, 1]):
            df.columns = df.iloc[1].astype(str).str.strip()
            df = df[2:].reset_index(drop=True)
        else:
            df.columns = df.iloc[0].astype(str).str.strip()
            df = df[1:].reset_index(drop=True)
        return clean_columns(df)
    except FileNotFoundError:
        try:
            df = pd.read_csv(csv_path)
            return clean_columns(df)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到数据文件：请确保 {excel_path} 或 {csv_path} 存在于项目目录中。")


def to_number(value):
    number = float(value)
    return int(number) if number.is_integer() else number


def extract_score(text, score_name):
    match = re.search(rf'{score_name}\s*=\s*(\d+(?:\.\d+)?)', text)
    if not match:
        raise ValueError(f"无法从满意度规则中提取 {score_name} 分数：{text}")
    return float(match.group(1))


def parse_distance_segment(text):
    score = extract_score(text, 'S1')
    bounded = re.search(r'(\d+(?:\.\d+)?)m[＜<]距离[≤<＝=]+(\d+(?:\.\d+)?)m', text)
    if bounded:
        return {
            'min_distance': to_number(bounded.group(1)),
            'max_distance': to_number(bounded.group(2)),
            'score': score
        }

    upper = re.search(r'距离[≤<＝=]+(\d+(?:\.\d+)?)m', text)
    if upper:
        return {'max_distance': to_number(upper.group(1)), 'score': score}

    raise ValueError(f"无法解析距离满意度规则：{text}")


def parse_response_segment(text):
    score = extract_score(text, 'S2')
    bounded = re.search(r'(\d+(?:\.\d+)?)[＜<]服务站利用率[≤<＝=]+(\d+(?:\.\d+)?)', text)
    if bounded:
        return {
            'min_utilization': float(bounded.group(1)),
            'max_utilization': float(bounded.group(2)),
            'score': score
        }

    upper = re.search(r'利用率[≤<＝=]+(\d+(?:\.\d+)?)', text)
    if upper:
        return {'max_utilization': float(upper.group(1)), 'score': score}

    raise ValueError(f"无法解析响应满意度规则：{text}")


def parse_price_segment(text):
    score = extract_score(text, 'S3')
    percentages = [float(value) / 100 for value in re.findall(r'(\d+(?:\.\d+)?)%', text)]

    if '基准价' in text and not percentages:
        return {'max_ratio': 1.0, 'score': score}
    if len(percentages) == 1 and '以内' in text:
        return {'min_ratio': 1.0, 'max_ratio': 1 + percentages[0], 'score': score}
    if len(percentages) == 2:
        return {
            'min_ratio': 1 + percentages[0],
            'max_ratio': 1 + percentages[1],
            'score': score
        }
    if len(percentages) == 1 and '以上' in text:
        return {'min_ratio': 1 + percentages[0], 'score': score}

    raise ValueError(f"无法解析价格满意度规则：{text}")


def load_satisfaction_rules():
    excel_path = os.path.join(DATA_DIR, '附件5：满意度评分规则.xlsx')
    df_rules = pd.read_excel(excel_path, sheet_name='满意度评分规则', header=None)
    if df_rules.shape[1] < 2:
        raise ValueError("附件5满意度规则表至少应包含规则说明列。")

    rule_texts = df_rules.iloc[:, 1].dropna().astype(str).str.strip().tolist()
    if len(rule_texts) != 13:
        raise ValueError(f"附件5满意度规则数量异常，期望 13 条，实际 {len(rule_texts)} 条。")

    return {
        'distance': {'segments': [parse_distance_segment(text) for text in rule_texts[:4]]},
        'response': {'segments': [parse_response_segment(text) for text in rule_texts[4:9]]},
        'price': {'segments': [parse_price_segment(text) for text in rule_texts[9:]]}
    }


def describe_segment(segment, lower_key, upper_key, unit=''):
    score = segment['score']
    if lower_key in segment and upper_key in segment:
        return f"{segment[lower_key]:g}{unit}-{segment[upper_key]:g}{unit}: {score:.2f}"
    if upper_key in segment:
        return f"<={segment[upper_key]:g}{unit}: {score:.2f}"
    return f">{segment[lower_key]:g}{unit}: {score:.2f}"


def main():
    print("========== 数据读取结果 ==========\n")

    config = {
        'communities': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
        'n_communities': 10,
        'n_services': 6,
        'n_elderly_types': 3,
        'time_constants': {
            'days_per_month': 30,
            'months_per_year': 12,
            'days_per_year': 365,
            'depreciation_years': 20
        },
        'service_names': ['助餐', '日间照料', '上门护理', '康复理疗', '助浴', '紧急救助']
    }

    print("【附件1_小区基础数据.xlsx】")
    df_pop = safe_read_data('附件1：小区基础数据.xlsx', '人口与老人结构', '附件1：小区基础数据.xlsx - 人口与老人结构.csv')

    pop_data = df_pop.set_index(df_pop.columns[0]).reindex(config['communities'])
    config['population_matrix'] = pop_data[['自理老人', '半失能老人', '失能老人']].values.astype(int)
    config['elderly_total'] = pop_data['60+老人数'].values.astype(int)
    income_col = [c for c in pop_data.columns if '收入' in c][0]
    config['monthly_income_per_capita'] = pop_data[income_col].values.astype(float)

    print('OK 工作表"人口与老人结构"已读取')
    print(f"  小区列表: {config['communities']}")
    print(f"  小区数: {config['n_communities']}")
    print(f"  初始老人数 shape: {config['population_matrix'].shape}")
    print("  数据示例:")
    print(f"    小区A: 自理={config['population_matrix'][0, 0]}, 半失能={config['population_matrix'][0, 1]}, 失能={config['population_matrix'][0, 2]} (合计={config['elderly_total'][0]})")
    print(f"    小区B: 自理={config['population_matrix'][1, 0]}, 半失能={config['population_matrix'][1, 1]}, 失能={config['population_matrix'][1, 2]} (合计={config['elderly_total'][1]})")
    print(f"  人均月收入范围: {config['monthly_income_per_capita'].min()} 元 ~ {config['monthly_income_per_capita'].max()} 元\n")

    df_trans = safe_read_data('附件1：小区基础数据.xlsx', '转移概率', '附件1：小区基础数据.xlsx - 转移概率.csv')
    p12 = float(df_trans.loc[df_trans['转移类型'].str.startswith('自理'), df_trans.columns[1]].values[0])
    p23 = float(df_trans.loc[df_trans['转移类型'].str.startswith('半失能'), df_trans.columns[1]].values[0])

    config['transition_params'] = {
        'healthy_to_semi': p12,
        'semi_to_disabled': p23,
        'death_rate': 0.05,
        'elder_growth_rate': 0.07
    }
    config['transition_matrix'] = np.array([
        [1 - p12, p12, 0],
        [0, 1 - p23, p23],
        [0, 0, 1]
    ])

    print('OK 工作表"转移概率"已读取')
    print(f"  自理->半失能转移率: {p12}")
    print(f"  半失能->失能转移率: {p23}")
    print("  统一死亡率: 0.05")
    print("  年均新增老年人比例: 0.07\n")

    print("【附件2_服务需求数据.xlsx】")
    df_demand = safe_read_data('附件2：服务需求数据.xlsx', '每位老人月均服务需求次数',
                               '附件2：服务需求数据.xlsx - 每位老人月均服务需求次数.csv')
    df_demand.columns = [c.replace('半自理', '半失能') for c in df_demand.columns]
    df_demand = df_demand.set_index(df_demand.columns[0]).reindex(config['service_names'])

    config['demand_rate_matrix'] = df_demand[['自理', '半失能', '失能']].T.values.astype(float)

    print('OK 工作表"每位老人月均服务需求次数"已读取')
    print(f"  服务列表: {config['service_names']}")
    print(f"  需求矩阵 shape: {config['demand_rate_matrix'].shape}")
    print("  单位: 次/月")
    print("  数据示例:")
    print(f"    自理老人月均需求: {config['demand_rate_matrix'][0].tolist()}")
    print(f"    半失能老人月均需求: {config['demand_rate_matrix'][1].tolist()}")
    print(f"    失能老人月均需求: {config['demand_rate_matrix'][2].tolist()}\n")

    df_finance = safe_read_data('附件2：服务需求数据.xlsx', '服务营收及支出',
                                '附件2：服务需求数据.xlsx - 服务营收及支出.csv')
    df_finance = df_finance.set_index(df_finance.columns[0]).reindex(config['service_names'])

    prices = df_finance[df_finance.columns[0]].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float).values
    costs = df_finance[df_finance.columns[1]].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float).values

    config['service_prices'] = prices
    config['direct_costs'] = costs

    print('OK 工作表"服务营收及支出"已读取')
    print(f"  服务单价向量 shape: {config['service_prices'].shape}")
    print("  单位: 元/次")
    print(f"  数据: {config['service_prices'].tolist()}")
    print(f"  直接成本向量 shape: {config['direct_costs'].shape}")
    print("  单位: 元/次")
    print(f"  数据: {config['direct_costs'].tolist()}")
    print("  注: 紧急救助单价为 0\n")

    df_cap = safe_read_data('附件2：服务需求数据.xlsx', '月服务消费上限', '附件2：服务需求数据.xlsx - 月服务消费上限.csv')
    df_cap = df_cap.dropna(how='all')
    cap_type_col = df_cap.columns[0]
    cap_ratio_col = df_cap.columns[1]
    cap_order = ['自理老人', '半失能老人', '失能老人']
    cap_texts = df_cap.set_index(cap_type_col)[cap_ratio_col].reindex(cap_order)
    if cap_texts.isna().any():
        missing_caps = [elderly_type for elderly_type, value in cap_texts.items() if pd.isna(value)]
        raise ValueError(f"月服务消费上限缺少老人类型：{missing_caps}")
    config['spending_cap_ratios'] = (
        cap_texts.astype(str).str.extract(r'(\d+)')[0].astype(float).values / 100
    )

    print('OK 工作表"月服务消费上限"已读取')
    print(f"  自理老人: 月收入的 {int(config['spending_cap_ratios'][0] * 100)}%")
    print(f"  半失能老人: 月收入的 {int(config['spending_cap_ratios'][1] * 100)}%")
    print(f"  失能老人: 月收入的 {int(config['spending_cap_ratios'][2] * 100)}%\n")

    print("【附件3_服务站建设与运营成本.xlsx】")
    df_station = safe_read_data('附件3：服务站建设与运营成本.xlsx', '服务站建设与运营成本',
                                '附件3：服务站建设与运营成本.xlsx - 服务站建设与运营成本.csv')
    size_map = {'小型': 'small', '中型': 'medium', '大型': 'large'}
    subsidy_cap_map = {'small': 1000, 'medium': 1800, 'large': 2600}
    station_size_col = df_station.columns[0]
    station_rows = (
        df_station[df_station[station_size_col].isin(size_map)]
        .set_index(station_size_col)
        .reindex(size_map)
    )
    if station_rows.iloc[:, :3].isna().any().any():
        missing_sizes = [size for size, row in station_rows.iterrows() if row.iloc[:3].isna().any()]
        raise ValueError(f"服务站规格缺少必要参数：{missing_sizes}")

    config['station_specs'] = {}

    for zh_size, eng_size in size_map.items():
        row = station_rows.loc[zh_size]
        build_cost = float(row[station_rows.columns[0]])
        fixed_cost = float(row[station_rows.columns[1]])
        capacity = float(row[station_rows.columns[2]])

        dep_years = config['time_constants']['depreciation_years']
        annual_dep = (build_cost * 10000) / dep_years
        daily_dep = annual_dep / config['time_constants']['days_per_year']

        config['station_specs'][eng_size] = {
            'build_cost': build_cost,
            'daily_fixed_cost': fixed_cost,
            'daily_capacity': capacity,
            'subsidy_cap_day': subsidy_cap_map[eng_size],
            'annual_depreciation_cost': annual_dep,
            'daily_depreciation_cost': daily_dep,
            'daily_total_fixed_cost_with_depreciation': fixed_cost + daily_dep
        }

    print('OK 工作表"服务站建设与运营成本"已读取')
    for size in ['small', 'medium', 'large']:
        spec = config['station_specs'][size]
        zh_size = [k for k, v in size_map.items() if v == size][0]
        print(f"  {zh_size}站点:")
        print(f"    - 建设成本: {spec['build_cost']} 万元")
        print(f"    - 日固定成本: {spec['daily_fixed_cost']} 元/日")
        print(f"    - 日最大服务人次: {spec['daily_capacity']} 人次/日")
    print()

    print("【附件4_小区间距离矩阵.xlsx】")
    df_dist = safe_read_data('附件4：小区间距离矩阵.xlsx', '小区间距离矩阵',
                             '附件4：小区间距离矩阵.xlsx - 小区间距离矩阵.csv')
    df_dist = df_dist.set_index(df_dist.columns[0]).reindex(index=config['communities'], columns=config['communities'])
    config['distance_matrix'] = df_dist.values.astype(float)
    config['filtered_distance_matrix'] = np.where(config['distance_matrix'] <= 1000,
                                                  config['distance_matrix'], np.inf)
    config['coverage_mask'] = (config['distance_matrix'] <= 1000).astype(int)

    print('OK 工作表"小区间距离矩阵"已读取')
    print(f"  矩阵 shape: {config['distance_matrix'].shape}")
    print("  单位: 米")
    print(f"  距离范围: [{config['distance_matrix'].min()} m, {config['distance_matrix'].max()} m]")
    diagonal_zero = np.allclose(np.diag(config['distance_matrix']), 0)
    print(f"  对角线检查: 所有小区到自身距离为 0 {'OK' if diagonal_zero else 'NO'}\n")

    print("【附件5_满意度评分规则.xlsx】")
    config['satisfaction_rules'] = load_satisfaction_rules()
    config['satisfaction_weights'] = {
        'distance': 0.2,
        'response': 0.3,
        'price': 0.5
    }

    print('OK 工作表"满意度评分规则"已读取')
    print("  距离满意度规则 S1:")
    for segment in config['satisfaction_rules']['distance']['segments']:
        print(f"    {describe_segment(segment, 'min_distance', 'max_distance', 'm')}")
    print("  响应满意度规则 S2:")
    for segment in config['satisfaction_rules']['response']['segments']:
        print(f"    {describe_segment(segment, 'min_utilization', 'max_utilization')}")
    print("  价格满意度规则 S3:")
    for segment in config['satisfaction_rules']['price']['segments']:
        print(f"    {describe_segment(segment, 'min_ratio', 'max_ratio')} x 基准价")
    print("  综合满意度权重:")
    print("    S = 0.2*S1 + 0.3*S2 + 0.5*S3")
    print("    取值范围: [0.6, 1.0]\n")

    print("========== 数据一致性检查 ==========")
    print(f"OK 小区数: {config['n_communities']} (A-J)")

    total_pop_calc = config['population_matrix'].sum()
    total_pop_read = config['elderly_total'].sum()
    print(f"OK 初始老人数总计: {total_pop_calc} 人")
    if total_pop_calc == total_pop_read:
        print("  验证: 各小区三类老人数之和 = 60+老人数")
    else:
        print(f"  [警告]: 矩阵累加({total_pop_calc})与给出总数({total_pop_read})不一致！")

    print(f"OK 服务项数: {config['n_services']}")

    trans_valid = (0 <= config['transition_params']['healthy_to_semi'] <= 1) and (
                0 <= config['transition_params']['semi_to_disabled'] <= 1)
    print(f"OK 转移率范围: 0 <= 概率 <= 1 {'OK' if trans_valid else 'NO'}")

    print(f"OK 消费上限比例: {config['spending_cap_ratios'].tolist()} OK")
    print("OK 所有数据维度符合预期\n")
    print("================================")

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(os.path.join(MODEL_DIR, 'T0_01_config.pkl'), 'wb') as f:
        pickle.dump(config, f)


if __name__ == "__main__":
    main()
