import os
import pickle
from T0_validation_utils import validate_config


def save_latex_table(filename, content):
    """保存 LaTeX 表格代码到指定文件"""
    filepath = os.path.join('T0_03_table', filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f"OK {filename}")


def format_segment_range(segment, lower_key, upper_key, variable, unit=''):
    if lower_key in segment and upper_key in segment:
        interval = f"${segment[lower_key]:g} < {variable} \\leq {segment[upper_key]:g}$"
    elif upper_key in segment:
        interval = f"${variable} \\leq {segment[upper_key]:g}$"
    else:
        interval = f"${variable} > {segment[lower_key]:g}$"
    return f"{interval} {unit}".rstrip()


def format_rule_rows(segments, lower_key, upper_key, variable, unit=''):
    return '\n'.join(
        f"{format_segment_range(segment, lower_key, upper_key, variable, unit)} & {segment['score']:.2f} \\\\"
        for segment in segments
    )


def main():
    print("========== 表格生成 ==========")

    # 确保输出目录存在
    os.makedirs('T0_03_table', exist_ok=True)

    # 加载已校验的数据
    try:
        with open('model/T0_01_config.pkl', 'rb') as f:
            config = pickle.load(f)
    except FileNotFoundError:
        print("错误：找不到 model/T0_01_config.pkl，请先运行 T0_01_data_loader.py")
        return

    validate_config(config)

    # ==========================================
    # 表格 1：小区基础信息与初始老人结构
    # ==========================================
    t1_lines = []
    total_all = config['elderly_total'].sum()
    for i, comm in enumerate(config['communities']):
        income = int(config['monthly_income_per_capita'][i])
        zili = config['population_matrix'][i, 0]
        ban = config['population_matrix'][i, 1]
        shi = config['population_matrix'][i, 2]
        total = config['elderly_total'][i]
        ratio = (total / total_all) * 100
        t1_lines.append(f"{comm} & {income} & {zili} & {ban} & {shi} & {total} & {ratio:.1f} \\\\")

    total_zili = config['population_matrix'][:, 0].sum()
    total_ban = config['population_matrix'][:, 1].sum()
    total_shi = config['population_matrix'][:, 2].sum()

    table1 = r"""\begin{table}[h]
\centering
\caption{10个小区基础信息与初始老人结构}
\begin{tabular}{|c|c|c|c|c|c|c|}
\hline
小区 & 月收入(元) & 自理 & 半失能 & 失能 & 合计 & 比例(\%) \\
\hline
""" + "\n".join(t1_lines) + r"""
\hline
总计 & — & """ + f"{total_zili} & {total_ban} & {total_shi} & {total_all}" + r""" & 100.0 \\
\hline
\end{tabular}
\end{table}"""
    save_latex_table('T0_03_communities_population.tex', table1)

    # ==========================================
    # 表格 2：老人状态转移参数
    # ==========================================
    p12 = config['transition_params']['healthy_to_semi']
    p23 = config['transition_params']['semi_to_disabled']
    death = config['transition_params']['death_rate']
    table2 = r"""\begin{table}[h]
\centering
\caption{老人健康状态转移概率与死亡率}
\begin{tabular}{|c|c|c|}
\hline
参数 & 数值 & 说明 \\
\hline
自理 $\rightarrow$ 半失能 & """ + str(p12) + r""" & 年转移率 \\
半失能 $\rightarrow$ 失能 & """ + str(p23) + r""" & 年转移率 \\
统一死亡率 & """ + str(death) + r""" & 各老人状态适用 \\
\hline
\end{tabular}
\end{table}"""
    save_latex_table('T0_03_transition_parameters.tex', table2)

    # ==========================================
    # 表格 3：6项服务需求与经济参数
    # ==========================================
    t3_lines = []
    for i, service in enumerate(config['service_names']):
        d_zili = config['demand_rate_matrix'][0, i]
        d_ban = config['demand_rate_matrix'][1, i]
        d_shi = config['demand_rate_matrix'][2, i]
        price = config['service_prices'][i]
        cost = config['direct_costs'][i]

        # 格式化显示（去除不必要的小数点）
        d_zili_str = f"{d_zili:g}"
        d_ban_str = f"{d_ban:g}"
        d_shi_str = f"{d_shi:g}"
        price_str = f"{price:g}"
        cost_str = f"{cost:g}"

        attr = "公益免费" if price == 0 else "付费"
        t3_lines.append(
            f"{service} & {d_zili_str} & {d_ban_str} & {d_shi_str} & {price_str} & {cost_str} & {attr} \\\\")

    table3 = r"""\begin{table}[h]
\centering
\caption{6项服务的月需求频次、单价与直接成本}
\begin{tabular}{|c|c|c|c|c|c|c|}
\hline
服务项目 & 自理(次/月) & 半失能(次/月) & 失能(次/月) & 单价(元/次) & 直接成本(元/次) & 定价属性 \\
\hline
""" + "\n".join(t3_lines) + r"""
\hline
\end{tabular}
\end{table}"""
    save_latex_table('T0_03_service_parameters.tex', table3)

    # ==========================================
    # 表格 4：消费约束与站点规格
    # ==========================================
    caps = config['spending_cap_ratios']
    s_small = config['station_specs']['small']
    s_medium = config['station_specs']['medium']
    s_large = config['station_specs']['large']

    table4 = r"""\begin{table}[h]
\centering
\caption{月消费上限比例与服务站规格参数}

\vspace{0.3cm}

(1) 消费上限比例\\
\begin{tabular}{|c|c|}
\hline
老人类型 & 月消费上限比例 \\
\hline
自理 & """ + f"{int(caps[0] * 100)}\\%" + r""" (月收入) \\
半失能 & """ + f"{int(caps[1] * 100)}\\%" + r""" (月收入) \\
失能 & """ + f"{int(caps[2] * 100)}\\%" + r""" (月收入) \\
\hline
\end{tabular}

\vspace{0.3cm}

(2) 服务站规格\\
\begin{tabular}{|c|c|c|c|}
\hline
站点规模 & 建设成本(万元) & 日固定成本(元/日) & 日最大服务人次 \\
\hline
小型 & """ + f"{s_small['build_cost']:g} & {s_small['daily_fixed_cost']:g} & {s_small['daily_capacity']:g}" + r""" \\
中型 & """ + f"{s_medium['build_cost']:g} & {s_medium['daily_fixed_cost']:g} & {s_medium['daily_capacity']:g}" + r""" \\
大型 & """ + f"{s_large['build_cost']:g} & {s_large['daily_fixed_cost']:g} & {s_large['daily_capacity']:g}" + r""" \\
\hline
\end{tabular}
\end{table}"""
    save_latex_table('T0_03_station_specs_and_limits.tex', table4)

    # ==========================================
    # 表格 5-8：满意度评分标准与权重
    # ==========================================
    distance_segments = config['satisfaction_rules']['distance']['segments']
    response_segments = config['satisfaction_rules']['response']['segments']
    price_segments = config['satisfaction_rules']['price']['segments']
    weights = config['satisfaction_weights']

    distance_rows = format_rule_rows(distance_segments, 'min_distance', 'max_distance', 'd', 'm')
    distance_limit = distance_segments[-1]['max_distance']
    table5 = r"""\begin{table}[h]
\centering
\caption{距离满意度评分标准 $S_1$（来自附件5）}
\begin{tabular}{|c|c|}
\hline
距离范围 & 满意度分数 \\
\hline
""" + distance_rows + r"""
$d > """ + f"{distance_limit:g}" + r"""$ m & 无有效服务 \\
\hline
\end{tabular}
\end{table}"""
    save_latex_table('T0_03_distance_satisfaction_rules.tex', table5)

    response_rows = format_rule_rows(
        response_segments,
        'min_utilization',
        'max_utilization',
        r'\rho'
    )
    table6 = r"""\begin{table}[h]
\centering
\caption{响应满意度评分标准 $S_2$（站点利用率）}
\begin{tabular}{|c|c|}
\hline
站点利用率 & 满意度分数 \\
\hline
""" + response_rows + r"""
\hline
\end{tabular}
\end{table}"""
    save_latex_table('T0_03_response_satisfaction_rules.tex', table6)

    price_rows = format_rule_rows(
        price_segments,
        'min_ratio',
        'max_ratio',
        r'p / p_{base}'
    )
    table7 = r"""\begin{table}[h]
\centering
\caption{价格满意度评分标准 $S_3$（基于基准价）}
\begin{tabular}{|c|c|}
\hline
价格比例 & 满意度分数 \\
\hline
""" + price_rows + r"""
\hline
\end{tabular}
\end{table}"""
    save_latex_table('T0_03_price_satisfaction_rules.tex', table7)

    weight_rows = '\n'.join([
        f"距离满意度 $S_1$ & {weights['distance']:g} & 服务可及性 \\\\",
        f"响应满意度 $S_2$ & {weights['response']:g} & 服务质量/响应速度 \\\\",
        f"价格满意度 $S_3$ & {weights['price']:g} & 经济可承受性 \\\\"
    ])
    satisfaction_formula = (
        f"$S = {weights['distance']:g}S_1 + {weights['response']:g}S_2 + "
        f"{weights['price']:g}S_3$"
    )
    table8 = r"""\begin{table}[h]
\centering
\caption{综合满意度计算权重与取值范围}
\begin{tabular}{|c|c|c|}
\hline
因素 & 权重 & 说明 \\
\hline
""" + weight_rows + r"""
\hline
综合满意度 & """ + satisfaction_formula + r""" & 取值: $[0.6, 1.0]$ \\
\hline
\end{tabular}
\end{table}"""
    save_latex_table('T0_03_satisfaction_weights.tex', table8)

    print("\n所有输出文件生成完毕!")
    print("================================")


if __name__ == "__main__":
    main()
