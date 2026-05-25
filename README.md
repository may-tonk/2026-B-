# 2026电工杯比赛B题

本仓库整理了 2026 电工杯数学建模比赛 B 题“嵌入式社区养老服务站的建设与优化问题”的建模、求解与论文支撑材料。

项目围绕人口预测、养老服务站选址、服务定价与补贴、参数敏感性分析四个阶段展开，形成从数据读取到结果图表输出的完整计算流程。

## 项目结构

```text
.
├── data/                         # 赛题附件数据
├── docs/                         # 项目说明、分工与算法文档
├── tests/                        # 基础测试
├── modeling/                     # 可复用建模模块
├── T0_*.py                       # 数据读取、校验与基础表格输出
├── T1_*.py                       # 问题1：人口预测与需求计算
├── T2_*.py                       # 问题2：服务站选址与归属优化
├── T3_*.py                       # 问题3：定价、补贴与可及性分析
├── T4_*.py                       # 问题4：场景扰动、敏感性与鲁棒性分析
├── T*_table/                     # 各阶段 LaTeX 表格结果
├── T*_figure/                    # 各阶段图形结果
├── 项目总文档.md
├── 模型公式与符号说明.md
└── requirements.txt
```

## 环境配置

建议使用 Python 3.10+。

```bash
pip install -r requirements.txt
```

主要依赖包括：

- numpy
- pandas
- matplotlib
- openpyxl
- pytest

## 运行顺序

按阶段顺序运行脚本即可复现实验流程：

```bash
python T0_01_data_loader.py
python T0_02_validation.py
python T0_03_output_tables.py

python T1_01_population_forecast.py
python T1_02_theoretical_demand.py
python T1_03_constrained_demand.py
python T1_04_output_tables.py

python T2_01_layout_enumeration.py
python T2_02_satisfaction_assignment.py
python T2_03_evaluation_and_ranking.py
python T2_04_output_tables_and_charts.py

python T3_01_build_financial_model.py
python T3_02_candidate_prices.py
python "T3_03_price_evaluation .py"
python T3_04_constraint_filtering.py
python "T3_05_optimal_pricing .py"
python "T3_06_accessibility_analysis .py"
python T3_07_output_tables_and_charts.py

python T4_01_scenario_config.py
python T4_02_scenario_solver.py
python T4_03_sensitivity_analysis.py
python T4_04_robustness_uncertainty.py
python T4_05_output_tables_and_charts.py
```

部分 T3 脚本文件名中保留了空格，运行时需要加引号。

## 核心结果

- 问题1：建立马尔可夫链人口预测模型，并结合消费能力约束修正服务需求。
- 问题2：在 120 万元建设预算和 1000 米服务半径约束下，求得 A、C、E 三站点布局。
- 问题3：在补贴上限与保本微利约束下，求解分站差异化定价方案。
- 问题4：针对人口压力、成本上升和预算宽松场景进行敏感性与鲁棒性分析。

主要论文支撑图表位于：

- `T1_04_table/`, `T1_04_figure/`
- `T2_04_table/`, `T2_04_figure/`
- `T3_07_table/`, `T3_07_figure/`
- `T4_05_table/`, `T4_05_figure/`

## 测试

```bash
pytest
```

## 说明

本仓库仅保留比赛建模与论文支撑所需的重要文件。运行过程中生成的二进制中间模型、缓存文件、虚拟环境和临时审稿文件未纳入版本管理。
