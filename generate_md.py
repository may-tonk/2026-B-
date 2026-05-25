import pandas as pd
import os

def read_excel_file(file_path):
    try:
        return pd.read_excel(file_path, sheet_name=None, header=None)
    except Exception as e:
        print(f"读取文件 {file_path} 失败: {e}")
        return None

def df_to_markdown(df, sheet_name):
    md = f"## {sheet_name}\n\n"
    df = df.dropna(how='all', axis=1)
    df = df.dropna(how='all', axis=0)
    df = df.fillna('')
    if len(df) == 0:
        return md
    cols = len(df.columns)
    header = "|" + "|".join([str(c) for c in df.iloc[0]]) + "|\n"
    separator = "|" + "|".join(["-" * max(len(str(df.iloc[0, j])), 3) for j in range(cols)]) + "|\n"
    md += header
    md += separator
    for i in range(1, len(df)):
        row = "|" + "|".join([str(df.iloc[i, j]) for j in range(cols)]) + "|\n"
        md += row
    md += "\n"
    return md

def generate_md():
    files = [
        ("附件1：小区基础数据.xlsx", "小区基础数据"),
        ("附件2：服务需求数据.xlsx", "服务需求数据"),
        ("附件3：服务站建设与运营成本.xlsx", "服务站建设与运营成本"),
        ("附件4：小区间距离矩阵.xlsx", "小区间距离矩阵"),
        ("附件5：满意度评分规则.xlsx", "满意度评分规则")
    ]
    
    md_content = "# 嵌入式社区养老服务站数据汇总\n\n"
    
    for filename, title in files:
        file_path = os.path.join("data", filename)
        print(f"正在处理: {file_path}")
        
        sheets = read_excel_file(file_path)
        if sheets is None:
            md_content += f"## {title}\n\n**读取失败**\n\n"
            continue
        
        md_content += f"# {title}\n\n"
        
        for sheet_name, df in sheets.items():
            md_content += df_to_markdown(df, sheet_name)
    
    output_path = "data汇总.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"Markdown文档已生成: {output_path}")

if __name__ == "__main__":
    generate_md()