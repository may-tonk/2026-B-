from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


OUT_DIR = Path("docs")
OUT_PATH = OUT_DIR / "论文总流程图_调整版.png"


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def add_box(ax, x, y, w, h, text, fc, ec="none", color="black", size=12, weight="bold"):
    rect = Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=1.2)
    ax.add_patch(rect)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=size,
        color=color,
        fontweight=weight,
        linespacing=1.25,
    )
    return rect


def add_panel(ax, x, y, w, h, title, fc):
    rect = Rectangle(
        (x, y),
        w,
        h,
        facecolor=fc,
        edgecolor="black",
        linewidth=1.8,
        linestyle=(0, (5, 4)),
    )
    ax.add_patch(rect)
    ax.text(x + 0.15, y + h + 0.20, title, ha="left", va="bottom", fontsize=16, fontweight="bold")
    return rect


def arrow(ax, start, end, color="black", lw=1.2, mutation=12):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=mutation,
            linewidth=lw,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 9), dpi=220)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("#F4FBF9")
    ax.set_facecolor("#F4FBF9")

    # Panel coordinates
    p1 = (0.8, 5.25, 6.9, 3.05)
    p2 = (0.8, 0.65, 6.9, 3.65)
    p4 = (8.6, 5.30, 6.6, 3.05)
    p3 = (8.6, 0.65, 6.6, 3.65)

    add_panel(ax, *p1, "问题1：人口预测与需求分析", "#FFF8E7")
    add_panel(ax, *p2, "问题2：站点布局优化", "#EAF4FE")
    add_panel(ax, *p3, "问题3：定价策略与可及性分析", "#FDECEC")
    add_panel(ax, *p4, "问题4：敏感度分析与鲁棒性评价", "#EEF8EA")

    # Q1
    add_box(ax, 1.35, 7.42, 2.25, 0.48, "马尔科夫链模型", "#1F4E79", color="white", size=11)
    add_box(ax, 4.55, 7.32, 2.65, 0.68, "预测第5年各小区\n老人数量与结构", "#7C7A73", color="white", size=10)
    arrow(ax, (3.65, 7.66), (4.48, 7.66))

    add_box(ax, 1.35, 6.22, 2.25, 0.48, "消费能力约束模型", "#1F4E79", color="white", size=11)
    add_box(ax, 4.55, 6.12, 2.65, 0.68, "计算消费约束后的\n实际服务需求", "#7C7A73", color="white", size=10)
    arrow(ax, (3.65, 6.46), (4.48, 6.46))

    # Q2
    add_box(ax, 1.20, 3.25, 2.20, 0.52, "全局枚举", "#B7D5EC", size=11)
    add_box(ax, 1.20, 2.58, 2.20, 0.52, "预算约束剪枝", "#B7D5EC", size=11)
    add_box(ax, 1.20, 1.72, 2.45, 0.52, "1000米服务半径约束", "#B9CDED", size=10)
    add_box(ax, 1.20, 1.05, 2.45, 0.52, "覆盖率-满意度-利润排序", "#B9CDED", size=10)

    add_box(ax, 4.55, 2.92, 2.05, 0.66, "生成预算可行的\n站点配置方案", "#B7D5EC", size=10)
    add_box(ax, 4.55, 1.88, 2.90, 0.58, "最高满意度唯一归属\n并校验2%容量弹性", "#B7D5EC", size=10)
    add_box(ax, 4.55, 1.02, 2.90, 0.58, "输出最优站点位置、规模\n和小区归属", "#B7D5EC", size=10)

    arrow(ax, (3.45, 3.51), (4.48, 3.25))
    arrow(ax, (3.68, 1.98), (4.48, 2.17))
    arrow(ax, (3.68, 1.31), (4.48, 1.31))

    # Q3
    add_box(ax, 9.05, 3.45, 2.10, 0.52, "分站模板枚举", "#FFD2D7", size=11)
    add_box(ax, 9.05, 2.42, 2.55, 0.52, "生成候选定价方案", "#FFD2D7", size=11)
    add_box(ax, 9.05, 1.30, 2.75, 0.64, "筛选满足财务、补贴\n与容量约束的方案", "#FFD2D7", size=10)
    arrow(ax, (10.10, 3.42), (10.10, 2.98))
    arrow(ax, (10.10, 2.38), (10.10, 1.98))

    add_box(ax, 11.85, 1.55, 0.72, 2.62, "空间\n经济\n服务\n可及性评估", "#F7C8CE", size=10)
    add_box(ax, 13.40, 3.35, 1.40, 0.52, "空间维度", "#FFD2D7", size=10)
    add_box(ax, 13.40, 2.38, 1.40, 0.52, "经济维度", "#FFD2D7", size=10)
    add_box(ax, 13.40, 1.38, 1.40, 0.52, "服务维度", "#FFD2D7", size=10)
    ax.plot([12.60, 13.00], [2.86, 2.86], color="#F39AA7", lw=1.4)
    ax.plot([13.00, 13.30], [2.86, 3.61], color="#F39AA7", lw=1.4)
    ax.plot([13.00, 13.30], [2.86, 2.64], color="#F39AA7", lw=1.4)
    ax.plot([13.00, 13.30], [2.86, 1.64], color="#F39AA7", lw=1.4)

    # Q4
    add_box(ax, 8.95, 7.76, 1.80, 0.54, "场景扰动设置", "#B6D7A8", size=10)
    add_box(ax, 10.95, 7.76, 3.85, 0.54, "基准、人口压力、成本上升、预算宽松", "#A8D08D", color="white", size=10)
    arrow(ax, (10.80, 8.03), (10.92, 8.03))

    add_box(ax, 8.95, 6.70, 2.20, 0.52, "重复求解问题1-问题3", "#B6D7A8", size=10)
    add_box(ax, 11.45, 6.72, 3.35, 0.52, "评估扰动下的稳定性\n与风险暴露", "#C5E0B4", size=10)
    add_box(ax, 8.95, 5.76, 2.95, 0.52, "计算敏感度系数\n与关键指标变化", "#B6D7A8", size=10)
    add_box(ax, 12.00, 5.76, 2.62, 0.52, "鲁棒性检验\n与预算脆弱性分析", "#A8D08D", color="white", size=10)

    arrow(ax, (10.05, 7.72), (10.05, 7.26))
    arrow(ax, (10.05, 6.66), (10.05, 6.32))
    arrow(ax, (12.70, 5.74), (12.70, 6.68))

    # Cross-problem arrows
    arrow(ax, (4.20, 5.23), (4.20, 4.33), color="#B18D20", lw=1.2)
    arrow(ax, (7.70, 2.45), (8.55, 2.45), color="#9ECBE8", lw=1.5)
    arrow(ax, (12.05, 4.34), (12.05, 5.25), color="#F2A0AA", lw=1.5)

    ax.text(
        8.0,
        4.55,
        "形成完整求解链",
        rotation=32,
        fontsize=28,
        color="#D8D8D8",
        ha="center",
        va="center",
        alpha=0.50,
        fontweight="bold",
    )

    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(OUT_PATH)


if __name__ == "__main__":
    main()
