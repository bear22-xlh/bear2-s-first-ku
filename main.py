"""KNN 课堂实践项目入口。

第 1 步完成环境自检；第 2 步加入数据读取、数据概况统计和简单可视化。
后续步骤会继续接入预处理、KNN 实现、评估与报告。
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 使用无界面后端，保证在终端里也能保存图片
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from knn import RANDOM_SEED

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "iris_dataset.csv"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
METRICS_DIR = PROJECT_ROOT / "outputs" / "metrics"

FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
TARGET_COLUMN = "target_name"
CLASS_ORDER = ["setosa", "versicolor", "virginica"]


def show_environment() -> None:
    """打印运行环境和关键依赖版本，便于在报告中记录实验条件。"""
    print("=" * 50)
    print("KNN 课堂实践 - 环境自检")
    print("=" * 50)
    print(f"Python      : {sys.version.split()[0]}")
    print(f"操作系统    : {platform.system()} {platform.release()}")
    print(f"numpy       : {np.__version__}")
    print(f"pandas      : {pd.__version__}")
    print(f"matplotlib  : {matplotlib.__version__}")
    print(f"scikit-learn: {sklearn.__version__}")
    print(f"随机种子    : {RANDOM_SEED}")
    print("=" * 50)


def load_data(csv_path: Path = DATA_PATH) -> pd.DataFrame:
    """读取 Iris CSV，并检查必要列是否存在。"""
    df = pd.read_csv(csv_path)
    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"数据缺少必要列: {missing_columns}")
    return df


def summarize_data(df: pd.DataFrame) -> str:
    """生成数据概况文本：规模、缺失值、类别分布、重复值和特征统计。"""
    lines = [
        f"样本数: {df.shape[0]}",
        f"特征数: {len(FEATURE_COLUMNS)}",
        f"列名: {list(df.columns)}",
        f"缺失值总数: {int(df.isna().sum().sum())}",
        "缺失值明细:",
    ]
    for column in df.columns:
        lines.append(f"  {column}: {int(df[column].isna().sum())}")

    lines.append("类别分布:")
    class_counts = df[TARGET_COLUMN].value_counts()
    for class_name in CLASS_ORDER:
        count = int(class_counts.get(class_name, 0))
        ratio = count / len(df) * 100
        lines.append(f"  {class_name}: {count} ({ratio:.1f}%)")

    lines.append(f"重复样本数: {int(df.duplicated().sum())}")
    lines.append("数值特征统计:")

    feature_stats = df[FEATURE_COLUMNS].describe().T
    for feature, row in feature_stats.iterrows():
        lines.append(
            f"  {feature}: "
            f"min={row['min']:.2f}, mean={row['mean']:.2f}, "
            f"max={row['max']:.2f}, std={row['std']:.2f}"
        )
    return "\n".join(lines)


def plot_feature_histograms(df: pd.DataFrame, output_path: Path) -> None:
    """为 4 个特征分别画按类别着色的直方图。"""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, feature in zip(axes.ravel(), FEATURE_COLUMNS):
        for class_name in CLASS_ORDER:
            values = df.loc[df[TARGET_COLUMN] == class_name, feature]
            ax.hist(values, bins=12, alpha=0.5, label=class_name)
        ax.set_title(feature)
        ax.set_xlabel("value")
        ax.set_ylabel("count")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")
    fig.suptitle("Feature distributions by class")
    fig.tight_layout(rect=(0, 0, 0.92, 0.96))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_petal_scatter(df: pd.DataFrame, output_path: Path) -> None:
    """画花瓣长度-宽度散点图，观察三类样本是否容易分开。"""
    fig, ax = plt.subplots(figsize=(7, 6))
    for class_name in CLASS_ORDER:
        subset = df[df[TARGET_COLUMN] == class_name]
        ax.scatter(
            subset["petal length (cm)"],
            subset["petal width (cm)"],
            label=class_name,
            alpha=0.8,
        )

    ax.set_xlabel("Petal length (cm)")
    ax.set_ylabel("Petal width (cm)")
    ax.set_title("Petal length vs petal width")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main() -> None:
    show_environment()

    raw_df = load_data()
    print("\n原始数据前 5 行:")
    print(raw_df.head().to_string(index=False))

    raw_summary = summarize_data(raw_df)
    print("\n原始数据概况:")
    print(raw_summary)

    # Iris 中有 1 条完全重复的记录，去重避免同一朵花同时进入训练集和测试集。
    clean_df = raw_df.drop_duplicates().reset_index(drop=True)
    clean_summary = summarize_data(clean_df)
    print("\n去重后数据概况:")
    print(clean_summary)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    overview_path = METRICS_DIR / "data_overview.txt"
    overview_path.write_text(
        "原始数据概况\n"
        + "=" * 40
        + "\n"
        + raw_summary
        + "\n\n去重后数据概况\n"
        + "=" * 40
        + "\n"
        + clean_summary
        + "\n",
        encoding="utf-8",
    )

    histogram_path = FIGURES_DIR / "feature_histograms.png"
    scatter_path = FIGURES_DIR / "petal_scatter.png"
    plot_feature_histograms(clean_df, histogram_path)
    plot_petal_scatter(clean_df, scatter_path)

    print(f"\n数据概况已保存: {overview_path}")
    print(f"特征分布图已保存: {histogram_path}")
    print(f"花瓣散点图已保存: {scatter_path}")


if __name__ == "__main__":
    main()
