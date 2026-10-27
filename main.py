"""KNN 课堂实践项目入口。

第 1 步完成环境自检；第 2 步加入数据读取、概况统计和简单可视化；
第 3 步加入分层划分和 Z-score 标准化；第 4-5 步加入距离函数和最近邻搜索演示。
后续步骤会继续接入 KNN 分类、回归、评估与报告。
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

from knn import (
    RANDOM_SEED,
    StandardScaler,
    euclidean_distance,
    manhattan_distance,
    nearest_neighbors,
    train_test_split_stratified,
)

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


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """分层划分训练/测试集，并只用训练集拟合标准化器。

    返回：(X_train, X_test, y_train, y_test, scaler)
    """
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy()

    X_train_raw, X_test_raw, y_train, y_test = train_test_split_stratified(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    return X_train, X_test, y_train, y_test, scaler


def summarize_preprocessing(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
) -> str:
    """生成预处理记录：划分规模、类别分布、标准化参数和变换后统计。"""
    if scaler.mean_ is None or scaler.std_ is None:
        raise RuntimeError("标准化器尚未拟合，无法生成预处理记录")

    lines = [
        f"训练集样本数: {len(y_train)}",
        f"测试集样本数: {len(y_test)}",
        "训练集类别分布:",
    ]
    for class_name in CLASS_ORDER:
        lines.append(f"  {class_name}: {int(np.sum(y_train == class_name))}")

    lines.append("测试集类别分布:")
    for class_name in CLASS_ORDER:
        lines.append(f"  {class_name}: {int(np.sum(y_test == class_name))}")

    lines.append("标准化参数（仅由训练集计算）:")
    for feature, mean, std in zip(FEATURE_COLUMNS, scaler.mean_, scaler.std_):
        lines.append(f"  {feature}: mean={mean:.4f}, std={std:.4f}")

    lines.append("标准化后训练集统计（应接近 mean=0, std=1）:")
    for index, feature in enumerate(FEATURE_COLUMNS):
        lines.append(
            f"  {feature}: mean={X_train[:, index].mean():.4f}, "
            f"std={X_train[:, index].std():.4f}"
        )

    lines.append("标准化后测试集统计（不应恰好为 mean=0, std=1）:")
    for index, feature in enumerate(FEATURE_COLUMNS):
        lines.append(
            f"  {feature}: mean={X_test[:, index].mean():.4f}, "
            f"std={X_test[:, index].std():.4f}"
        )
    return "\n".join(lines)


def demonstrate_distance_and_neighbors(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    k: int = 5,
) -> str:
    """校验距离函数，并展示一个测试样本的 k 个最近邻。"""
    p1 = np.array([0.0, 0.0])
    p2 = np.array([3.0, 4.0])

    query = X_test[0]
    neighbor_indices, neighbor_distances = nearest_neighbors(X_train, query, k=k)
    neighbor_labels = y_train[neighbor_indices[0]]

    lines = [
        "距离函数校验:",
        f"  欧氏距离([0,0], [3,4]) = {euclidean_distance(p1, p2):.4f}（手算 5.0000）",
        f"  曼哈顿距离([0,0], [3,4]) = {manhattan_distance(p1, p2):.4f}（手算 7.0000）",
        "",
        f"最近邻搜索演示（标准化后测试集第 1 个样本，k={k}）:",
        f"  真实类别: {y_test[0]}",
        f"  邻居索引: {neighbor_indices[0].tolist()}",
        f"  邻居距离: {np.round(neighbor_distances[0], 4).tolist()}",
        f"  邻居类别: {neighbor_labels.tolist()}",
    ]
    return "\n".join(lines)


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

    X_train, X_test, y_train, y_test, scaler = prepare_data(clean_df)
    preprocessing_summary = summarize_preprocessing(
        X_train, X_test, y_train, y_test, scaler
    )
    print("\n预处理结果:")
    print(preprocessing_summary)

    preprocessing_path = METRICS_DIR / "preprocessing_summary.txt"
    preprocessing_path.write_text(preprocessing_summary + "\n", encoding="utf-8")
    print(f"\n预处理记录已保存: {preprocessing_path}")

    neighbor_demo = demonstrate_distance_and_neighbors(
        X_train, X_test, y_train, y_test, k=5
    )
    print("\n距离与最近邻演示:")
    print(neighbor_demo)

    neighbor_demo_path = METRICS_DIR / "nearest_neighbors_demo.txt"
    neighbor_demo_path.write_text(neighbor_demo + "\n", encoding="utf-8")
    print(f"\n最近邻演示记录已保存: {neighbor_demo_path}")


if __name__ == "__main__":
    main()
