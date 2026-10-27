"""KNN 课堂实践项目入口。

第 1 步完成环境自检；第 2 步加入数据读取、概况统计和简单可视化；
第 3 步加入分层划分和 Z-score 标准化；第 4-5 步加入距离函数和最近邻搜索演示；
第 6 步加入 KNN 分类器演示；第 7 步加入 KNN 回归演示；
第 8-9 步加入模型评估、交叉验证与 k 值选择。后续步骤会继续接入可视化与报告。
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
    KNNClassifier,
    KNNRegressor,
    RANDOM_SEED,
    StandardScaler,
    accuracy_score,
    classification_report,
    confusion_matrix,
    cross_validate_knn,
    euclidean_distance,
    majority_vote,
    manhattan_distance,
    nearest_neighbors,
    stratified_split_indices,
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
REGRESSION_FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
]
REGRESSION_TARGET_COLUMN = "petal width (cm)"
TARGET_COLUMN = "target_name"
CLASS_ORDER = ["setosa", "versicolor", "virginica"]
K_VALUES = list(range(1, 16))
CV_FOLDS = 5


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


def split_classification_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按类别分层返回原始（未标准化）的分类训练/测试数据。"""
    X = df[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[TARGET_COLUMN].to_numpy()
    train_index, test_index = stratified_split_indices(
        y, test_size=test_size, random_state=random_state
    )
    return X[train_index], X[test_index], y[train_index], y[test_index]


def prepare_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """分层划分训练/测试集，并只用训练集拟合标准化器。

    返回：(X_train, X_test, y_train, y_test, scaler)
    """
    X_train_raw, X_test_raw, y_train, y_test = split_classification_data(
        df, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    return X_train, X_test, y_train, y_test, scaler


def prepare_regression_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """为回归准备数据：预测花瓣宽度，其他 3 个特征作为输入。

    使用与分类相同的分层索引，保证两个任务的训练/测试样本一致。
    """
    X = df[REGRESSION_FEATURE_COLUMNS].to_numpy(dtype=float)
    y = df[REGRESSION_TARGET_COLUMN].to_numpy(dtype=float)
    class_labels = df[TARGET_COLUMN].to_numpy()

    train_index, test_index = stratified_split_indices(
        class_labels, test_size=test_size, random_state=random_state
    )

    X_train_raw, X_test_raw = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

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


def format_confusion_matrix(matrix: np.ndarray, labels: list[str]) -> str:
    """把混淆矩阵格式化成可读文本。"""
    lines = ["真实\\预测\t" + "\t".join(labels)]
    for label, row in zip(labels, matrix):
        lines.append(label + "\t" + "\t".join(str(int(value)) for value in row))
    return "\n".join(lines)


def evaluate_classifier(
    model: KNNClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    title: str,
) -> tuple[str, float, np.ndarray]:
    """在测试集上评估分类器，返回文本、准确率和预测结果。"""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    matrix = confusion_matrix(y_test, predictions, labels=CLASS_ORDER)
    report = classification_report(y_test, predictions, labels=CLASS_ORDER)

    lines = [
        title,
        f"测试集样本数: {len(y_test)}",
        f"准确率: {accuracy:.4f}",
        "",
        "混淆矩阵（行=真实类别，列=预测类别）:",
        format_confusion_matrix(matrix, CLASS_ORDER),
        "",
        "分类报告:",
        report,
    ]
    return "\n".join(lines), accuracy, predictions


def format_cv_results(
    cv_results: dict[int, dict[str, float | list[float]]],
    k_values: list[int],
    best_k: int,
) -> str:
    """格式化交叉验证结果表。"""
    lines = [
        f"{CV_FOLDS} 折交叉验证结果（仅使用训练集，标准化在每折内部完成）:",
        f"{'k':>4}{'平均准确率':>12}{'标准差':>10}",
    ]
    for k in k_values:
        result = cv_results[k]
        lines.append(
            f"{k:>4}{result['mean_accuracy']:>12.4f}"
            f"{result['std_accuracy']:>10.4f}"
        )
    lines.append("")
    lines.append(f"最优 k: {best_k}（平均准确率最高；并列时取较小 k）")
    return "\n".join(lines)


def plot_k_accuracy_curve(
    cv_results: dict[int, dict[str, float | list[float]]],
    best_k: int,
    output_path: Path,
) -> None:
    """画 k-平均准确率曲线，误差棒表示 5 折的标准差。"""
    ks = sorted(cv_results)
    means = [float(cv_results[k]["mean_accuracy"]) for k in ks]
    stds = [float(cv_results[k]["std_accuracy"]) for k in ks]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(ks, means, yerr=stds, marker="o", capsize=3, label="mean ± std")
    ax.axvline(best_k, color="red", linestyle="--", label=f"best k={best_k}")
    ax.set_xlabel("k")
    ax.set_ylabel("Mean CV accuracy")
    ax.set_title("5-fold CV accuracy vs k")
    ax.set_xticks(ks)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


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


def demonstrate_classifier(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    k: int = 5,
) -> str:
    """训练 KNNClassifier，并展示前 10 个测试样本的预测和一次投票过程。"""
    classifier = KNNClassifier(k=k, metric="euclidean").fit(X_train, y_train)
    predictions = classifier.predict(X_test[:10])

    query = X_test[0]
    neighbor_indices, neighbor_distances = nearest_neighbors(X_train, query, k=k)
    neighbor_labels = y_train[neighbor_indices[0]]
    vote = majority_vote(neighbor_labels, neighbor_distances[0])
    correct = int(np.sum(predictions == y_test[:10]))

    lines = [
        f"KNNClassifier(k={k}) 前 10 个测试样本:",
        f"  真实类别: {y_test[:10].tolist()}",
        f"  预测类别: {predictions.tolist()}",
        f"  前 10 个中预测正确: {correct}/10",
        "",
        "第 1 个测试样本的投票过程:",
        f"  邻居索引: {neighbor_indices[0].tolist()}",
        f"  邻居距离: {np.round(neighbor_distances[0], 4).tolist()}",
        f"  邻居类别: {neighbor_labels.tolist()}",
        f"  投票结果: {vote}",
    ]
    return "\n".join(lines)


def demonstrate_regression(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    k: int = 5,
) -> tuple[str, pd.DataFrame]:
    """训练两种 KNN 回归器，返回演示文字和预测结果表。"""
    uniform = KNNRegressor(k=k, weights="uniform").fit(X_train, y_train)
    weighted = KNNRegressor(k=k, weights="distance").fit(X_train, y_train)

    pred_uniform = uniform.predict(X_test)
    pred_weighted = weighted.predict(X_test)

    mae_uniform = float(np.mean(np.abs(pred_uniform - y_test)))
    mae_weighted = float(np.mean(np.abs(pred_weighted - y_test)))
    rmse_uniform = float(np.sqrt(np.mean((pred_uniform - y_test) ** 2)))
    rmse_weighted = float(np.sqrt(np.mean((pred_weighted - y_test) ** 2)))

    lines = [
        "KNN 回归演示（输入：花萼长度/宽度、花瓣长度；目标：花瓣宽度，单位 cm）:",
        f"  测试集样本数: {len(y_test)}",
        f"  前 10 个真实值: {np.round(y_test[:10], 4).tolist()}",
        f"  前 10 个均值预测: {np.round(pred_uniform[:10], 4).tolist()}",
        f"  前 10 个距离加权预测: {np.round(pred_weighted[:10], 4).tolist()}",
        f"  均值预测 MAE: {mae_uniform:.4f} cm, RMSE: {rmse_uniform:.4f} cm",
        f"  距离加权 MAE: {mae_weighted:.4f} cm, RMSE: {rmse_weighted:.4f} cm",
    ]

    results = pd.DataFrame(
        {
            "true_petal_width": y_test,
            "pred_uniform": pred_uniform,
            "pred_distance_weighted": pred_weighted,
        }
    )
    return "\n".join(lines), results


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

    classifier_demo = demonstrate_classifier(
        X_train, X_test, y_train, y_test, k=5
    )
    print("\nKNN 分类器演示:")
    print(classifier_demo)

    classifier_demo_path = METRICS_DIR / "classifier_demo.txt"
    classifier_demo_path.write_text(classifier_demo + "\n", encoding="utf-8")
    print(f"\n分类器演示记录已保存: {classifier_demo_path}")

    Xr_train, Xr_test, yr_train, yr_test, _ = prepare_regression_data(clean_df)
    regression_demo, regression_results = demonstrate_regression(
        Xr_train, Xr_test, yr_train, yr_test, k=5
    )
    print("\nKNN 回归演示:")
    print(regression_demo)

    regression_demo_path = METRICS_DIR / "regression_demo.txt"
    regression_demo_path.write_text(regression_demo + "\n", encoding="utf-8")
    regression_csv_path = METRICS_DIR / "regression_predictions.csv"
    regression_results.to_csv(regression_csv_path, index=False)
    print(f"\n回归演示记录已保存: {regression_demo_path}")
    print(f"回归预测表已保存: {regression_csv_path}")

    # 第 8 步：用 k=5 做测试集基线评估。
    k5_model = KNNClassifier(k=5, metric="euclidean").fit(X_train, y_train)
    k5_text, k5_accuracy, k5_predictions = evaluate_classifier(
        k5_model, X_test, y_test, "k=5 基线评估"
    )
    print("\n第 8 步 模型评估（k=5）:")
    print(k5_text)

    evaluation_k5_path = METRICS_DIR / "evaluation_k5.txt"
    evaluation_k5_path.write_text(k5_text + "\n", encoding="utf-8")
    print(f"\nk=5 评估记录已保存: {evaluation_k5_path}")

    # 第 9 步：只在训练集上做分层 5 折交叉验证，选 k。
    X_train_raw, _, y_train_raw, _ = split_classification_data(clean_df)
    cv_results = cross_validate_knn(
        X_train_raw,
        y_train_raw,
        k_values=K_VALUES,
        n_splits=CV_FOLDS,
        random_state=RANDOM_SEED,
    )
    best_k = max(K_VALUES, key=lambda k: (cv_results[k]["mean_accuracy"], -k))

    cv_text = format_cv_results(cv_results, K_VALUES, best_k)
    print("\n第 9 步 5 折交叉验证:")
    print(cv_text)

    k_selection_path = METRICS_DIR / "k_selection.txt"
    k_selection_path.write_text(cv_text + "\n", encoding="utf-8")
    k_curve_path = FIGURES_DIR / "k_accuracy_curve.png"
    plot_k_accuracy_curve(cv_results, best_k, k_curve_path)
    print(f"\nk 值选择记录已保存: {k_selection_path}")
    print(f"k-准确率曲线已保存: {k_curve_path}")

    # 用交叉验证选出的 k 在测试集上做最终评估。
    final_model = KNNClassifier(k=best_k, metric="euclidean").fit(X_train, y_train)
    final_text, final_accuracy, final_predictions = evaluate_classifier(
        final_model, X_test, y_test, f"最优 k={best_k} 测试集评估"
    )
    print("\n最终测试集评估:")
    print(final_text)

    evaluation_best_path = METRICS_DIR / "evaluation_best_k.txt"
    evaluation_best_path.write_text(final_text + "\n", encoding="utf-8")

    evaluation_predictions = pd.DataFrame(
        {
            "true_label": y_test,
            "pred_k5": k5_predictions,
            "pred_best_k": final_predictions,
        }
    )
    evaluation_predictions_path = METRICS_DIR / "evaluation_predictions.csv"
    evaluation_predictions.to_csv(evaluation_predictions_path, index=False)
    print(f"\n最优 k 评估记录已保存: {evaluation_best_path}")
    print(f"测试集预测明细已保存: {evaluation_predictions_path}")


if __name__ == "__main__":
    main()
