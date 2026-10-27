"""KNN 手写实现模块。

本文件用于从零实现 KNN 的分类与回归，不调用 sklearn.neighbors。
第 1 步定义全局随机种子；第 3 步加入分层划分和 Z-score 标准化。
后续步骤会继续补充距离函数、最近邻搜索、分类器、回归器和评估函数。
"""

from __future__ import annotations

import numpy as np

# 全项目统一随机种子：所有涉及随机性的操作都必须使用它，保证结果可复现。
RANDOM_SEED: int = 42


class StandardScaler:
    """Z-score 标准化器：z = (x - mean) / std。

    只允许用训练集调用 fit() 计算均值和标准差；测试集只能调用 transform()，
    这样测试集的信息不会参与参数计算，避免信息泄漏。
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """用训练集计算每个特征的均值和标准差。"""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X 必须是二维数组，形状为 (样本数, 特征数)")

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        # 标准差为 0 的特征无法缩放，把除数改成 1，避免除以 0。
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """用训练集得到的均值和标准差变换数据。"""
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("请先调用 fit() 计算标准化参数")

        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X 必须是二维数组，形状为 (样本数, 特征数)")
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"特征数不一致：训练集有 {self.mean_.shape[0]} 个特征，"
                f"当前数据有 {X.shape[1]} 个特征"
            )
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """在训练集上计算参数并立即完成标准化。"""
        return self.fit(X).transform(X)


def train_test_split_stratified(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按类别分层划分训练集和测试集。

    每个类别分别按 test_size 比例抽取测试样本，保证训练集和测试集的
    类别分布接近原始数据。随机数生成器使用固定种子，划分结果可复现。

    返回：(X_train, X_test, y_train, y_test)
    """
    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2:
        raise ValueError("X 必须是二维数组，形状为 (样本数, 特征数)")
    if y.ndim != 1:
        raise ValueError("y 必须是一维数组")
    if len(X) != len(y):
        raise ValueError(f"X 和 y 的样本数不一致：{len(X)} != {len(y)}")
    if not 0 < test_size < 1:
        raise ValueError("test_size 必须在 0 和 1 之间")

    rng = np.random.default_rng(random_state)
    train_indices: list[np.ndarray] = []
    test_indices: list[np.ndarray] = []

    for class_name in np.unique(y):
        class_indices = np.flatnonzero(y == class_name)
        if len(class_indices) < 2:
            raise ValueError(
                f"类别 {class_name!r} 只有 {len(class_indices)} 个样本，无法分层划分"
            )

        rng.shuffle(class_indices)
        n_test = max(1, int(round(len(class_indices) * test_size)))
        n_test = min(n_test, len(class_indices) - 1)
        test_indices.append(class_indices[:n_test])
        train_indices.append(class_indices[n_test:])

    train_index = np.concatenate(train_indices)
    test_index = np.concatenate(test_indices)

    # 再打乱一次，避免训练集和测试集内部按类别聚集。
    rng.shuffle(train_index)
    rng.shuffle(test_index)

    return X[train_index], X[test_index], y[train_index], y[test_index]


__all__ = ["RANDOM_SEED", "StandardScaler", "train_test_split_stratified"]
