"""KNN 手写实现模块。

本文件用于从零实现 KNN 的分类与回归，不调用 sklearn.neighbors。
第 1 步定义全局随机种子；第 3 步加入分层划分和 Z-score 标准化；
第 4-5 步加入欧氏/曼哈顿距离和最近邻搜索；第 6 步加入多数投票分类器；
第 7 步加入 KNN 回归器。后续步骤会继续补充评估函数。
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


def stratified_split_indices(
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """按类别分层，返回训练集和测试集的索引。

    分类和回归可以共用同一组索引，保证两者的训练/测试样本完全一致。
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError("y 必须是一维数组")
    if len(y) == 0:
        raise ValueError("y 不能为空")
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
    return train_index, test_index


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

    train_index, test_index = stratified_split_indices(y, test_size, random_state)
    return X[train_index], X[test_index], y[train_index], y[test_index]


def euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """计算两个一维样本之间的欧氏距离。

    公式：sqrt(sum((x1 - x2) ** 2))
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError("单样本距离要求 x1 和 x2 都是一维数组")
    if x1.shape != x2.shape:
        raise ValueError(f"两个样本的特征数不一致：{x1.shape} != {x2.shape}")

    return float(np.sqrt(np.sum((x1 - x2) ** 2)))


def manhattan_distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """计算两个一维样本之间的曼哈顿距离。

    公式：sum(abs(x1 - x2))
    """
    x1 = np.asarray(x1, dtype=float)
    x2 = np.asarray(x2, dtype=float)

    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError("单样本距离要求 x1 和 x2 都是一维数组")
    if x1.shape != x2.shape:
        raise ValueError(f"两个样本的特征数不一致：{x1.shape} != {x2.shape}")

    return float(np.sum(np.abs(x1 - x2)))


def pairwise_distances(
    X: np.ndarray,
    Y: np.ndarray,
    metric: str = "euclidean",
) -> np.ndarray:
    """计算 X 中每个样本到 Y 中每个样本的距离。

    参数:
        X: 形状 (n_samples_x, n_features)
        Y: 形状 (n_samples_y, n_features)
        metric: "euclidean" 或 "manhattan"

    返回:
        形状 (n_samples_x, n_samples_y) 的距离矩阵。
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError("X 和 Y 必须是二维数组")
    if X.shape[1] != Y.shape[1]:
        raise ValueError(
            f"特征数不一致：X 有 {X.shape[1]} 个特征，Y 有 {Y.shape[1]} 个特征"
        )

    # 广播：X[:, None, :] 形状 (n_x, 1, n_features)，Y[None, :, :] 形状 (1, n_y, n_features)
    diff = X[:, None, :] - Y[None, :, :]

    if metric == "euclidean":
        return np.sqrt(np.sum(diff ** 2, axis=2))
    if metric == "manhattan":
        return np.sum(np.abs(diff), axis=2)

    raise ValueError(
        f"不支持的距离度量: {metric!r}，可选 'euclidean' 或 'manhattan'"
    )


def nearest_neighbors(
    X_train: np.ndarray,
    X_query: np.ndarray,
    k: int,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray]:
    """为每个查询样本找 k 个最近邻。

    X_query 可以是一维（单个样本）或二维（多个样本）。
    返回:
        neighbor_indices: 形状 (n_queries, k)，每行是邻居在训练集中的索引
        neighbor_distances: 形状 (n_queries, k)，每行是对应距离，已按距离升序
    """
    X_train = np.asarray(X_train, dtype=float)
    X_query = np.asarray(X_query, dtype=float)

    if X_train.ndim != 2:
        raise ValueError("X_train 必须是二维数组")
    if X_query.ndim == 1:
        X_query = X_query.reshape(1, -1)
    if X_query.ndim != 2:
        raise ValueError("X_query 必须是一维或二维数组")
    if X_train.shape[1] != X_query.shape[1]:
        raise ValueError(
            f"特征数不一致：训练集有 {X_train.shape[1]} 个特征，"
            f"查询样本有 {X_query.shape[1]} 个特征"
        )
    if not isinstance(k, (int, np.integer)):
        raise ValueError("k 必须是整数")
    if k <= 0:
        raise ValueError("k 必须大于 0")
    if k > X_train.shape[0]:
        raise ValueError(
            f"k 不能大于训练样本数：k={k}, 训练样本数={X_train.shape[0]}"
        )

    distances = pairwise_distances(X_query, X_train, metric=metric)
    # stable 排序保证距离相同时按训练集原索引顺序返回，结果可复现。
    neighbor_indices = np.argsort(distances, axis=1, kind="stable")[:, :k]
    neighbor_distances = np.take_along_axis(distances, neighbor_indices, axis=1)
    return neighbor_indices, neighbor_distances


def majority_vote(labels: np.ndarray, distances: np.ndarray | None = None) -> object:
    """对邻居标签做多数投票，返回票数最多的类别。

    平票处理：如果多个类别票数相同，选择距离最近的那个邻居所属的类别。
    labels 和 distances 应当已经按距离升序排列（nearest_neighbors 的返回值满足）。
    """
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("labels 必须是非空一维数组")

    if distances is not None:
        distances = np.asarray(distances, dtype=float)
        if distances.shape != labels.shape:
            raise ValueError(
                f"distances 必须与 labels 形状一致：{distances.shape} != {labels.shape}"
            )

    counts: dict[object, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1

    max_count = max(counts.values())
    tied_labels = [label for label, count in counts.items() if count == max_count]
    if len(tied_labels) == 1:
        return tied_labels[0]

    if distances is None:
        raise ValueError("出现平票时必须提供 distances，按最近邻距离打破平票")

    best_label = tied_labels[0]
    best_distance = float("inf")
    for label, distance in zip(labels, distances):
        if label in tied_labels and distance < best_distance:
            best_label = label
            best_distance = distance
    return best_label


class KNNClassifier:
    """K 近邻分类器：多数投票 + 最近邻距离打破平票。"""

    def __init__(self, k: int = 5, metric: str = "euclidean") -> None:
        if not isinstance(k, (int, np.integer)) or k <= 0:
            raise ValueError("k 必须是正整数")
        if metric not in {"euclidean", "manhattan"}:
            raise ValueError("metric 必须是 'euclidean' 或 'manhattan'")

        self.k = int(k)
        self.metric = metric
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        """保存训练集；KNN 没有需要迭代学习的参数。"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X 必须是二维数组，形状为 (样本数, 特征数)")
        if y.ndim != 1:
            raise ValueError("y 必须是一维数组")
        if len(X) != len(y):
            raise ValueError(f"X 和 y 的样本数不一致：{len(X)} != {len(y)}")
        if len(X) == 0:
            raise ValueError("训练集不能为空")
        if X.shape[1] == 0:
            raise ValueError("训练集特征数不能为 0")
        if self.k > len(X):
            raise ValueError(
                f"k 不能大于训练样本数：k={self.k}, 训练样本数={len(X)}"
            )

        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """对每个查询样本找 k 个邻居，投票得到预测类别。"""
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("请先调用 fit() 训练模型")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X 必须是一维或二维数组")
        if X.shape[0] == 0:
            raise ValueError("预测数据不能为空")
        if X.shape[1] != self.X_train_.shape[1]:
            raise ValueError(
                f"特征数不一致：训练集有 {self.X_train_.shape[1]} 个特征，"
                f"预测数据有 {X.shape[1]} 个特征"
            )

        neighbor_indices, neighbor_distances = nearest_neighbors(
            self.X_train_, X, k=self.k, metric=self.metric
        )

        predictions = []
        for indices, distances in zip(neighbor_indices, neighbor_distances):
            neighbor_labels = self.y_train_[indices]
            predictions.append(majority_vote(neighbor_labels, distances))
        return np.asarray(predictions)


class KNNRegressor:
    """K 近邻回归器：邻居均值或距离加权平均。"""

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        weights: str = "uniform",
    ) -> None:
        if not isinstance(k, (int, np.integer)) or k <= 0:
            raise ValueError("k 必须是正整数")
        if metric not in {"euclidean", "manhattan"}:
            raise ValueError("metric 必须是 'euclidean' 或 'manhattan'")
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights 必须是 'uniform' 或 'distance'")

        self.k = int(k)
        self.metric = metric
        self.weights = weights
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNRegressor":
        """保存训练集；y 必须是连续数值。"""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X 必须是二维数组，形状为 (样本数, 特征数)")
        if y.ndim != 1:
            raise ValueError("y 必须是一维数组")
        if len(X) != len(y):
            raise ValueError(f"X 和 y 的样本数不一致：{len(X)} != {len(y)}")
        if len(X) == 0:
            raise ValueError("训练集不能为空")
        if X.shape[1] == 0:
            raise ValueError("训练集特征数不能为 0")
        if self.k > len(X):
            raise ValueError(
                f"k 不能大于训练样本数：k={self.k}, 训练样本数={len(X)}"
            )

        self.X_train_ = X
        self.y_train_ = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """对每个查询样本找 k 个邻居，再聚合邻居的目标值。"""
        if self.X_train_ is None or self.y_train_ is None:
            raise RuntimeError("请先调用 fit() 训练模型")

        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if X.ndim != 2:
            raise ValueError("X 必须是一维或二维数组")
        if X.shape[0] == 0:
            raise ValueError("预测数据不能为空")
        if X.shape[1] != self.X_train_.shape[1]:
            raise ValueError(
                f"特征数不一致：训练集有 {self.X_train_.shape[1]} 个特征，"
                f"预测数据有 {X.shape[1]} 个特征"
            )

        neighbor_indices, neighbor_distances = nearest_neighbors(
            self.X_train_, X, k=self.k, metric=self.metric
        )

        predictions = []
        for indices, distances in zip(neighbor_indices, neighbor_distances):
            predictions.append(
                self._aggregate(self.y_train_[indices], distances)
            )
        return np.asarray(predictions, dtype=float)

    def _aggregate(self, neighbor_values: np.ndarray, distances: np.ndarray) -> float:
        """把 k 个邻居的目标值聚合成一个预测值。"""
        if self.weights == "uniform":
            return float(np.mean(neighbor_values))

        # 距离越小权重越大；加 1e-12 防止距离为 0 时除零。
        weights = 1.0 / (distances + 1e-12)
        return float(np.sum(weights * neighbor_values) / np.sum(weights))


__all__ = [
    "RANDOM_SEED",
    "StandardScaler",
    "stratified_split_indices",
    "train_test_split_stratified",
    "euclidean_distance",
    "manhattan_distance",
    "pairwise_distances",
    "nearest_neighbors",
    "majority_vote",
    "KNNClassifier",
    "KNNRegressor",
]
