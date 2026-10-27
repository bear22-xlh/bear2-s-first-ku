"""KNN 手写实现模块。

本文件用于从零实现 KNN 的分类与回归，不调用 sklearn.neighbors。
目前处于第 1 步，只定义全局随机种子；后续步骤会逐步补充距离函数、
最近邻搜索、分类器、回归器和评估函数。
"""

from __future__ import annotations

# 全项目统一随机种子：所有涉及随机性的操作都必须使用它，保证结果可复现。
RANDOM_SEED: int = 42

__all__ = ["RANDOM_SEED"]
