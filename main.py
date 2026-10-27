"""KNN 课堂实践项目入口。

第 1 步只做环境自检：打印 Python 版本、依赖版本和统一随机种子，
确认项目骨架可用。后续步骤会在这里逐步接入数据读取、实验和图表生成。
"""

from __future__ import annotations

import platform
import sys

import matplotlib
import numpy as np
import pandas as pd
import sklearn

from knn import RANDOM_SEED


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


if __name__ == "__main__":
    show_environment()
