# KNN 课堂实践（Iris 数据集）

## 1. 项目目标

在不调用 `sklearn.neighbors` 核心功能的前提下，用 Python 从零实现 KNN 分类与回归，
完成数据读取、预处理、距离计算、最近邻搜索、模型评估、k 值选择、可视化、
sklearn 对照、异常处理、性能测试和单元测试。

## 2. 运行环境

- Python 3.11.17（Anaconda 环境 `ai_ml`）
- numpy 2.4.6
- pandas 3.0.6
- matplotlib 3.11.2
- scikit-learn 1.9.1

运行前可激活环境：

```bash
conda activate ai_ml
```

如果终端里找不到 `conda`，直接使用解释器的完整路径：

```powershell
& "E:\annaconda1\envs\ai_ml\python.exe" main.py
```

## 3. 目录结构

```text
knn_practice/
├── data/
│   └── iris_dataset.csv
├── outputs/
│   ├── figures/        # 图表
│   └── metrics/        # 指标与实验记录
├── knn.py              # KNN 核心实现
├── main.py             # 项目入口与实验流程
├── requirements.txt    # 依赖版本
└── README.md
```

## 4. 数据说明

- 文件：`data/iris_dataset.csv`
- 规模：150 行 × 5 列（4 个数值特征 + 1 个类别列 `target_name`）
- 类别：setosa、versicolor、virginica，各 50 条
- 缺失值：无
- 重复样本：1 条，第 2 步会处理并记录

## 5. 复现约定

- 全项目统一随机种子：`42`（定义在 `knn.py` 的 `RANDOM_SEED`）
- 数据划分、交叉验证等所有随机操作都使用该种子
- 标准化参数只由训练集计算，测试集只做变换，避免信息泄漏

## 6. 运行方式

```bash
python main.py
```

当前进度：第 1 步（工程环境与代码规范）已完成。
