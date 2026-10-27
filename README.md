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
│   ├── figures/
│   │   ├── feature_histograms.png
│   │   └── petal_scatter.png
│   └── metrics/
│       ├── data_overview.txt
│       └── preprocessing_summary.txt
├── knn.py              # KNN 核心实现
├── main.py             # 项目入口与实验流程
├── requirements.txt    # 依赖版本
└── README.md
```

## 4. 数据说明

- 文件：`data/iris_dataset.csv`
- 原始规模：150 行 × 5 列（4 个数值特征 + 1 个类别列 `target_name`）
- 去重后规模：149 行 × 5 列
- 类别分布：原始三类各 50 条；去重后 setosa 50、versicolor 50、virginica 49
- 缺失值：无
- 重复样本：1 条（virginica，特征 5.8/2.7/5.1/1.9），保留一条

数据探索结论：

- 四个特征量纲差异明显：花瓣长度标准差约 1.77，花萼宽度只有约 0.44，后续需要标准化。
- 花瓣长度和花瓣宽度对类别区分度最高，setosa 可以仅凭花瓣特征与另外两类分开。
- versicolor 和 virginica 在花瓣特征上有少量重叠，是后续分类的主要误差来源。

## 5. 数据预处理

- 分层划分：80% 训练、20% 测试，随机种子 42
- 训练集 119 条：setosa 40、versicolor 40、virginica 39
- 测试集 30 条：setosa 10、versicolor 10、virginica 10
- 标准化：手写 `StandardScaler`，只用训练集计算每个特征的均值和标准差
- 测试集只调用 `transform()`，不参与参数计算，避免信息泄漏
- 验证：两次运行的划分完全一致；标准化结果与 sklearn `StandardScaler` 一致
- 预处理记录：`outputs/metrics/preprocessing_summary.txt`

## 6. 复现约定

- 全项目统一随机种子：`42`（定义在 `knn.py` 的 `RANDOM_SEED`）
- 数据划分、交叉验证等所有随机操作都使用该种子
- 标准化参数只由训练集计算，测试集只做变换，避免信息泄漏

## 7. 运行方式

```bash
python main.py
```

当前进度：第 3 步（数据预处理）已完成。
