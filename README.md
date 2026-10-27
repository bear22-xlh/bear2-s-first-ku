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
│       ├── preprocessing_summary.txt
│       ├── nearest_neighbors_demo.txt
│       └── classifier_demo.txt
├── knn.py              # KNN 核心实现
├── main.py             # 项目入口与实验流程
├── test_knn.py         # 单元测试
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

## 6. 距离函数与最近邻搜索

- 欧氏距离：`sqrt(sum((x1 - x2) ** 2))`
- 曼哈顿距离：`sum(abs(x1 - x2))`
- `pairwise_distances()` 用 NumPy 广播计算任意两组样本的距离矩阵，不写双重循环
- `nearest_neighbors()` 对距离做稳定排序，取前 k 个邻居，返回索引和距离
- 参数检查：k 必须是正整数且不超过训练样本数；训练集和查询样本特征数必须一致
- 验证：距离相关测试通过；真实数据上的欧氏/曼哈顿距离矩阵与 sklearn 一致
- 演示记录：`outputs/metrics/nearest_neighbors_demo.txt`

测试命令：

```bash
python -m unittest -v test_knn.py
```

## 7. KNN 分类器

- `majority_vote()`：统计 k 个邻居的类别票数，票数最多者获胜
- 平票规则：票数相同时，选距离最近的那个邻居所属的类别
- `KNNClassifier`：`fit()` 保存训练集，`predict()` 对每个样本找邻居并投票
- 参数检查：k 为正整数且不超过训练样本数；训练和预测特征数必须一致；未 `fit` 不允许 `predict`
- 验证：当前 24 个单元测试全部通过；k=5 时前 10 个测试样本全部预测正确
- 演示记录：`outputs/metrics/classifier_demo.txt`

## 8. 复现约定

- 全项目统一随机种子：`42`（定义在 `knn.py` 的 `RANDOM_SEED`）
- 数据划分、交叉验证等所有随机操作都使用该种子
- 标准化参数只由训练集计算，测试集只做变换，避免信息泄漏

## 9. 运行方式

```bash
python main.py
```

当前进度：第 6 步（KNN 分类器）已完成。
