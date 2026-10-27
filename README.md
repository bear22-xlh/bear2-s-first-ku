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
│   │   ├── petal_scatter.png
│   │   ├── k_accuracy_curve.png
│   │   ├── petal_scatter_with_errors.png
│   │   ├── petal_decision_boundary.png
│   │   └── confusion_matrix.png
│   └── metrics/
│       ├── data_overview.txt
│       ├── preprocessing_summary.txt
│       ├── nearest_neighbors_demo.txt
│       ├── classifier_demo.txt
│       ├── regression_demo.txt
│       ├── regression_predictions.csv
│       ├── evaluation_k5.txt
│       ├── k_selection.txt
│       ├── evaluation_best_k.txt
│       ├── evaluation_predictions.csv
│       ├── visualization_summary.txt
│       ├── sklearn_comparison.txt
│       ├── exception_checks.txt
│       ├── performance_comparison.txt
│       ├── parameter_comparison.txt
│       └── test_results.txt
├── knn.py              # KNN 核心实现
├── main.py             # 项目入口与实验流程
├── test_knn.py         # 单元测试
├── requirements.txt    # 依赖版本
├── README.md
└── report.md           # 完整实验报告
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
- 验证：分类相关测试通过；k=5 时前 10 个测试样本全部预测正确
- 演示记录：`outputs/metrics/classifier_demo.txt`

## 8. KNN 回归

- 回归任务：用花萼长度、花萼宽度、花瓣长度预测花瓣宽度（单位 cm）
- 与分类共用同一组分层索引，训练集 119、测试集 30
- 输入特征只用训练集做 Z-score 标准化，目标值保持原始厘米单位
- `KNNRegressor` 支持两种聚合方式：
  - `weights="uniform"`：k 个邻居目标值的平均
  - `weights="distance"`：按 `1 / (距离 + 1e-12)` 加权，越近权重越大
- k=5 测试集结果：均值预测 MAE 0.1620 cm、RMSE 0.2165 cm；距离加权 MAE 0.1622 cm、RMSE 0.2138 cm
- 两种方法误差接近，距离加权的 RMSE 略低，说明远处邻居的干扰有所减少
- 并列邻居处理：距离完全相同时按训练集索引升序取邻居；与 sklearn 对比时 30 个测试样本中 29 个预测一致，唯一差异来自第 5 个邻居距离完全相同但目标值不同（0.34 对 0.30），属于合理的并列选择差异
- 验证：回归相关测试通过；当前项目共 32 个测试全部通过
- 演示记录：`outputs/metrics/regression_demo.txt`
- 预测明细：`outputs/metrics/regression_predictions.csv`

## 9. 模型评估

- 手写 `accuracy_score()`、`confusion_matrix()`、`classification_report()`
- 用 k=5 在测试集上评估：准确率 0.9667（29/30）
- 混淆矩阵（行=真实，列=预测）：

| 真实\预测 | setosa | versicolor | virginica |
|---|---:|---:|---:|
| setosa | 10 | 0 | 0 |
| versicolor | 0 | 10 | 0 |
| virginica | 0 | 1 | 9 |

- 每类 precision / recall / F1：
  - setosa：1.0000 / 1.0000 / 1.0000
  - versicolor：0.9091 / 1.0000 / 0.9524
  - virginica：1.0000 / 0.9000 / 0.9474
- 唯一错误：1 个 virginica 被判成 versicolor，所以 virginica 的 recall 是 0.9，versicolor 的 precision 是 0.9091
- 宏平均：precision 0.9697、recall 0.9667、F1 0.9666
- 评估记录：`outputs/metrics/evaluation_k5.txt`

## 10. k 值选择与交叉验证

- 在训练集上做分层 5 折交叉验证，k=1 到 15
- 每一折内部重新拟合标准化器，验证折只做 transform，避免信息泄漏
- 平均准确率最高的是 k=5：0.9663，标准差 0.0169
- k 曲线：`outputs/figures/k_accuracy_curve.png`
- 用 k=5 在测试集上做最终评估：准确率 0.9667（29/30），与基线一致
- 选择理由：k=5 的 5 折平均准确率最高，且标准差较小，说明不同折之间比较稳定
- 记录：`outputs/metrics/k_selection.txt`、`outputs/metrics/evaluation_best_k.txt`、`outputs/metrics/evaluation_predictions.csv`
- 验证：评估指标、分层折和交叉验证测试全部通过；当前项目共 53 个测试全部通过

## 11. 可视化分析

- 二维散点图：`outputs/figures/petal_scatter_with_errors.png`
  - 按真实类别着色
  - 小点是训练样本，大点是测试样本
  - 黑色叉号标出完整 4 维模型的误分类样本，共 1 个（virginica 被判成 versicolor）
- 决策边界图：`outputs/figures/petal_decision_boundary.png`
  - 用花瓣长度、花瓣宽度训练二维 KNN（k=5），背景颜色表示预测区域
  - 二维模型测试集准确率 0.9667；完整 4 维模型也是 0.9667
  - 二维模型只用于可视化，完整模型使用 4 个特征，评估以完整模型为准
- 混淆矩阵热力图：`outputs/figures/confusion_matrix.png`
  - 完整 4 维模型、k=5 的测试集混淆矩阵
- k-准确率曲线：`outputs/figures/k_accuracy_curve.png`（第 9 步生成）
- 所有图都有标题、坐标轴标签和图例；误分类样本已标注
- 可视化说明：`outputs/metrics/visualization_summary.txt`

## 12. sklearn 对照实验

- 使用 `sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, metric="euclidean")`
- 与手写实现使用相同的训练/测试划分、相同的标准化数据、相同的 k=5
- 手写实现准确率：0.9667
- sklearn 准确率：0.9667
- 逐样本预测差异：0 / 30
- 结论：手写 KNN 分类器与 sklearn 结果完全一致，说明距离计算、最近邻搜索和投票逻辑正确
- 记录：`outputs/metrics/sklearn_comparison.txt`

## 13. 异常处理与性能优化

异常检查（全部通过，提示信息清晰）：

| 场景 | 异常提示 |
|---|---|
| 空训练集 | 训练集不能为空 |
| 特征数不一致 | 特征数不一致：训练集有 4 个特征，预测数据有 3 个特征 |
| k 大于训练样本数 | k 不能大于训练样本数：k=1000, 训练样本数=119 |
| k<=0 | k 必须是正整数 |
| 非法距离度量 | metric 必须是 'euclidean' 或 'manhattan' |
| 空预测数据 | 预测数据不能为空 |
| 长度不一致 | X 和 y 的样本数不一致：119 != 118 |
| 平票但缺少距离 | 出现平票时必须提供 distances，按最近邻距离打破平票 |

性能对比（4 个特征，重复 3 次取平均）：

| 数据规模 | 循环版 | 向量化版 | 加速 |
|---|---:|---:|---:|
| 100 x 100 | 136.52 ms | 0.71 ms | 192.5 倍 |
| 300 x 150 | 638.22 ms | 3.26 ms | 195.8 倍 |
| 500 x 250 | 1467.08 ms | 7.37 ms | 199.0 倍 |

- 循环版 `pairwise_distances_loop()` 只用于性能对比，生产代码使用向量化 `pairwise_distances()`
- 向量化版用 NumPy 广播，把双重 Python 循环交给底层 C 实现，本次运行快约 190 倍
- 具体耗时会随机器和运行状态波动，但趋势稳定
- 记录：`outputs/metrics/exception_checks.txt`、`outputs/metrics/performance_comparison.txt`
- 验证：循环版与向量化版结果一致；当前项目共 53 个测试全部通过

## 14. 测试与报告

- 测试命令：

```powershell
& "E:\annaconda1\envs\ai_ml\python.exe" -m unittest discover -v
```

- 测试结果：53 个测试全部通过，记录在 `outputs/metrics/test_results.txt`
- 覆盖范围：距离函数、最近邻搜索、多数投票、分类器、回归器、评估指标、分层交叉验证、异常处理
- 参数影响实验记录：`outputs/metrics/parameter_comparison.txt`
  - 测试集三种设置准确率都是 0.9667
  - 5 折交叉验证：欧氏 + 标准化 0.9663、欧氏 + 未标准化 0.9750、曼哈顿 + 标准化 0.9413、曼哈顿 + 未标准化 0.9496
- 完整实验报告：`report.md`

## 15. 复现约定

- 全项目统一随机种子：`42`（定义在 `knn.py` 的 `RANDOM_SEED`）
- 数据划分、交叉验证等所有随机操作都使用该种子
- 标准化参数只由训练集计算，测试集只做变换，避免信息泄漏

## 16. 运行方式

```bash
python main.py
```

当前进度：全部 14 步已完成，报告见 `report.md`。
