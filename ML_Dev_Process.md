# 建模流程 Ongoing

## **1、问题分析**

- 明确业务问题是机器学习的先决条件，即抽象出该问题为机器学习的预测问题：需要学习什么样的数据作为输入，目标是得到什么样的模型做决策作为输出

## **2、数据构造**

- **数据的含义**
  - 每一行称之为一个样本或样例，每一列称之为一个属性或特征
- **数据选择**
  -  数据和特征决定了机器学习结果的上限，而模型算法只是尽可能逼近这个上限
  -  数据的代表性
  -  数据的业务范围
  -  数据的时间范围
  -  数据样本要与未来的潜在数据相类似
  -  数据样本中要有足够的被观测数据
  -  样本中的好坏比要合理
- **标签定义**
  - 特征是用于描述实例的属性或特征，标签是指一个实例的正确输出(回归问题)或类别(分类问题)，用于训练和评估机器学习模型的目标变量

## **3、探索性分析(EDA)**

- **数据可视化**
- **降维**
- **聚类**
- **线性回归**

## **4、数据预处理**

- **缺失值处理**
  - 缺失率较高，并结合业务可以直接删除该特征变量。经验上可以新增一个`bool`类型的变量特征记录该字段的缺失情况，缺失记为1，非缺失记为0
  - 缺失率较低，结合业务可使用一些缺失值填充手段，如Pandas的`fillna`方法、训练回归模型预测缺失值并填充
  - 不做处理：部分模型如随机森林、xgboost、lightgbm能够处理数据缺失的情况，不需要对缺失数据再做处理
- **异常值处理**
  - 箱线图(Box-plot)
  - S-G滤波器
- **数据离散化**
  - 距离分箱(无监督)
  - 等频分箱(无监督)
  - 利用聚类分箱(无监督)
  - 信息熵分箱(有监督)
  - 基于决策树分箱(有监督)
  - 卡方分箱(有监督)
- **数据标准化**
  - 归一化
    - Min-Max
    - Mean
  - 标准化
    - Z-score
  - 正则化
    - L1
    - L2
    - Dropout
- **数据变换**
  - 对数变换 - 用于减小数据的偏度，使其更接近正态分布
  - Box-Cox变换 - 用于稳定方差并减小偏度
  - **数据标准化**

## **5、特征工程(可选)**

- **特征表示**
  - 离散特征表示
    - 独热编码(One-Hot Encoding)
    - 哑变量编码(Dummy Encoding)
  - 图像表示
  - 文本表示
- **特征衍生**
  - 分组聚合方式
    - 计数
    - 最大值
    - 最小值
    - 平均数
    - 自定义函数
  - 转换方式
    - 数值
      - 加减乘除等运算
      - 排序编码
      - 多列统计
    - 字符串
      - 截取
      - 统计频次
      - 统计字符串长度
    - 日期类型
      - 日期间隔
      - 月份、周几、小时数
- **特征选择**
  - 过滤法
    - 方差选择法
    - 相关系数法
      - 皮尔森相关系数
      - 距离相关系数
    - 卡方检验
    - 互信息法
    - Relief算法
  - 包装法
    - 递归特征消除法
  - 嵌入法
    - 基于惩罚项
    - 基于树模型
- **特征降维**
  - 参考“机器学习大纲”的降维算法

## **6、模型训练**

- **数据集划分**
  - 1万以内
    - 训练集 60%
    - 开发验证集 20%
    - 测试集 20%
  - 100万以内
    - 训练集 98%
    - 开发验证集 1%
    - 测试集 1%
  - 100万以上
    - 训练集 99.5%
    - 开发验证集 0.25%
    - 测试集 0.25%
- **模型方法选择**
- **训练过程**
  - 高偏差(欠拟合) - 训练集和验证集均拟合的不是很佳
    - 增加数据特征
    - 提高模型复杂度
    - 减小正则化系数
  - 高方差(过拟合) - 训练集拟合的很好，但是验证集误差很大，说明模型的泛化能力很差
    - 减小模型复杂度
    - 增大正则化系数
    - 引入先验知识
    - 使用集成模型，即通过多个高方差模型的平均来降低方差
  - 观察指标
    - 损失函数曲线
    - 准确率曲线
    - 曲线在训练和验证集上的收敛情况

## **7、模型评估**

- **模型选择**
  - K折交叉验证
  - 特征选择
- **评估指标**
  - 分类
    - 混淆矩阵(Confusion Matrix)
    - 准确率(Accuracy)
    - 错误率(Error rate)
    - 精确率(Precision)
    - 召回率(Recall)
    - F1-Score
    - ROC曲线(Receiver operating characteristic curve)
    - AUC(Area Under Curve)
    - P-R曲线(Precision Recall Curve)
    - 对数损失(log_loss)
    - 分类指标的文本报告(classification_report)
  - 回归
    - 平均绝对误差(MAE)
    - 平均绝对百分比误差(MAPE)
    - 均方误差(MSE)
    - 均方根误差(RMSE)
    - 归一化均方根误差(NRMSE)
    - 决定系数(R2)

## **8、模型部署**

- TODO  