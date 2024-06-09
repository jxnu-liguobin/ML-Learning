# 机器学习大纲 Ongoing

1. 使用 VS Code + Markmap 插件，以“思维导图”查看
2. 使用 Markmap 输出为HTML，以“思维导图”查看
3. 使用浏览器打开[README.html](README.html)
   
## **技术栈**
- pandas - 数据分析
- numpy - 基础科学计算，维度数组与矩阵运算
- scipy - 基于numpy的科学计算库
  - cluster 矢量量化、K-均值
  - constants 物理和数学常数
  - fftpack 傅里叶变换
  - integrate 积分程序
  - interpolate 插值
  - linalg 线性代数程序
  - ndimage n维图像包
  - odr 正交距离回归
  - optimize 优化
  - signal 信号处理
  - sparse 稀疏矩阵
  - spatial 空间数据结构和算法
  - special 任何特殊数学函数
  - stats 统计
- matplotlib - 可视化、曲线拟合
- scikit-learn - 基于scipy的机器学习库
  - feature_selection 特征选择
  - preprocessing 特征缩放、标准化、编码、缺失值处理
  - model_selection 交叉验证、参数搜索和数据集分割
  - linear_model 线性回归、岭回归、Lasso回归等线性模型
  - svm 支持向量机
  - neighbors k-近邻回归方法
  - linear_model 逻辑回归、线性判别分析等用于分类的线性模型
  - tree 决策树分类器
  - ensemble 随机森林、AdaBoost、梯度提升等集成学习方法
  - cluster K均值、DBSCAN、层次聚类等聚类算法
  - decomposition 主成分分析、独立成分分析等降维方法
  - metrics 模型评估指标：准确率、F1分数、ROC曲线
  - externals 用于模型保存和加载的工具
- pytorch - GPU 加速的张量计算，深度神经网络构建
- tsai - 时间序列深度学习库
- emd-signal - 经验模态分解
- keras - 深度学习库
- tensorflow - 机器学习框架

## **数学基础**

- **线性代数**
  - 标量
  - 向量
  - 张量
  - 范数
  - 矩阵
  - 转置
  - 点积、数量积
  - 向量积、叉积
  - 内积

- **微积分**
  - 导数
  - 微分
  - 偏导数
  - 梯度

- **概率**
  - 贝叶斯定理
  - 方差
  - 标准差
  
## **学习形式分类**

- **监督学习** - 给定数据，预测标签
  - 在监督学习中，要求用于训练算法的训练集必须包含明确的标识或结果。在建立预测模型的时候，监督式学习建立一个学习过程，将预测结果与“训练数据”的实际结果进行比较，不断的调整预测模型，直到模型的预测结果达到一个预期的准确率。监督式学习的常见应用场景如分类问题和回归问题
- **无监督学习** - 给定数据，寻找隐藏的结构
  - 在无监督式学习中，数据并不被特别标识，学习模型是为了推断出数据的一些内在结构。常见的应用场景包括关联规则的学习以及聚类等
- **强化学习** - 给定数据，学习如何选择一系列行动，以最大化长期收益
  - 在强化学习中，输入数据直接反馈到模型，模型必须对此立刻作出调整。常见的应用场景包括动态系统以及机器人控制等。在自动驾驶、视频质量评估、机器人等领域强化学习算法非常流行

## **算法**

### **任务目标分类**
- **回归算法** 
  - 线性回归(Linear Regression)
  - 非线性回归(Non-linear Regression)
  - 逻辑回归(Logistic Regression) 或 对数几率回归
  - 多项式回归(Polynomial Regression)
  - 岭回归(Ridge Regression)
  - 套索回归(Lasso Regression)
  - 弹性网络回归(ElasticNet Regression)
- **分类算法**
  - 逻辑回归(Logistic Regression, LR)
  - K最近邻(k-Nearest Neighbor, KNN)
  - 朴素贝叶斯模型(Naive Bayesian Model, NBM)
  - 隐马尔科夫模型(Hidden Markov Model)
  - 支持向量机(Support Vector Machine)
  - 决策树(Decision Tree)
  - 神经网络(Neural Network)
    - 径向基函数网络(Radial Basis Function Network)
    - 自适应共振理论网络(Adaptive Resonance Theory)
    - 自组织映射神经网络(Self-Organizing Map, SOM)
    - Elman网络
    - 波茨曼机(Boltzmann Machine)
    - 深度学习
  - 集成学习(ada-boost)
- **聚类算法**
  - K均值聚类(K-Means)
  - 层次聚类(Hierarchical Clustering)
  - 混合高斯模型(Gaussian Mixture Model)
  - 降维算法包括：
    - 主成因分析(Principal Component Analysis)
    - 线性判别分析(Linear Discriminant Analysis)

### **梯度下降算法**
- 批量梯度下降法(Batch Gradient Descent)
- 随机梯度下降法(Stochastic Gradient Descent)
- 小批量梯度下降法(Mini-batch Gradient Descent)
- 动量梯度下降法(Gradient descent with Momentum)
- Adam
- RMSprop

## **深度学习**

- **CNN(卷积神经网络)**
  - 图像分类
  - 图像生成
  - 目标检测
  - 语义分割
- **RNN(循环神经网络)**
  - 自然语言处理
  - 语音识别
  - 时间序列预测
- **GAN(生成对抗网络)**
  - 图像生成
  - 风格迁移
  - 文本生成
- **Autoencoder(自动编码器)**
  - 图像去噪
  - 数据重建
  - 生成数据
- **LSTM(长短时记忆网络)**
- **核心概念**
  - **损失函数(loss Function)**
  - **代价函数(cost function)**
  - **梯度下降(gradient descent)**
    - 正向梯度下降
    - 反向梯度下降
  - **激活函数**
    - softmax
    - sigmoid
    - 双曲正切函数(tanh)
    - 修正线性单元(ReLU)
    - 带泄漏的ReLU(Leaky ReLU)
  - **超参数(hyper parameters)**

## **建模流程**

### **1、问题分析**
- 明确业务问题是机器学习的先决条件，即抽象出该问题为机器学习的预测问题：需要学习什么样的数据作为输入，目标是得到什么样的模型做决策作为输出

### **2、数据构造**

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

### **3、探索性分析(EDA)**

- **数据可视化**
- **降维**
- **聚类**
- **线性回归**

### **4、数据预处理**

- **缺失值处理**
  - 缺失率较高，并结合业务可以直接删除该特征变量。经验上可以新增一个`bool`s类型的变量特征记录该字段的缺失情况，缺失记为1，非缺失记为0
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

### **5、特征工程(可选)**

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
  - 主成分分析法(PCA)
  - 多维尺度变换(MDS)
  - 独立成分分析(ICA)

### **6、模型训练**

- **数据集划分**
  - 1万以内
    - 训练集 60%
    - 开发验证集 20%
    - 测试集 20%
  - 100万以内
    - 训练集 98%
    - 开发验证集 1%
    - 测试集 1%
  - 100以上
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

### **7、模型评估**

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
    - 均方误差(MSE)
    - 均方根误差(RMSE)
    - 归一化均方根误差(NRMSE)
    - 决定系数(R2)
- **评估优化**
  - TODO

### **8、模型部署**

- TODO  