# 机器学习大纲 Ongoing

## **基础概念**

- 损失函数(Loss Function)
- 代价函数(Cost Function)
- 梯度下降(Gradient Descent)
- 激活函数(Activation Function)
  - Softmax
  - Sigmoid
  - 双曲正切函数(Tanh)
  - 修正线性单元(ReLU)
  - 带泄漏的ReLU(Leaky ReLU)
- 超参数(Hyper Parameters)
- 神经网络类型
  - 前馈神经网络(Feedforward Neural Network)
  - 反馈神经网络(Feedback Neural Network)
  - 图神经网络(Graph Neural Network, GNN)
- 神经网络结构
  - 输入层(Input Layer)
  - 隐藏层(Hidden Layer)
  - 输出层(Output Layer)
- 学习形式分类
  - **监督学习** - 给定数据，预测标签
    - 在监督学习中，要求用于训练算法的训练集必须包含明确的标识或结果。在建立预测模型的时候，监督式学习建立一个学习过程，将预测结果与“训练数据”的实际结果进行比较，不断的调整预测模型，直到模型的预测结果达到一个预期的准确率。监督式学习的常见应用场景如分类问题和回归问题
  - **无监督学习** - 给定数据，寻找隐藏的结构
    - 在无监督式学习中，数据并不被特别标识，学习模型是为了推断出数据的一些内在结构。常见的应用场景包括关联规则的学习以及聚类等
  - **强化学习** - 给定数据，学习如何选择一系列行动，以最大化长期收益
    - 在强化学习中，输入数据直接反馈到模型，模型必须对此立刻作出调整。常见的应用场景包括动态系统以及机器人控制等。在自动驾驶、视频质量评估、机器人等领域强化学习算法非常流行
  - **集成学习** - 一种思想，通过组合多个学习器(通常称为基学习器或弱学习器)来提高整体模型的预测性能
    - Bagging，有放回地随机抽样，产生不同数据集，训练不同学习器，通过平权投票、求平均值得出最终结果
    - Boosting，每个学习器重点关注前一个学习器不足的的地方进行训练，通过加权投票、加权求和得出最终结果

## **技术栈**

### **基础库(Python)**
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
  
### **可视化库**
- matplotlib - 可视化、曲线拟合
- seaborn - 数据可视化库
- plotly - 数据可视化库

### **算法库**
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
- xgboost - 极限梯度提升算法

## **算法**

### **回归算法**
- 线性回归(Linear Regression)
- 非线性回归(Non-linear Regression)
- 逻辑回归(Logistic Regression) 或 对数几率回归
- 多项式回归(Polynomial Regression)
- 岭回归(Ridge Regression)
- 套索回归(Lasso Regression)
- 弹性网络回归(ElasticNet Regression)
- 自适应增强回归(AdaBoost Regression)
- 极端梯度提升回归(Extreme Gradient Boosting Regression)
- 核脊回归(Kernel Ridge Regression, KRR)
- 梯度提升回归(GradientBoosting Regression, GBR)
- 支持向量回归(Support Vector Regression, SVR)

### **分类算法**
- 逻辑回归(Logistic Regression, LR)
- K最近邻(k-Nearest Neighbor, KNN)
- 朴素贝叶斯模型(Naive Bayesian Model, NBM)
- 贝叶斯网络(Bayesian Network, BN)
- 隐马尔科夫模型(Hidden Markov Model, HMM)
- 支持向量机(Support Vector Machine, SVM)
- 决策树(Decision Tree)
- 神经网络(Neural Network)
  - 径向基函数网络(Radial Basis Function Network, RBF Network)
  - 自适应共振理论网络(Adaptive Resonance Theory, ART)
  - 自组织映射神经网络(Self-Organizing Map, SOM)
  - Elman网络
  - 波茨曼机(Boltzmann Machine)
  - 深度学习(Deep Learning, DL)
- 集成学习
  - 自适应提升(Adaptive Boosting, AdaBoost)
  - 随机森林(Random Forest, RF)
  - 梯度提升决策树(Gradient Boosting Decision Tree, GBDT)
  - 极端梯度提升(Extreme Gradient Boosting, XGBoost)

### **聚类算法**
- K均值聚类(K-Means)
- 层次聚类(Hierarchical Clustering)
- 混合高斯模型(Gaussian Mixture Model, GMM)
- 降维算法包括：
  - 主成因分析(Principal Component Analysis, PCA)
  - 线性判别分析(Linear Discriminant Analysis, LDA)

### **梯度下降算法**
- 批量梯度下降法(Batch Gradient Descent)
- 随机梯度下降法(Stochastic Gradient Descent, SGD)
- 小批量梯度下降法(Mini-batch Gradient Descent)
- 动量梯度下降法(Gradient Descent with Momentum)
- Adam算法(Adaptive Moment Estimation)
- RMSProp算法(Root Mean Square Prop)

## **深度学习**

### **卷积神经网络(CNN)**
- 核心结构
  - 卷积层(Convolutional Layer)
  - 池化层(Pooling Layer)
  - 全连接池(Full Connected Layer)
- 应用场景
  - 图像分类
  - 图像生成
  - 目标检测
  - 语义分割
  
### **循环神经网络(RNN)**
- 应用场景
  - 自然语言处理
  - 时间序列预测
  - 语音识别
  
### **生成对抗网络(GAN)**
- 核心结构
  - 生成器(Generator)
  - 判别器(Discriminator)
- 应用场景
  - 图像生成、图像增强、视频生成
  - 风格迁移
  - 文本生成
  
### **自动编码器(Autoencoder)**
- 核心结构
  - 编码器(Encoder)
  - 解码器(Decoder)
- 应用场景
  - 图像去噪
  - 数据重建
  - 生成数据
  - 特征提取
  
### **长短时记忆网络(LSTM)**
- 核心结构
  - 遗忘门(ForgetGate)
  - 输入门(InputGate)
  - 输出门(OutputGate)
- 应用场景
  - 自然语言处理
  - 时间序列预测
  - 语音识别
  - 图像描述生成
  - 行为识别

## **数学基础**

### **线性代数**
- 标量
- 向量
- 张量
- 范数
- 矩阵
- 转置
- 点积、数量积
- 向量积、叉积
- 内积

### **微积分**
- 导数
- 微分
- 偏导数
- 梯度

### **概率**
- 贝叶斯定理
- 方差
- 标准差