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
  - 高斯误差线性单元(GELU)
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
  - **监督学习** 给定数据，预测标签
    - 在监督学习中，要求用于训练算法的训练集必须包含明确的标识或结果。在建立预测模型的时候，监督式学习建立一个学习过程，将预测结果与“训练数据”的实际结果进行比较，不断的调整预测模型，直到模型的预测结果达到一个预期的准确率。监督式学习的常见应用场景如分类问题和回归问题
  - **无监督学习** 给定数据，寻找隐藏的结构
    - 在无监督式学习中，数据并不被特别标识，学习模型是为了推断出数据的一些内在结构。常见的应用场景包括关联规则的学习以及聚类等
  - **强化学习** 给定数据，学习如何选择一系列行动，以最大化长期收益
    - 在强化学习中，输入数据直接反馈到模型，模型必须对此立刻作出调整。常见的应用场景包括动态系统以及机器人控制等。在自动驾驶、视频质量评估、机器人等领域强化学习算法非常流行
  - **集成学习** 一种思想，通过组合多个学习器(通常称为基学习器或弱学习器)来提高整体模型的预测性能
    - Bagging，有放回地随机抽样，产生不同数据集，训练不同学习器，通过平权投票、求平均值得出最终结果
    - Boosting，每个学习器重点关注前一个学习器不足的的地方进行训练，通过加权投票、加权求和得出最终结果

## **技术栈(Python)**

### **基础库**
- pandas 数据分析
- numpy 基础科学计算，维度数组与矩阵运算
- scipy 基于numpy的科学计算库
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
- matplotlib 可视化、曲线拟合
- seaborn 数据可视化库
- plotly 数据可视化库
- Netron 机器学习和深度学习模型可视化
- PlotNeuralNet 神经网络结构可视化
- PyTorchviz 可视化PyTorch网络结构

### **算法库**
- scikit-learn 基于scipy的机器学习库
  - neural_network 神经网络 
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
- pytorch GPU 加速的张量计算，深度神经网络构建
- tsai 时间序列深度学习库
- emd-signal 经验模态分解
- keras 深度学习库
- tensorflow 机器学习框架
- xgboost 极端梯度提升算法
- shap 用于解释机器学习模型的输出

## **算法**

### **回归算法**
- 线性回归(Linear Regression)
- 非线性回归(Non-linear Regression)
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
- 逻辑回归(Logistic Regression, LR) 或 对数几率回归
- K最近邻(k-Nearest Neighbor, KNN)
- 朴素贝叶斯模型(Naive Bayesian Model, NBM)
- 贝叶斯网络(Bayesian Network, BN)
- 隐马尔科夫模型(Hidden Markov Model, HMM)
- 支持向量机(Support Vector Machine, SVM)
- 决策树(Decision Tree)
- 神经网络(Neural Network)
  - 径向基函数(Radial Basis Function, RBF)
  - 自适应谐振理论(Adaptive Resonance Theory, ART)
  - 自组织映射(Self-Organizing Map, SOM)
  - Elman网络
  - 受限波尔茨曼机(Restricted Boltzmann Machine)
  - 深度学习(Deep Learning, DL)
  - 自编码器(Autoencoder, AE)
  - 感知机(Perception)
- 集成学习
  - 自适应提升(Adaptive Boosting, AdaBoost)
  - 随机森林(Random Forest, RF)
  - 梯度提升决策树(Gradient Boosting Decision Tree, GBDT)
  - 极端梯度提升(Extreme Gradient Boosting, XGBoost)

### **降维算法**
- 自编码器(Autoencoder, AE)
- 主成分分析(Principal Components Analysis, PCA)
- 多维缩放(Multiple Dimensional Scaling, MDS)
- 线性判别分析(Linear Discriminant Analysis, LDA)
- 等度量映射(Isometric Mapping, Isomap)
- 局部线性嵌入(Locally Linear Embedding, LLE)
- T分布随机近邻嵌入(T-Distribution Stochastic Neighbour Embedding, t-SNE)

### **聚类算法**
- 原型聚类(Prototype-Based Clustering)
  - K均值聚类(K-Means Clustering)
  - 高斯混合聚类(Mixture-of-Gaussian Clustering)
  - 学习向量量化(Learning Vector Quantization, LVQ)
- 层次聚类(Hierarchical Clustering)
  - AGNES(Agglomerative Nesting)
- 密度聚类(Density-Based Clustering)
  - 均值漂移(Mean Shift)
  - DBSCAN(Density-Based Spatial Clustering of Applications with Noise)
- 图团体检测(Graph Community Detection)

### **梯度下降算法**
- 批量梯度下降法(Batch Gradient Descent)
- 随机梯度下降法(Stochastic Gradient Descent, SGD)
- 小批量梯度下降法(Mini-batch Gradient Descent)
- 动量梯度下降法(Gradient Descent with Momentum)
- Adam算法(Adaptive Moment Estimation)
- RMSProp算法(Root Mean Square Prop)

### **超参优化算法**
- 随机搜索(Random Search)
- 网格搜索(Grid Search)
- 贝叶斯优化(Bayesian Optimization)
- 遗传算法(Genetic Algorithm, GA)
- 基于梯度的优化(Gradient-Based Optimization)
- 基于种群的优化(Population-Based Optimization, PBO)
- 参数配置空间中的迭代局部搜索(ParamILS)
- LBFGS (Limited-memory Broyden–Fletcher–Goldfarb–Shanno)
  
## **深度学习**

### **卷积神经网络(CNN)**
- 核心结构
  - 卷积层(Convolutional Layer)
  - 池化层(Pooling Layer)
  - 全连接层(Full Connected Layer)
- 应用场景
  - 图像分类
  - 图像生成
  - 目标检测
  - 语义分割
  
### **循环神经网络(RNN)**

#### **应用场景**
 - 自然语言处理
 - 时间序列预测

#### **长短时记忆网络(LSTM)**
- 核心结构
  - 遗忘门(Forget Gate)
  - 输入门(Input Gate)
  - 输出门(Output Gate)
  - 候选记忆元(Candidate Memory Cell)

#### **门控循环单元(GRU)**
- 核心结构
  - 重置门(Reset Gate)
  - 更新门(Update Gate)
  - 隐状态(Hidden State)
  - 候选隐状态(Candidate Hidden State)
  
### **生成对抗网络(GAN)**
- 核心结构
  - 生成器(Generator)
  - 判别器(Discriminator)
- 应用场景
  - 图像生成、图像增强、视频生成
  - 风格迁移
  - 文本生成

### [tsai](https://github.com/timeseriesAI/tsai)库支持的时序模型:

- [LSTM](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN.py) (Hochreiter, 1997) ([paper](https://ieeexplore.ieee.org/abstract/document/6795963/))
- [GRU](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN.py) (Cho, 2014) ([paper](https://arxiv.org/abs/1412.3555))
- [MLP](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/MLP.py) - Multilayer Perceptron (Wang, 2016) ([paper](https://arxiv.org/abs/1611.06455))
- [FCN](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/FCN.py) - Fully Convolutional Network (Wang, 2016) ([paper](https://arxiv.org/abs/1611.06455))
- [ResNet](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/ResNet.py) - Residual Network (Wang, 2016) ([paper](https://arxiv.org/abs/1611.06455))
- [LSTM-FCN](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py) (Karim, 2017) ([paper](https://arxiv.org/abs/1709.05206))
- [GRU-FCN](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py) (Elsayed, 2018) ([paper](https://arxiv.org/abs/1812.07683))
- [mWDN](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/mWDN.py) - Multilevel wavelet decomposition network (Wang, 2018) ([paper](https://dl.acm.org/doi/abs/10.1145/3219819.3220060))
- [TCN](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TCN.py) - Temporal Convolutional Network (Bai, 2018) ([paper](https://arxiv.org/abs/1803.01271))
- [MLSTM-FCN](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/RNN_FCN.py) - Multivariate LSTM-FCN (Karim, 2019) ([paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608019301200))
- [InceptionTime](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/InceptionTime.py) (Fawaz, 2019) ([paper](https://arxiv.org/abs/1909.04939))
- [Rocket](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/ROCKET.py) (Dempster, 2019) ([paper](https://arxiv.org/abs/1910.13051))
- [XceptionTime](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/XceptionTime.py) (Rahimian, 2019) ([paper](https://arxiv.org/abs/1911.03803))
- [ResCNN](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/ResCNN.py) - 1D-ResCNN (Zou , 2019) ([paper](https://www.sciencedirect.com/science/article/pii/S0925231219311506))
- [TabModel](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TabModel.py) - modified from fastai’s [TabularModel](https://docs.fast.ai/tabular.model.html#TabularModel)
- [OmniScale](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/OmniScaleCNN.py) - Omni-Scale 1D-CNN (Tang, 2020) ([paper](https://arxiv.org/abs/2002.10061))
- [TST](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TST.py) - Time Series Transformer (Zerveas, 2020) ([paper](https://dl.acm.org/doi/abs/10.1145/3447548.3467401))
- [TabTransformer](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TabTransformer.py) (Huang, 2020) ([paper](https://arxiv.org/pdf/2012.06678))
- [TSiT](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TSiTPlus.py) Adapted from ViT (Dosovitskiy, 2020) ([paper](https://arxiv.org/abs/2010.11929))
- [MiniRocket](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/MINIROCKET.py) (Dempster, 2021) ([paper](https://arxiv.org/abs/2102.00457))
- [XCM](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/XCM.py) - An Explainable Convolutional Neural Network (Fauvel, 2021) ([paper](https://hal.inria.fr/hal-03469487/document))
- [gMLP](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/gMLP.py) - Gated Multilayer Perceptron (Liu, 2021) ([paper](https://arxiv.org/abs/2105.08050))
- [TSPerceiver](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TSPerceiver.py) - Adapted from Perceiver IO (Jaegle, 2021) ([paper](https://arxiv.org/abs/2107.14795))
- [GatedTabTransformer](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/GatedTabTransformer.py) (Cholakov, 2022) ([paper](https://arxiv.org/abs/2201.00199))
- [TSSequencerPlus](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/TSSequencerPlus.py) - Adapted from Sequencer (Tatsunami, 2022) ([paper](https://arxiv.org/abs/2205.01972))
- [PatchTST](https://github.com/timeseriesAI/tsai/blob/main/tsai/models/PatchTST.py) - (Nie, 2022) ([paper](https://arxiv.org/abs/2211.14730))

其他变体，诸如: TransformerModel、LSTMAttention、GRUAttention等

## **数学基础**

### **线性代数**
- 标量
- 向量
- 张量
- 范数
- 矩阵
- 转置
- 矩阵乘积
- (向量)点积
- (矩阵)向量积

### **微积分**
- 导数
- 微分
- 偏导数
- 梯度

### **概率**
- 贝叶斯规则
- 期望
- 方差
- 标准差
- 协方差
- 概率分布
