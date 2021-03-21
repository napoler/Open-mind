---
layout: home
title: 关于OpenMind
---

## 关于OpenMind

收集分享各种深度学习相关内容

## 关于资源

内容中难免会大量使用Youtube、Google、Facebook、Github访问困难的资源，如何访问需用户自行解决。

## License

This work is open sourced under the Apache License, Version 2.0.


# 算法& 模型

各种算法合集，各种模型合集

[潜在语义分析（latent semantic analysis](./2021-03-21-潜在语义分析.html)

主题生成模型（Latent Dirichlet Allocation）
>LDA（Latent Dirichlet Allocation）是一种文档主题生成模型，也称为一个三层贝叶斯概率模型，包含词、主题和文档三层结构。所谓生成模型，就是说，我们认为一篇文章的每个词都是通过“以一定概率选择了某个主题，并从这个主题中以一定概率选择某个词语”这样一个过程得到。文档到主题服从多项式分布，主题到词服从多项式分布。
    LDA是一种非监督机器学习技术，可以用来识别大规模文档集（document collection）或语料库（corpus）中潜藏的主题信息。它采用了词袋（bag of words）的方法，这种方法将每一篇文档视为一个词频向量，从而将文本信息转化为了易于建模的数字信息。但是词袋方法没有考虑词与词之间的顺序，这简化了问题的复杂性，同时也为模型的改进提供了契机。每一篇文档代表了一些主题所构成的一个概率分布，而每一个主题又代表了很多单词所构成的一个概率分布。


因子分析（Factor Analysis）
> 因子分析法是指从研究指标相关矩阵内部的依赖关系出发，把一些信息重叠、具有错综复杂关系的变量归结为少数几个不相关的综合因子的一种多元统计分析方法。 基本思想是：根据相关性大小把变量分组，使得同组内的变量之间相关性较高，但不同组的变量不相关或相关性较低，每组变量代表一个基本结构一即公共因子。

核函主成分（kernal pca）
> 核主成分分析（英语：kernel principal component analysis，简称kernel PCA）是多变量统计领域中的一种分析方法，是使用核方法对主成分分析的非线性扩展，即将原数据通过核映射到再生核希尔伯特空间后再使用原本线性的主成分分析。
https://zh.wikipedia.org/wiki/%E6%A0%B8%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90

主成分方法（PCA）
> 在多元统计分析中，主成分分析（英語：Principal components analysis，PCA）是一種统计分析、簡化數據集的方法。 它利用正交变换来对一系列可能相关的变量的观测值进行线性变换，从而投影为一系列线性不相关变量的值，这些不相关变量称为主成分（Principal Components）。
https://zh.wikipedia.org/zh-sg/%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90#:~:text=%E5%9C%A8%E5%A4%9A%E5%85%83%E7%BB%9F%E8%AE%A1%E5%88%86%E6%9E%90%E4%B8%AD,%E4%B8%BB%E6%88%90%E5%88%86%EF%BC%88Principal%20Components%EF%BC%89%E3%80%82

层次聚类（Hierarchical clustering）——支持多种距离
> 层次聚类(Hierarchical Clustering)是聚类算法的一种，通过计算不同类别数据点间的相似度来创建一棵有层次的嵌套聚类树。 在聚类树中，不同类别的原始数据点是树的最低层，树的顶层是一个聚类的根节点。 创建聚类树有自下而上合并和自上而下分裂两种方法，本篇文章介绍合并方法。

Kmeans算法
> k均值聚类算法（k-means clustering algorithm）是一种迭代求解的聚类分析算法，其步骤是，预将数据分为K组，则随机选取K个对象作为初始的聚类中心，然后计算每个对象与各个种子聚类中心之间的距离，把每个对象分配给距离它最近的聚类中心。 聚类中心以及分配给它们的对象就代表一个聚类。

Knn算法
> K最近邻(k-Nearest Neighbors，KNN) 算法是一种分类算法，也是最简单易懂的机器学习算法，没有之一。 ... 该算法的思想是：一个样本与数据集中的k个样本最相似，如果这k个样本中的大多数属于某一个类别，则该样本也属于这个类别。

典型相关分析（CCA）
>在统计学中，典型相关分析(Canonical Correlation Analysis)是对互协方差矩阵的一种理解。如果我们有两个随机变量向量 X = (X1, ..., Xn) 和 Y = (Y1, ..., Ym) 并且它们是相关的，那么典型相关分析会找出 Xi 和 Yj 的相互相关最大的线性组合。[1]T·R·Knapp指出“几乎所有常见的参数测试的意义可视为特殊情况的典型相关分析，这是研究两组变量之间关系的一般步骤。”[2] 这个方法在1936年由哈罗德·霍特林首次引入
https://zh.wikipedia.org/wiki/%E5%85%B8%E5%9E%8B%E7%9B%B8%E5%85%B3


逻辑回归（Logistic regression）
>logistic回归又称logistic回归分析，是一种广义的线性回归分析模型，常用于数据挖掘，疾病自动诊断，经济预测等领域。例如，探讨引发疾病的危险因素，并根据危险因素预测疾病发生的概率等。以胃癌病情分析为例，选择两组人群，一组是胃癌组，一组是非胃癌组，两组人群必定具有不同的体征与生活方式等。因此因变量就为是否胃癌，值为“是”或“否”，自变量就可以包括很多了，如年龄、性别、饮食习惯、幽门螺杆菌感染等。自变量既可以是连续的，也可以是分类的。然后通过logistic回归分析，可以得到自变量的权重，从而可以大致了解到底哪些因素是胃癌的危险因素。同时根据该权值可以根据危险因素预测一个人患癌症的可能性。


稳健回归（Robustness regression）
>稳健回归（robust regression）是统计学稳健估计中的一种方法，其主要思路是将对异常值十分敏感的经典最小二乘回归中的目标函数进行修改。 经典最小二乘回归以使误差平方和达到最小为其目标函数。 因为方差为一不稳健统计量，故最小二乘回归是一种不稳健的方法。 不同的目标函数定义了不同的稳健回归方法。

多项式回归（Polynomial regression——多项式基函数回归）
>在统计学中， 多项式回归是回归分析的一种形式，其中自变量 x 和因变量 y 之间的关系被建模为关于x 的n 次多项式。 多项式回归拟合x的值与y 的相应条件均值之间的非线性关系，表示为E(y|x)，并且已被用于描述非线性现象，例如组织的生长速率、湖中碳同位素的分布以及沉积物和流行病的发展。
https://zh.wikipedia.org/zh-sg/%E5%A4%9A%E9%A1%B9%E5%BC%8F%E5%9B%9E%E5%BD%92#:~:text=%E5%9C%A8%E7%BB%9F%E8%AE%A1%E5%AD%A6%E4%B8%AD%EF%BC%8C%20%E5%A4%9A%E9%A1%B9%E5%BC%8F,%E5%92%8C%E6%B5%81%E8%A1%8C%E7%97%85%E7%9A%84%E5%8F%91%E5%B1%95%E3%80%82

高斯过程回归（Gaussian Process Regression）
> 高斯过程回归是一种贝叶斯推断。 什么是贝叶斯推断这里就不展开细讲了，简单来说就是利用贝叶斯定理结合新的证据及以前的先验概率，来得到后验概率

偏最小二乘回归（PLS）
>偏最小二乘回归（英语：Partial least squares regression， PLS回归）是一种统计学方法，与主成分回归有关系，但不是寻找响应和独立变量之间最小方差的超平面，而是通过投影预测变量和观测变量到一个新空间来寻找一个线性回归模型。因为数据X和Y都会投影到新空间，PLS系列的方法都被称为双线性因子模型。当Y是分类数据时有“偏最小二乘判别分析（英语：Partial least squares Discriminant Analysis， PLS-DA）”，是PLS的一个变形。https://zh.wikipedia.org/wiki/%E5%81%8F%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98%E5%9B%9E%E5%BD%92

套索回归（Lasso）
>Lasso回归是一种同时进行特征选择和正则化（数学）的回归分析方法，旨在增强统计模型的预测准确性和可解释性，.

弹性网络回归（Elastic Net）
>弹性网络是一种使用 L1，L2范数作为先验正则项训练的线性回归模型.这种组合允许学习到一个只有少量参数是非零稀疏的模型，就像 Lasso一样，但是它仍然保持一些像Ridge的正则性质。我们可利用 l1_ratio 参数控制L1和L2的凸组合。弹性网络是一不断叠代的方法。

贝叶斯回归（Bayesian Regression）
>贝叶斯线性回归（Bayesian linear regression）是使用统计学中贝叶斯推断（Bayesian inference）方法求解的线性回归（linear regression）模型

AdaBoost
>Adaboost算法是1995年由 Freund和 Schapire提出的

GBDT（Gradient Tree Boosting）
>GBDT模型是一个加法模型，它串行地训练一组CART回归树，最终对所有回归树的预测结果加和，由此得到一个强学习器，每一颗新树都拟合当前损失函数的负梯度方向。

最小二乘回归（OLS）
>最小二乘法（英语：least squares method），又称最小平方法，是一种数学优化建模方法。它通过最小化误差的平方和寻找数据的最佳函数匹配。
    利用最小二乘法可以简便的求得未知的数据，并使得求得的数据与实际数据之间误差的平方和为最小。
    https://zh.wikipedia.org/wiki/%E6%9C%80%E5%B0%8F%E4%BA%8C%E4%B9%98%E6%B3%95

岭回归（Ridge Regression）

核岭回归（Kernel ridge regression）

支持向量机回归（SVR）

决策树算法（decision tree）

Bagging

随机森林（Random Forest）

朴素贝叶斯算法（Naive Bayes）

二次判别分析（QDA）

支持向量机（SVM）

Knn算法

线性判别分析（LDA）

big bird

Switch Transformer

MT5

UniLM

alphafold2

XLM

XLM-ProphetNet

MASS

PREFOMER

XLM-RoBERTa

XLNet

XLSR-Wav2Vec2

Reformer

RetriBERT

RoBERTa

Speech2Text

SqueezeBERT

T5

TAPAS

Transformer XL

Wav2Vec2

MPNet

OpenAI GPT

OpenAI GPT2

Pegasus

PhoBERT

ProphetNet

RAG

M2M100

MarianMT

MBart and MBart-50

MobileBERT

LXMERT

Longformer

LED

LayoutLM

I-BERT

herBERT

Funnel Transformer

FSMT

FlauBERT

Encoder Decoder Models

ELECTRA

DPR

DistilBERT

DialoGPT

DeBERTa

CTRL

ConvBERT

CamemBERT

BORT

BertGeneration

Blenderbot

Bertweet

BARThez

BART

ALBERT

NNLM(Neural Network Language Model) 

BERT

The Transformer

Bi-LSTM with Attention 

Seq2Seq

Seq2Seq with Attention

Bi-LSTM

TextLSTM 

TextCNN

TextRNN

FastText(Application Level)

Word2Vec(Skip-gram)












# 数据

开源数据

# 学习资料

好用的学习资料
