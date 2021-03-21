---
title: nlp-tutorial各种nlp示例合集
author: TerryChan
date: 2021-03-21
category: nlp
layout: post
---

## 原始地址

[访问原始数据地址](https://github.com/graykode/nlp-tutorial)


## nlp-tutorial

`nlp-tutorial`是一个使用**Pytorch**学习NLP（自然语言处理）的教程。NLP中的大部分模型都是用不到**100行**的代码实现的（注释或空行除外）。

- [08-14-2020]旧的TensorFlow v1代码在[存档文件夹](archive)中存档。为了方便初学者阅读，只支持pytorch 1.0以上版本。


## Curriculum - (Example Purpose)

#### 1. 基本嵌入模型

- 1-1. [NNLM(Neural Network Language Model)](1-1.NNLM) - **预测下一个词*
  - Paper -  [A Neural Probabilistic Language Model(2003)](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
  - Colab - [NNLM.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-1.NNLM/NNLM.ipynb)
- 1-2. [Word2Vec(Skip-gram)](1-2.Word2Vec) - **嵌入词语和图表展示**
  - Paper - [Distributed Representations of Words and Phrases
    and their Compositionality(2013)](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
  - Colab - [Word2Vec.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-2.Word2Vec/Word2Vec_Skipgram(Softmax).ipynb)
- 1-3. [FastText(Application Level)](1-3.FastText) - **Sentence Classification**
  - Paper - [Bag of Tricks for Efficient Text Classification(2016)](https://arxiv.org/pdf/1607.01759.pdf)
  - Colab - [FastText.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/1-3.FastText/FastText.ipynb)



#### 2. CNN(Convolutional Neural Network)

- 2-1. [TextCNN](2-1.TextCNN) - **二元情绪分类**
  - Paper - [Convolutional Neural Networks for Sentence Classification(2014)](http://www.aclweb.org/anthology/D14-1181)
  - [TextCNN.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/2-1.TextCNN/TextCNN.ipynb)



#### 3. RNN(Recurrent Neural Network)

- 3-1. [TextRNN](3-1.TextRNN) - **预测下一步**
  - Paper - [Finding Structure in Time(1990)](http://psych.colorado.edu/~kimlab/Elman1990.pdf)
  - Colab - [TextRNN.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-1.TextRNN/TextRNN.ipynb)
- 3-2. [TextLSTM](https://github.com/graykode/nlp-tutorial/tree/master/3-2.TextLSTM) - **自动完成**
  - Paper - [LONG SHORT-TERM MEMORY(1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)
  - Colab - [TextLSTM.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-2.TextLSTM/TextLSTM.ipynb)
- 3-3. [Bi-LSTM](3-3.Bi-LSTM) - **预测长句中的下一个单词**
  - Colab - [Bi_LSTM.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/3-3.Bi-LSTM/Bi_LSTM.ipynb)



#### 4. Attention Mechanism

- 4-1. [Seq2Seq](4-1.Seq2Seq) - **更改字词**
  - Paper - [Learning Phrase Representations using RNN Encoder–Decoder
    for Statistical Machine Translation(2014)](https://arxiv.org/pdf/1406.1078.pdf)
  - Colab - [Seq2Seq.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-1.Seq2Seq/Seq2Seq.ipynb)
- 4-2. [Seq2Seq with Attention](4-2.Seq2Seq(Attention)) - **翻译**
  - Paper - [Neural Machine Translation by Jointly Learning to Align and Translate(2014)](https://arxiv.org/abs/1409.0473)
  - Colab - [Seq2Seq(Attention).ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-2.Seq2Seq(Attention)/Seq2Seq(Attention).ipynb)
- 4-3. [Bi-LSTM with Attention](4-3.Bi-LSTM(Attention)) - **二元情绪分类**
  - Colab - [Bi_LSTM(Attention).ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/4-3.Bi-LSTM(Attention)/Bi_LSTM(Attention).ipynb)



#### 5. Model based on Transformer

- 5-1.  [The Transformer](5-1.Transformer) - **翻译**
  - Paper - [Attention Is All You Need(2017)](https://arxiv.org/abs/1706.03762)
  - Colab - [Transformer.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer.ipynb), [Transformer(Greedy_decoder).ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-1.Transformer/Transformer(Greedy_decoder).ipynb)
- 5-2. [BERT](5-2.BERT) - **分类下句与预测遮蔽的Tokens**
  - Paper - [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding(2018)](https://arxiv.org/abs/1810.04805)
  - Colab - [BERT.ipynb](https://colab.research.google.com/github/graykode/nlp-tutorial/blob/master/5-2.BERT/BERT.ipynb)



## 依赖

- Python 3.5+
- Pytorch 1.0.0+



## Author


- 郑太焕(Jeff Jung) @graykode
- 作者邮箱：nlkey2022@gmail.com
- 感谢[mojitok](http://mojitok.com/)作为NLP研究实习生。

