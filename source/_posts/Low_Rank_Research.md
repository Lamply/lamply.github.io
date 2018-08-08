---
title: 关于 LRA 和 Force Regularization 的探索
date: 2018-08-07 13:13:06
tags: [技术经验, 模型加速和压缩]
---
这部分是将上一篇提到的 _Force Regularization_ 和 _LRA_ 用于实际项目的效果，虽然现在看来不是很严谨，不过算是一次很好的尝试。
<!--more-->
### 设定
为了探索 _Low-Rank Approximations ( LRA )_ 和 _Force Regularization_  ( 参考 _Wen. "Coordinating Filters for Faster Deep Neural Networks" ICCV 2017_ ) 在我的工程上的实际效果, 进行了一些实际测试. 由于时间限制, 主要进行了两次探索, 分别为:
- LRA
  - 单次大降秩(rank ratio 0.48)
  - 在大降秩的基础上小降秩(rank ratio 0.8)
- LRA + Force Regularization
  - 多次迭代 LRA (每次 rank ratio 0.9), FR (0.003 Degradation)

此外还有在 MNIST 上对 LeNet 的对照测试, 结果与文中叙述结论基本一致, Force Regularization 后 LRA 带来的压缩率有进一步的提高, 但主要在全连接层体现 (实验时卷积层个数完全没变), 尚未使用其他网络进行测试, 也没有观察出 Force Regularization 后卷积核的变化, 可能需要进一步实验 (调整 Force Regularization 参数, 用更好的可视化方法 t-SNE 等)

### 结果
- 原模型结果

|    测试  |  分数  |
| :------: |:------:|
| 原准确率 | 0.8993 |
| 原召回率 | 0.9017 | 
| F1-Score | 0.8919 |

- LRA  
	单次 0.48 大降秩后 finetune 17300 iters, 接着 0.8 小降秩 finetune 40000 iters:

|  测试  |  0.48分数  | 0.48+0.8分数  |
| :----: |:------:|:------:|
| 准确率 | 0.8947 | 0.9181 |
| 召回率 | 0.8868 | 0.8157 |
| F1-Score | 0.8813 | 0.8489 |

- LRA + Force Regularization  
  此版本由于原工程正在改进, 所以使用了新的方法 (修改了网络输出层的卷积个数以及输入的通道数, 准确率略微提高, 召回率变化不大)  
  在新方法训练的模型下进行 0.003 Degradation 的 Force Regularization, 400 iters ( 50 iters/epoch ) 后 0.9 rank ratio 降秩, finetune 1000 iters后 继续 0.9 rank ratio 降秩, finetune 1300 iters 后得到收敛结果

|   测试 | 源模型 FR | 第一次降秩 |  第二次降秩  |
| :----: |:------:|:------:|:------:|
| 准确率 | 0.9252 | 0.9207 | 0.9228 |
| 召回率 | 0.7729 | 0.8520 | 0.8443 | 
| F1-Score | 0.8298 | 0.8731 | 0.8698 |

### 分析
1. 从 LRA 可以看出, 单次大降秩也能恢复到接近源模型的效果(召回率下降大约2%), 模型大小压缩明显(12.7M => 4.8M), 但是再度降秩模型效果开始较大幅度下降(召回率再度下降7%), 且模型大小变化不大(4.8M => 3.7M)
2. FR 后模型的召回率迅速降低, 但理论上在此基础上再进行多次降秩并最终 finetune 应该是能恢复效果的, 问题 loss 已经几乎收敛, 无法看出有明显下降, 召回率仍然有较大损失, 所以怀疑可能需要降低训练 force regularization, 和 learning rate, 或者有可能是 FR 未足够 finetune 的问题??  

### 后续
进行了FR再测试, 对原模型进行了更久的 finetune, 得到

|    测试  |  分数  |
| :------: |:------:|
|   准确率 | 0.9194 |
|   召回率 | 0.9018 | 
| F1-Score | 0.9030 |

对于速度, 进行了几个小实验, 似乎该方法在小网络上会由于增加卷积层所以减慢前向速度, 而且比起低秩增速, 卷积层的增加带来的负面影响似乎更大. 至少以本项目来说是有些许降速的.



