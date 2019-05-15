---
title: Kaggle "Amazon from Space" 经验分享
date: 2017-10-17 13:13:06
tags: [技术经验]
---
看了 Kaggle 亚马逊雨林卫星图分类比赛第一名 [Planet: Understanding the Amazon from Space, 1st Place Winner's Interview](http://blog.kaggle.com/2017/10/17/planet-understanding-the-amazon-from-space-1st-place-winners-interview/), 学到了些 trick, 这里记录一下  
1. 关于 precision 和 recall 的权衡, 可以在原有的 log loss 上增加 F_beta - score loss 来达到, 如:
$$F_1 = 2\cdot\frac{precision\cdot recall}{precision+recall}$$
而
$$F_\beta = (1+\beta^2)\cdot\frac{precision\cdot recall}{(\beta^2\cdot precision)+recall}$$
使用 F_2 - score 时, 代表 recall 比 precision 重要, F_0.5 - score, 代表 precision 比 recall 重要  

2. 关于分类标签变量存在相关或部分对立的情况下的预测, 可以对最后一层输出的概率使用 ridge regression ( 岭回归, 即加了 L2 正则项的最小二乘, 鼓励回归参数尽可能利用到所有的相关变量, 惩罚可能存在的个别变量 dominant 的大正负值参数 cancel 得出结果的情况 ). 该方法也可以用在 ensemble 多个模型上, 最终对输出结果做 ridge regression





