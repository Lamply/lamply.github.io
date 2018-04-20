---
title: 零核现象
date: 2018-04-19 19:17:12
tags:
---
<div align=center>
![zero](zero-kernels/TingMengDe.bmp)  
</div>
<div align=left>
这里是对零核现象的观察实验记录.  
具体来说, 就是在训练卷积神经网络的过程中发现模型中有大量卷积核的 L1 逐步变为 0 的情况, 这里为了方便简称零核现象.  
<!--more-->
最初遇到这种问题是在 [DAN](https://github.com/MarekKowalski/DeepAlignmentNetwork) 训练时发现的, 当时觉得是太大学习率, ReLU 死亡, 后面降低了 lr 就没出现过了.  
直到后来做分割观察 [ENet](https://github.com/TimoSaemann/ENet) 的预训练模型时又发现了零核的现象, 而且自己用 ShuffleNet 做的分割网络也出现了非常多零核, 这对模型性能显然是有很大影响的, 所以就下定决心解决这个问题.  
根据之前 DAN 的经验, 我自然先试了一下降低学习率, 结果没用, 虽然零增长的速度变慢了, 但还是会出现, 而且随着训练过程零核几乎线性增加 ( 像上图那样 ).  
把每层的零核数作纵坐标, 层数作横坐标, 打印成曲线出来就是这个样子:  
<div align=center>
![zero_shuffle](zero-kernels/coco_train.png)  
</div>
<div align=left>
一共 2312 时个零核, 简直壮观. 当时的网络用了大量 Depthwise Convolution, 所以多半  
于是目光转向












