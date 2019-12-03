---
title: DeepLab系列论文简略记录
date: 2018-03-01 21:12:06
tags: [论文笔记]
---
这部分是关于语义分割网络 DeepLab 系列的三篇论文。尽管经验性的技巧很多，但就效果而言还是很不错的，有不少值得参考的地方。
<!--more-->
<br>
### DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs
--------------------------------
#### 原文理解
三个贡献:  
1. 明确表明了上采样滤波器或者叫 '空洞卷积' 是 dense prediction 任务中的重要工具. 空洞卷积允许明确控制滤波器在计算特征响应后的分辨率, 也允许有效放大滤波器的感受野, 从而无计算量和参数量增加地聚合更大的上下文信息
2. 提出了空间金字塔空洞池化 ( ASPP ), 能在多尺度上分割物体
3. 结合 DCNN 和概率图模型提升边缘准确度

#### 方法简述
将后面一层 pooling 和 conv stride 改为 1, 换成 atrous conv, 最后并行多个不同 rate 的 atrous conv, 然后 fuse 在一起. 加上多尺度输入, COCO 预训练, randomly rescaling 扩增, CRF, 最终得出结果.

<br>
<br>
### Rethinking Atrous Convolution for Semantic Image Segmentation
------------------------------------
#### 原文理解
针对多尺度分割, 设计了级联或并联的 atrous conv 模块, 改进了 ASPP, 没有 CRF 后处理也达到了 SOTA  
- Multi-grid Method
  - 一个 block 内几个卷积, 分不同的 dilation rate, 比如三个卷积则原来 { 1,1,1 } 可以变为 { 1,2,4 }, 然后乘上该 block 的 dilation rate. 也就是说本来要 stride 2 的, 改用 atrous conv 后, 后面 block 卷积的 dilation rate 为 2 x { 1,2,4 } = { 2,4,8 }
  - 但是 ResNet 的 block 难道不是只有一个 3x3 ??? 1x1 哪来的 dilation ???? 原文没有提及, github 有相关讨论, 大致可能一是 google 所用 ResNet Block 加了三个 3x3 卷积, 二是指多个 bottleneck 内的 3x3 卷积做 multi-grid
- ASPP
  - 加上 batch normalization
  - dilation rate 过大会导致 valid 的 weights 减少 ( 非 pad 0 区域与 filters 区域相交减小 ), 这会导致大 dilation rate 的 filters 退化, 为解决这个问题, 且整合 global context, 使用了 image-level 的 feature. 特别的, 在模型最后的 feature map 采用 global average pooling, 然后送进 256 个 1x1 卷积 BN 中, 然后双线性插值到需要的维度.
  - 最终 ASPP 有: 一个 1x1, 三个 3x3 dilation rate { 6,12,18 } ( 缩小 16 倍时 ), 以及 image-level feature, 最终 concat 在一起做 1x1 卷积. 其中卷积都是 256 output channel 和 BN.

___原文提及主要性能提升来自于 Batch Normalization 的引入及 COCO 预训练. 实际本人尝试在小数据集上从 16s 到 8s 冻结 Batch Normalization 做 finetune, 最终效果并没有提升, 可能真的要在大量数据下才能得到较好的 Batch Normalization 参数做初始化吧___  


<br>
<br>
### Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
----------------------------------------
#### Introduction
融合了 Encoder-Decoder 和 SPP, 加上 depthwise 卷积的大量使用
#### Methods
1. Atrous Convolution 的 Encoder-Decoder  
  * Encoder 阶段改进 Xception + DeepLabv3, 主要改进点为 _atrous separable convolution_, 其实就是都大量使用 depthwise 卷积替换 ASPP 等卷积结构, 可以显著的降低计算复杂度并保持相似或更好的性能  
  * Decoder 阶段在 DeepLabv3 输出端上采样 ( output_stride = 16 下 4 倍 ), 然后 concat 一个经过 1x1 降维后的网络低层特征, 再做 3x3 卷积, 然后上采样到原图大小. 这里也可以采用 depthwise 卷积提升效率.
2. 修改 Xception  
  * 除输入流外加入了更多的层数  
  * 去掉 max-pooling, 以 stride depthwise 卷积替代  
  * 所有 3x3 depthwise 后加 batchnorm 和 ReLU ___( 这个 ReLU 效果存疑 )___

#### Experiments
1. 关于 Decoder 的设计实验  
  * 这部分作者实验发现 Decoder 引入 before striding 的同分辨率 feature map 然后做 1x1 卷积压缩到 48 channel 再进行 concat 效果最好, 不过 mIoU 差距都比较小, 而且 64 channel 效果更差可以看出该选择可能与 Encoder 输出通道比例有关联, 玄学成分多些  
  * 还有就是 concat 后两个 3x3 256 卷积性能最好, 而且只做一个 skip-connection 会更高效
2. 关于 Network Backbone
  * 这里比较重要的一点就是 Decoder 的加入会带来 1%~2% 的性能提升, 这点在训练和测试相同 output stride 的情况下会比较明显, 不同的情况下比较不明显. ___( 这里总体计算量会增加几十 B, 个人认为在轻量级网络下不划算, 还是希望能有不会明显增加计算量的 decoder ) ___
  * 另外, 作者使用 Xception 实验时发现 multi-grid 不会提升性能, 于是没有使用.................
  * 关于 pretrain, 在 COCO 上 pretrain 会带来大约 2% 的提升, 在 JFT 上 pretrain backbone 会再额外带来大约 1% 的提升

#### Conclusion
DeepLabv3+ 采用 Encoder-Decoder 结构, 将 DeepLabv3 作为 Encoder, 并引入简单高效的 skip-connection 来恢复边缘. 还有就是改进 Xception 以及采用 atrous separable convolution 降低计算量. 总的来说没什么新鲜的, 都是整合之前的方法然后通过大量实验找到的最优最高效的结构, 或许这就是炼金术吧......








