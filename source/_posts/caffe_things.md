---
title: Caffe使用问题记录
date: 2018-08-07 12:44:35
tags: [技术经验,问题记录,caffe]
---
以往在使用 caffe 中遇到的部分问题记录。
<!--more-->
#### 缺陷记录
- `Xavier`初始化没有乘上增益 (ReLU应乘根号2, 等等)
- 在matlab上训练得出的模型是col-major,需要将所有矩阵参数转置才能在其他地方用
- 老版本caffe在初次前向时会比较慢, 新版未知
- caffe 初始化数据层时启动线程是 __TEST__ 和 __TRAIN__ 并行进行的, 即使将`test_initialization`设置为`false`也会进行一次__TEST__的数据 prefetch,  同样会进行`Transform`, 所以要注意相关的共享变量.
- BatchNorm 的 eps 默认为 1e-5, 这个数值是 切实 会对结果产生一定影响的, 在 absorb 参数时也要注意

#### 过程记录
- 后向根据`top_diff`和前向结果算出各`blob`参数的`diff`, 以及`bottom`的`diff`, 所以分别对`blob`和`bottom`求导
- 传播时记得不同微分层乘上前面的梯度值`top_diff`,后传多个梯度值的话全部加起来
- `setup`是在加载网络时调用的, 加载完后不再调用

#### 错误记录
1. Check failed: data_
  - 为 `blob shape` 错误, 一般是 `reshape` 函数出错, 也可能是网络设计错误导致 `shape` 传过来时负值错误

#### 问题记录
1. caffe模型测试时`batch_norm`层的use_global_stats设为false居然没影响????  错觉
2. 训练过程开始良好, 中途出现后方部分卷积开始死亡(参数值非常低), 然后向前传染, 大部分卷积死亡, 表现为验证集上非常不稳定
   - 推测是ReLU死亡
3. caffe 和 opencv 一起 import 会出错
   - added `-Wl,-Bstatic -lprotobuf -Wl,-Bdynamic` to `LDFLAGS` and removed `protobuf` from `LIBRARIES` ( 参照 https://github.com/BVLC/caffe/issues/1917 )

#### 犯2记录
1. `resize`层或者叫`upsample` `upscale` 层, 若训练时使用的缩放算法不同, 在卷积到比较小的时候(4x4)之类的, 会由于策略差异导致缩放前后误差非差大
2. test 或 upgrade 时 model 和 prototxt 写反
  > [libprotobuf ERROR google/protobuf/text_format.cc:274] Error parsing text-format caffe.NetParameter: 2:1: Invalid control characters encountered in text.  
.....   
*** Check failure stack trace: ***  
已放弃 (核心已转储)
3. 二分类问题 SoftmaxWithLoss 层不要设 ignore_label, ignore_label 是会忽略该 label 的 loss 和 diff 传递, 导致结果会完全倒向另一个 label , 因为 SoftmaxWithLoss 是计算准确率来算 loss 的


#### 常见安装问题
1. 一般常见 protobuf 问题, 因为 Tensorflow 也用 protobuf, 不仅用, 还会自动升级 protobuf, 而 caffe 不怎么支持新版本的 protobuf, 所以如果配置了其他开源库的开发环境之后 caffe 报错了, 基本可以从几个方面检查 protobuf 有没问题.
   - `pip list`, 查看 protobuf 版本, 一般 2.6.1 比较通用, 如果是 3.5 那就换吧. 如果同时使用了 python2.7 和 python 3.5 的话那还要注意 pip 也分 pip2 和 pip3, 安装的库也分别独立. 可以在 `/usr/bin`, `/usr/local/bin`, `/$HOME/.local/bin` 下找到 pip 脚本, 打开就能看到它用的是 python2.7 还是 python3.5. ( _然后出现了下一个问题_ )
   - `protoc --version`, protobuf 依赖的东西, 查看它的版本和 protobuf 的是否一样, 不一样的话可以通过下载相应版本 release, 或者从源码安装 protobuf. 然后在 `/etc/ld.so.conf` 里面添加上一行 `/usr/local/lib`, 然后 `sudo ldconfig` 更新下链接库就行了. ( _然后出现了下一个问题_ )
   - `apt list | grep "protobuf"`, 有时候会有用 `apt-get install` 和 `pip install` 装了两种不同版本的 protobuf 的情况, 这时候可以 `apt` 删除并重新安装 protobuf ( _然后出现了下一个问题_ )
   - `File already exists in database: caffe.proto `, 库链接问题或者版本问题 ( 2.6.1 不好用 ), `pip uninstall protobuf` 删掉 protobuf, 重启, 加 -fPIC 到 configure, 然后 `./configure --disable-shared`, 然后在 protobuf 3.3 版本下 `cd $PROTOBUF_BUILD_DIR/python`, `python setup.py build`, `python setup.py test`, `python setup.py install` ( _然而出现了下一个问题_ )
     - 还可能是 caffe 玄学问题, 总之最简单的就是直接把能用的 caffe 替换过来
   - `make all` 时出现一堆 protobuf 未定义的引用问题. ( _未解, 回溯 2.6.1_ )
   - 2.6.1:
     - `caffe_pb2.py: syntax error`, 注释掉默认 caffe 的 `python/caffe/proto/caffe_pb2.py`, 至于为什么项目 caffe 没有用自己的 `caffe_pb2.py` 而用到默认 caffe, 是因为没有成功 `make pycaffe` ??? 总之应该是版本问题.
     - `File already exists in database: caffe.proto` 依旧存在这个问题, 在 `import caffe` 后 `import cv2` 会发生, 还是需要静态链接 protobuf, 这样可以解决:
       - > linking caffe against libprotobuf.a instead of libprotobuf.so could solve this issue
       - > I changed caffe's Makefile. Specifically, I added -Wl,-Bstatic -lprotobuf -Wl,-Bdynamic to LDFLAGS and removed protobuf from LIBRARIES.
         > I have uploaded my Makefile to gist (https://gist.github.com/tianzhi0549/773c8dbc383c0cb80e7b). You could check it out to see what changes I made (Line 172 and 369).

     - `File "/usr/lib/python2.7/dist-packages/caffe/pycaffe.py", line 13, in <module>
    from ._caffe import Net, SGDSolver, NesterovSolver, AdaGradSolver,
    libcaffe.so.1.0.0: cannot open shared object file: No such file or directory`. 这是 python 又喵了咪了用了默认 release 版 caffe, 删掉 `/usr/lib/python2.7/dist-packages/caffe`, 然后在工程头处 `import sys` 加`sys.path.insert('/home/sad/ENet/caffe-enet/python')` 和 `sys.path.insert('/home/sad/ENet/caffe-enet/python/caffe')` 再 `import caffe `, 问题终于解决!
    
2. `libcudnn.so.5: cannot open shared object file: No such file or directory`, ld 抽风, 需要专门刷新下 cuda 链接路径 :
```
sudo ldconfig /usr/local/cuda-8.0/lib64
```

3. `*** SIGSEGV (@0x100000049) received by PID 703 (TID 0x7f52cbb1c9c0) from PID 73; stack trace: ***` 或者 `Segmentation fault (core dumped)`, 可能是 python 层的使用出了问题
4. 段错误, `import caffe` 退出后错误, 有可能是用了 opencv contrib 的 `LIBRARY`, 在 `Makefile` 里删掉 `opencv_videoc` 什么的...