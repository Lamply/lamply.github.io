---
title: 算法加速的几种方法
date: 2019-10-16 1:00:00
tags: [技术经验]
---
这里整理一下常见的算法加速方法。一般来说，对于高计算量的算法，加速是必然要考虑的事情。理想情况下，在编写算法时就应该考虑让算法便于优化，下面就针对性谈谈。
<!--more-->

#### 1. 多层循环
图像处理算法大多需要遍历图像，如果遍历操作较为复杂，这部分的速度就会很慢。所以最好就是能保证最里层循环只有__连续内存__的__简单运算__。如果能够展开来不用 for 循环有时候也会些优化。  

#### 2. SIMD
意思就是单指令多数据流，允许一次处理多个数据，比如有 128bit 寄存器使用 SIMD 指令则可以一次处理 4 个 32bit 数据。前面提到的多层循环内连续内存的简单运算就可以通过 SIMD 进行大幅优化。SIMD 的使用需要特定的数据结构，函数可以通过查 Intrinsics 或者直接翻头文件找到。因为是处理器相关，需要看具体处理器是否支持，以及具体实现，比如 Intel 的 SSE，ARM 的 Neon。知道了用法替换起来就很方便了：  
```C++
// neon 下的矩阵乘法实现
void gemm_neon(const float *lhs, const float *rhs, float *out_mat, int rhs_rows, int rhs_cols)
{
    for (int j = 0; j < rhs_rows; ++j)
    {
        float *data1 = (float*)lhs;
        float *data2 = (float*)(rhs + j*rhs_cols);
        float32x4_t production = vdupq_n_f32(0.f);
        int i = 0;
        for (i = 0; i < rhs_cols-3; i+=4)
        {
            float32x4_t lfactor = vld1q_f32(data1);
            float32x4_t rfactor = vld1q_f32(data2);
            data1 += 4; data2 += 4;
            production = vmlaq_f32(production, lfactor, rfactor);
        }
        float sum = 0.0f;
        for (;  i< rhs_cols; ++i)
        {
            sum += (*data1)*(*data2);
            data1++; data2++;
        }
        float temp[4];
        vst1q_f32(temp, production);
        sum += (temp[0]+temp[1]+temp[2]+temp[3]);

        out_mat[j] = sum;
    }
}
```

#### 3. 复杂算法重组
有时候算法的各步骤执行次序会对算法效率有很大影响，特别对于内存管理而言，最好集中在循环外处理。算法本身也可能会有冗余的计算，通过一些方法合并计算会带来一些性能优化。

#### 4. 牺牲空间换取速度
最为典型的就是查找表，通过预先计算一部分结果，并保存在一个大数组里，从而使后续算法能够加速运算。

#### 5. 二维数组优化
我们习惯将图像表示为二维数组，这样处理起来也更方便，但是二维数组运算的方便却会导致运算效率的降低，根本原因在于内存的不连续。虽然对于优秀的程序员来说不是什么问题，但我比较想提一提这种方法：  
```C++
float **init_matrix(int rows, int cols)
{
    int i;
    float **m;
    m = (float **)malloc((size_t)(rows * sizeof(float *)));
    m[0] = (float *)malloc((size_t)(rows * cols * sizeof(float)));
    for(i = 1; i < rows; ++i)
    {
        m[i] = m[i-1] + cols;
    }
    return m;
}
void delete_matrix(float **m)
{
    free(m[0]);
    free(m);
}
```

#### 6. 矩阵分解
将空间滤波分解成行列滤波，可以提升速度以及降低容量，这部分在 [Dlib](http://dlib.net/) 中有具体的实现，了解得不是很多。

#### 7. L1 cache
也就是利用缓存来优化密集存取的代码。大体来说减少变量的使用，尽量处理连续的空间。只是如果不是对速度有极致的追求，一般也不用特地去钻就是了。

#### 8. 多线程/多进程优化
也就是 pthread 之类的。只用过一次，不是特别了解。上面几类都不是很容易应用的，对于不打算深究的人来说，最好就是底层采用已经造好的库，例如 BLAS、LAPACK 等。













