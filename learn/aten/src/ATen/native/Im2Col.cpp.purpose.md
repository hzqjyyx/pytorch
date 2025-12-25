**Im2Col.cpp 主要功能分析**

这个文件实现了 `im2col`（image to column）操作，这是一个在深度学习中常用的张量变换算法。

**核心功能：**

- **im2col_out_cpu_template** (22-128行)：核心实现函数
  - 将输入张量按卷积参数（kernel size、padding、stride、dilation）重新组织成列矩阵格式
  - 支持 batched 和 non-batched 输入（通过检测 3D/4D 张量自动处理）
  - 计算输出高度和宽度：`(input_size + 2*padding - dilation*(kernel-1) - 1) / stride + 1`
  - 输出形状：`[batch, n_input_plane*kernel_h*kernel_w, output_h*output_w]`

- **im2col_out_cpu** (132-141行)：包装函数
  - 接收已有的输出张量，调用模板函数进行原位修改

- **im2col_cpu** (143-154行)：便利函数
  - 自动创建输出张量，然后调用模板函数

**关键特性：**

- 支持多种数据类型：浮点数、复数、BFloat16、Float16、Bool
- 参数验证：确保 kernel_size、dilation、padding、stride 都是 2 元组
- 张量处理：自动转换为连续内存布局 (contiguous)
- 批处理循环：逐个处理 batch 中的样本

**用途：**

- 作为 CPU 端卷积操作的预处理步骤
- 将局部图像块展平为矩阵，便于使用矩阵乘法实现卷积
