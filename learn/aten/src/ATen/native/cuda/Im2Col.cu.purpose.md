这个文件实现了 CUDA 中的 im2col（image to column）操作，将图像转换为列矩阵格式。

**主要功能：**

- **im2col_out_cuda_template()** - 核心模板函数，执行 im2col 转换：
  - 验证 kernel_size、dilation、padding、stride 都是 2D 的
  - 提取参数并计算输出尺寸
  - 处理 3D 输入（无 batch），自动添加 batch 维度
  - 计算输出高宽：`(input + 2*pad - dilation*(kernel-1) - 1) / stride + 1`
  - 调用底层 CUDA kernel `im2col<scalar_t>()` 逐 batch 处理

- **im2col_out_cuda()** - 输出张量版本，直接调用模板函数

- **im2col_cuda()** - 返回值版本，先创建空输出张量再调用模板函数

- **支持多种数据类型**：浮点、复数、Half、BFloat16、Bool

**典型应用场景：**
- 卷积操作的前处理，将输入张量转为可批量矩阵运算的格式
- 实现高效的卷积计算（通过 GEMM）
