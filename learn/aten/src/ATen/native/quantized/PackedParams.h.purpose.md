这个文件定义了两个基类，用于量化神经网络中的参数打包和操作：

**LinearPackedParamsBase**
- 为量化线性层（linear layer）的打包参数提供接口
- 核心方法包括：
  - `apply()`: 执行线性计算，输入为float32或quantized张量
  - `apply_relu()`: 线性计算后跟ReLU激活
  - `apply_dynamic()`: 动态量化的线性计算（不需预先指定输出scale/zero_point）
  - `apply_dynamic_relu()`: 动态量化线性+ReLU
  - `unpack()`: 解包得到原始权重和偏置
  - `bias()`: 获取偏置项
  - `set_bias()`: 设置偏置项
- 支持输出变体（`*_out`）和融合模式（如`apply_with_input_q_dq_qweight_dq_output_fp32`）

**ConvPackedParamsBase**
- 为量化卷积层的打包参数提供接口
- 核心方法包括：
  - `apply()` / `apply_relu()` / `apply_dynamic()`: 类似线性层的操作
  - `unpack()`: 解包权重和偏置
  - 卷积参数查询：stride、padding、output_padding、dilation、groups、transpose

**主要特点：**
- 继承自 `torch::jit::CustomClassHolder`，支持TorchScript编译
- 纯虚基类，具体实现由子类提供
- 支持多种量化方案（静态量化、动态量化、融合操作）
- 参数打包优化了推理性能（权重预处理）

**核心功能：**
- 定义量化线性/卷积层的统一接口
- 支持不同输出数据类型的计算
- 管理量化参数（scale、zero_point）
- 提供权重的预打包存储格式
