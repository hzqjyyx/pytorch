这个文件定义了三个 CUDA 全局kernel函数，用于实现高效的attention反向传播计算：

**主要功能：**

- **auto-generated文件**：由 `generate_kernels.py` 自动生成，不应手动编辑

- **三个SM架构版本的kernel**：
  - `fmha_cutlassB_f32_notaligned_64x64_k64_dropout_sm50`：针对 SM 5.0-5.x
  - `fmha_cutlassB_f32_notaligned_64x64_k64_dropout_sm70`：针对 SM 7.0-7.4
  - `fmha_cutlassB_f32_notaligned_64x64_k64_dropout_sm75`：针对 SM 7.5-7.x

- **参数配置**：
  - float32 数据类型（f32）
  - 非对齐内存访问（notaligned）
  - k维度=64，tile大小=64x64
  - 启用dropout正则化（dropout标志为true）

- **kernel执行逻辑**：
  - 检查编译时GPU架构是否匹配目标范围
  - 调用 `AttentionBackwardKernel::attention_kernel()` 执行实际计算
  - 支持动态block分配（`advance_to_block()`）
  - 架构不匹配时输出fatal错误信息

- **使用Cutlass库**：基于NVIDIA的Cutlass模板库实现高效的GPU计算
