# Loss.cu 主要功能分析

这个文件实现了两类损失函数的 CUDA 加速版本：

## 1. Binary Cross Entropy (BCE) Loss

**前向传播** (binary_cross_entropy_out_cuda):
- 计算公式: `loss = -[target * log(input) + (1-target) * log(1-input)]`
- 使用 `TensorIterator` + `gpu_kernel` 实现逐元素并行计算
- 关键数值稳定性处理:
  - 断言输入在 [0,1] 范围内
  - 对数值 clamp 到 -100 以上，防止 log(0) 导致的数值问题
- 支持可选的 weight 张量逐元素加权
- 支持三种 reduction 模式: None/Mean/Sum

**技术细节**:
- 支持 FP16/BFloat16/FP32/FP64 数据类型
- 使用 `GPU_LAMBDA` 在设备端执行计算
- epsilon (1e-12) 用于反向传播时防止除零

## 2. Negative Log Likelihood (NLL) Loss

NLL Loss 针对分类任务，根据输入维度和 reduction 模式分为三个不同的 kernel:

**Kernel 1: nll_loss_forward_no_reduce_cuda_kernel** (2D 输入, 无 reduction)
- 处理批量数据 (batch_size, n_classes)
- 每个线程处理一个样本
- 计算: `output[i] = -weight[target[i]] * input[i][target[i]]`
- 支持 `ignore_index` 跳过特定类别

**Kernel 2: nll_loss_forward_reduce_cuda_kernel_1d** (1D 输入)
- 单线程执行 (<<<1,1>>>)
- 处理单个样本的分类
- 支持 size_average: 除以 weight 进行归一化

**Kernel 3: nll_loss_forward_reduce_cuda_kernel_2d** (2D 输入 + reduction)
- 使用 32 个线程 (NLL_LOSS_THREADS) 并行规约
- 使用共享内存 `sh_inputs` 和 `acc_weight` 累积部分和
- 线程 0 最后汇总所有线程的结果
- 累积类型使用 `accscalar_t` (FP16→FP32) 提高精度

**通用特性**:
- 类别索引越界检查: `CHECK_INDEX_IN_CLASS` 宏
- 支持 Byte (uint8) 和 Long (int64) 两种索引类型
- 边界情况处理:
  - 空张量返回 NaN (Mean) 或 0 (Sum)
  - 单个元素被 ignore 时返回 0
- 支持可选的 per-class weight

**数值稳定性**:
- 使用累积类型防止精度损失
- 对 size_average 时的零除进行显式 NaN 处理

---

**Backward 相关内容**:
- `binary_cross_entropy_backward_out_kernel`: BCE 梯度计算，公式含 `(input-target) / ((1-input)*input)`
- `nll_loss_backward_*_cuda_kernel_*`: 对应三种前向 kernel 的梯度版本，将梯度回传到对应的 input[target] 位置

**ROCm 兼容性**: 未见 ROCm 特定代码，使用标准 CUDA API
