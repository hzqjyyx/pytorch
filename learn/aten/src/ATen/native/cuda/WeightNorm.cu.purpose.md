# WeightNorm.cu 主要功能分析

这个文件实现了 Weight Normalization 的 CUDA 加速版本。Weight Normalization 是一种神经网络权重重参数化技术，将权重向量 w 分解为方向 v 和大小 g：**w = g * (v / ||v||)**

## 核心实现策略

文件针对两种情况提供了优化的 CUDA kernel：

### 1. First Dim 归一化 (dim=0)
- **处理对象**：沿着第 0 维的每一行（slowest dimension）
- **线程组织**：每个 block 处理一行，block size = 256
- **kernel**: `weight_norm_fwd_first_dim_kernel`

**计算流程** (aten/src/ATen/native/cuda/WeightNorm.cu:102-150)：
1. 每个线程计算部分元素的平方和：`thread_sum += v[i]²`
2. Block 内归约求和得到 `||v||²`
3. 计算范数：`norm = sqrt(||v||²)`
4. 每个线程写出归一化结果：`w[i] = g * v[i] / norm`

### 2. Last Dim 归一化 (dim=ndims-1)
- **处理对象**：沿着最后一维的每一列（fastest dimension）
- **线程组织**：2D block (TILE_W=16, TILE_H=64)，确保 half-precision 访问的完全合并
- **kernel**: `weight_norm_fwd_last_dim_kernel`

**计算流程** (aten/src/ATen/native/cuda/WeightNorm.cu:155-211)：
1. 2D block 中每列线程协作计算该列的 `||v||²`
2. threadIdx.y==0 的线程计算范数并存入 shared memory
3. 同步后，所有线程读取对应列的 `1/norm` 并写出结果

## 关键优化技术

**Block 内归约** (aten/src/ATen/native/cuda/WeightNorm.cu:51-97)：
- `reduce_block_into_lanes` 函数实现高效的 block 内归约
- 使用 shared memory + warp shuffle 指令
- 前半部分用 `__syncthreads()` 归约到 warp 级别
- warp 内使用 `WARP_SHFL_DOWN` 无需同步完成最终归约

**内存访问优化**：
- First dim: 合并访问连续的行元素
- Last dim: 16-wide tile 保证 half-precision 的完全合并访问（16 字节对齐）

**精度处理**：
- 使用 `AccumulateType`：Half/BFloat16 时用 Float 累加，避免精度损失
- 累加类型 `accscalar_t` 用于中间计算和 norms 存储

## 主机端入口函数

**weight_norm_cuda** (aten/src/ATen/native/cuda/WeightNorm.cu:337-426)：
- 根据 `dim` 参数选择 first_dim 或 last_dim kernel
- 计算 grid/block 配置并启动 kernel
- 返回 `(w, norms)` 供后续使用

---

**其他内容**：
- ROCm 相关：条件编译 `#if !defined(USE_ROCM)` 禁用部分 pragma unroll
- Backward kernels：`weight_norm_bwd_first_dim_kernel` 和 `weight_norm_bwd_last_dim_kernel` 实现梯度反向传播，计算 `grad_v` 和 `grad_g`
- Backward 入口：`weight_norm_backward_cuda` 函数处理反向传播逻辑
