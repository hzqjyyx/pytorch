# SoftMax.cu 主要功能分析

这个文件实现了 CUDA 上的 SoftMax 和 LogSoftMax 前向/反向传播操作，针对不同的输入形状和大小采用了多种优化策略。

## 核心计算模板

文件定义了四个 Epilogue 模板结构体，封装最终的计算逻辑：

- **LogSoftMaxForwardEpilogue** (lines 39-50): 实现 `input - max_input - log(sum)`
- **SoftMaxForwardEpilogue** (lines 64-76): 实现 `exp(input - max_input) / sum`
- **LogSoftMaxBackwardEpilogue** (lines 52-62): LogSoftMax 梯度计算
- **SoftMaxBackwardEpilogue** (lines 78-90): SoftMax 梯度计算

## 三种内核实现策略

### 1. 常规内核 (Regular Kernel)

**适用场景**: `inner_size == 1` 时（dim 是最后一个维度）

**三种变体** (按性能优化程度递增):

a) **基础版本 `cunn_SoftMaxForward`** (lines 667-701)
   - 使用 ILP (Instruction Level Parallelism) 向量化读取
   - 两次 reduction: 找 max，计算 sum(exp(x - max))
   - 根据输入/输出对齐情况选择向量化或标量写入

b) **寄存器优化版本 `cunn_SoftMaxForwardReg`** (lines 703-756)
   - **触发条件**: `potential_reg_cnt < 10` (lines 990-1032)
   - 将数据缓存在寄存器数组中，避免重复访问全局内存
   - 通过模板参数 `reg_cnt` 静态展开循环

c) **共享内存优化版本 `cunn_SoftMaxForwardSmem`** (lines 758-828)
   - **触发条件** (lines 985-988):
     - `dim_size < max_elements_per_smem`
     - 输入/输出指针 16 字节对齐
     - `dim_size` 能被 ILP 整除
   - 第一遍：加载到共享内存并计算 max
   - 第二遍：从共享内存读取计算 sum
   - 第三遍：从共享内存读取并应用 epilogue

**决策逻辑** (lines 990-1040):
```
if (potential_reg_cnt < 10) → 使用寄存器版本
else if (can_use_smem) → 使用共享内存版本
else → 使用基础版本
```

### 2. 空间内核 (Spatial Kernel)

**适用场景**: `inner_size > 1` 时（lines 1085-1114）

**实现**: `cunn_SpatialSoftMaxForward` (lines 254-308)
- 使用 2D grid: x 轴并行 outer_size，y 轴并行 inner_size
- 2D block: threadIdx.x 并行 dim，threadIdx.y 分组处理不同 inner slice
- `spatialBlockReduceX` (lines 230-252): 只在 threadIdx.y 相同的线程间 reduce

**启动配置**:
- Block size: `SpatialSoftMax_getBlockSize()` (lines 120-131) 根据 dim_size 和 inner_size 平衡
- Grid size: `SpatialSoftMax_getGridSize()` (lines 104-116) 优先填充 y 轴（inner），再填充 x 轴（outer）

### 3. 持久化 SoftMax (Persistent Softmax)

**应用场景**: Masked SoftMax (lines 1299-1391)

**触发条件** (lines 1328-1334):
- `softmax_elements <= 1024`
- `softmax_elements * element_size <= 4096`
- `mask.is_contiguous()`
- `dim == input.dim() - 1`

使用 `dispatch_softmax_forward` (来自 PersistentSoftmax.cuh)，支持：
- Transformer 特殊优化：输入 [B, H, T, T]，mask [B, T] (lines 1350-1370)
- 通用 masked softmax (lines 1372-1389)

## 关键优化技术

### 向量化内存访问

- **ILP (Instruction Level Parallelism)**: 使用 `aligned_vector<T, ILP>` 一次读取多个元素
- **对齐检测** (lines 679-680, 847-849): 计算 shift 决定是否可以向量化
- **WriteFpropResultsVectorized** (lines 505-556): 处理非对齐头部，向量化主体，标量化尾部

### 数值稳定性

所有实现都使用 "max trick"：
1. 找到输入的最大值 `max_k`
2. 计算 `sum = Σ exp(x - max_k)`
3. 输出 `exp(x - max_k) / sum` 或 `x - max_k - log(sum)`

避免了直接计算 `exp(x)` 可能导致的溢出。

### Warp-level Reduction

`blockReduceWarp` (lines 440-450) 使用 warp shuffle 指令加速 block 内 reduction，而不是纯粹依赖共享内存。

### 大张量分块处理

对于小 `dim_size` (<= 1024/2048)，使用持久化内核并分块处理避免超出 grid 限制 (lines 969-977, 1047-1056)。

## API 入口

- `log_softmax_cuda_out` (line 1250): 调用 `host_softmax<LogSoftMaxForwardEpilogue, true>`
- `softmax_cuda_out` (line 1274): 调用 `host_softmax<SoftMaxForwardEpilogue, false>`
- `masked_softmax_cuda` (line 1299): 特殊处理 masked 情况

## 简要说明（被忽略部分）

**ROCm 相关**:
- Line 143-148: HIP 平台的 `cudaOccupancyMaxActiveBlocksPerMultiprocessor` 签名兼容性处理
- Line 414-416: `__syncwarp` 在 ROCm 上禁用

**Backward 相关**:
- `cunn_SoftMaxBackward` / `cunn_SoftMaxBackwardSmem` / `cunn_SpatialSoftMaxBackward`: 反向传播内核
- `host_softmax_backward`: 反向传播调度函数 (lines 1152-1247)
- `masked_softmax_backward_cuda`: Masked SoftMax 反向传播 (lines 1393-1454)
- Backward epilogues 使用 `gradOutput - output * sum` 或 `gradOutput - exp(output) * sum` 计算梯度
