# AdaptiveAveragePooling3d.cu 主要功能

这个文件实现了 3D 自适应平均池化（Adaptive Average Pooling）的 CUDA 加速版本，用于处理 5D 张量（B × D × T × H × W，即 Batch × Depth/Channels × Time/Frames × Height × Width）。

## 核心概念

**自适应池化**：与常规池化不同，它不是指定池化窗口大小，而是直接指定输出尺寸。算法会自动计算每个输出位置对应的输入区域。

## 关键辅助函数

```cuda
__device__ inline int64_t start_index(int64_t a, int64_t b, int64_t c)
__device__ inline int64_t end_index(int64_t a, int64_t b, int64_t c)
```

这两个函数计算输出位置 `a`（在大小为 `b` 的输出中）对应的输入区域范围（输入大小为 `c`）：
- `start_index`: 计算输入起始索引
- `end_index`: 计算输入结束索引

## 前向传播实现

### 1. `adaptiveaveragepool` 核函数

**线程组织**：
- 每个 block 处理一个 T×H×W 平面中的部分输出像素
- `blockIdx.x + offsetZ` 确定处理哪个平面（特征 × 时间帧的组合）
- `gridDim.y` 个 block 协作处理同一平面的 H 维度
- 线程在 W 维度展开

**计算流程**：
```
for 每个输出位置 (ot, oh, ow):
    1. 计算对应的输入范围 [istartT:iendT, istartH:iendH, istartW:iendW]
    2. 遍历这个 3D 输入区域，累加所有值到 sum
    3. 除以区域大小 (kT × kH × kW) 得到平均值
    4. 写入输出
```

### 2. `adaptiveaveragepool_loop` 主机函数

- 使用 32×8 线程块配置
- 处理超过 65535 个平面时分批次启动核函数（CUDA grid 维度限制）
- `offsetZ` 用于追踪已处理的平面数

### 3. `adaptive_avg_pool3d_out_cuda_template` 入口函数

**处理逻辑**：
- 支持 4D（D×T×H×W）和 5D（B×D×T×H×W）输入
- 4D 输入视为 batch 维度为 1 的 5D 张量
- 5D 输入需要先调用 `contiguous()` 确保内存连续
- 计算 `totalZ = B × D × osizeT`（总平面数）
- 使用 `AT_DISPATCH_FLOATING_TYPES_AND2` 支持 float/double/half/bfloat16

**维度映射**：
```
totalZ 个平面 = (B × D) 个特征切片 × osizeT 个时间帧
每个平面对应输出的一个 H×W 2D 图像
```

## 性能设计要点

1. **合并访问**：线程在 W 维度（最内层）展开，确保连续内存访问
2. **工作分配**：`blocksH` 参数根据总平面数动态调整，平衡负载
3. **批处理**：分批处理大量平面，规避 CUDA grid 大小限制
4. **类型精度**：使用 `accscalar_t` 累加类型避免精度损失（如 half → float）

---

**Backward 相关**：
- `adaptiveaveragegradinput`: 非原子版本梯度计算，适用于输入输出尺寸整除的情况
- `atomicadaptiveaveragegradinput`: 使用原子加法处理非整除情况（可能产生竞态）
- 根据 `isizeW%osizeW != 0` 等条件自动选择是否使用原子操作
- 原子版本会触发不确定性警告（`alertNotDeterministic`）
