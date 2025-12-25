# AdaptiveAveragePooling.cu 核心功能分析

这个文件实现了2D自适应平均池化的CUDA内核，用于将任意大小的输入特征图池化到指定的输出尺寸。

## 核心算法原理

**索引映射宏**（29-33行）：
- `START_IND(a,b,c)`: 计算输出位置a对应的输入起始索引
- `END_IND(a,b,c)`: 计算输入结束索引
- 这两个宏实现了自适应池化的关键：根据输入输出尺寸比例，动态确定每个输出像素对应的输入区域

**工作原理**：
输出位置 `(oh, ow)` 对应的输入区域为 `[istartH:iendH, istartW:iendW]`，池化窗口大小 `kH × kW` 根据位置自适应变化。

## NCHW布局实现（Contiguous格式）

**前向传播内核** `adaptive_average_pool`（52-106行）：

```
线程组织：
- blockIdx.x: 处理不同的特征平面（batch × channel）
- 每个线程块处理输出的一部分H×W区域

计算流程：
1. 每个线程遍历分配的输出像素(oh, ow)
2. 根据START_IND/END_IND确定对应的输入窗口
3. 累加输入窗口内所有值：sum = Σ input[ih, iw]
4. 除以窗口大小得到平均值：output = sum / (kH * kW)
```

**关键代码段**（90-103行）：
- 使用高精度 `opmath_t` 累加避免精度损失
- 直接遍历输入窗口计算平均值
- 启动配置：blocks(grid_x, blocksH)，threads(32, 8)

## NHWC布局实现（ChannelsLast格式）

**前向传播内核** `adaptive_average_pool_nhwc`（228-318行）：

```
优化策略：
1. 共享内存缓存：减少寄存器使用
2. 线程块组织：
   - block.x (C方向): 处理通道，最多warp_size个线程
   - block.y (W方向): 处理宽度
   - block.z (H方向): 处理高度
3. 网格组织：
   - grid.x: batch × 通道分组
   - grid.y/z: 空间维度分块（每块处理TILE区域）

内存访问模式：
- 按 h→w→c 层次遍历输入，利用缓存局部性
- 通道方向跨步访问以提高并行度
- 使用 kernel_stride_C 控制通道分组
```

**共享内存使用**（239-249行）：
- `out_cached`: 存储临时输出值，大小为 `kernel_size_C × blockDim.x × blockDim.y × blockDim.z`
- 每个线程独占自己的缓存区域，避免同步开销

**计算核心**（287-314行）：
1. 遍历输入窗口 `[istartH:iendH, istartW:iendW]`
2. 按通道分组累加到共享内存
3. 乘以预计算的因子 `factor = 1.0 / ((iendH-istartH) * (iendW-istartW))`
4. 写回全局内存并清零缓存

## 启动配置逻辑

**NHWC模式**（490-525行）：
- 根据GPU硬件限制（maxThreadsPerBlock, maxThreadsDim, maxGridSize）动态计算
- 优先保证 block_y 和 block_z 较大以提高缓存命中率
- block_x 保持在warp_size内以保证合并访问
- 限制网格大小防止超出硬件上限

**共享内存检查**（536-537行）：
确保 `shmem_size ≤ sharedMemPerBlock`，否则断言失败

## 公开API

**前向接口**（766-784行）：
- `adaptive_avg_pool2d_out_cuda`: 写入预分配输出
- `adaptive_avg_pool2d_cuda`: 自动分配输出张量

**入口逻辑**（440-599行）：
1. 检查输入维度（3D或4D）和输出尺寸参数
2. 根据内存格式（ChannelsLast/Contiguous）选择内核
3. 处理空张量的边界情况

---

## Backward 相关简述

- **atomic_adaptive_average_gradinput**（168-220行）：使用原子加法处理梯度累加，避免写冲突
- **adaptive_average_gradinput**（112-161行）：无原子操作版本（当前未使用）
- **adaptive_average_gradinput_nhwc**（326-436行）：NHWC格式反向传播，预计算索引和权重因子到共享内存
- **非确定性警告**（791-808行）：因使用atomicAdd导致结果非确定性

## ROCm 相关

文件中未包含ROCm特定代码，仅使用通用CUDA API（可能在其他编译分支中处理）
