这两个文件实现了**FP8矩阵乘法的行级缩放(Rowwise Scaled Matrix Multiplication)**功能，主要用于高效的低精度深度学习计算。

## 主要功能

### 核心操作
实现 `f8f8bf16_rowwise` 函数，执行以下计算：
```
out[i,j] = (XQ[i,:] * x_scale[i]) @ (WQ[:,j] * w_scale[j]) + bias[j]
```
- **输入**：FP8格式的矩阵 `XQ`（激活值）和 `WQ`（权重）
- **缩放**：每行/每列有独立的FP32缩放因子
- **输出**：BFloat16格式

## 文件结构

### RowwiseScaledMM.h
定义公共API接口：
```cpp
void f8f8bf16_rowwise(
    at::Tensor XQ,      // FP8 输入矩阵
    at::Tensor WQ,      // FP8 权重矩阵  
    at::Tensor x_scale, // FP32 行缩放因子
    at::Tensor w_scale, // FP32 列缩放因子
    std::optional<at::Tensor> bias,  // 可选的BF16偏置
    bool use_fast_accum,             // 是否使用快速累加
    at::Tensor& out     // BF16 输出
);
```

### RowwiseScaledMM.cu
包含完整实现，支持多种硬件架构：

#### 1. **SM90架构实现** (lines 119-298)
- 使用CUTLASS 3.x的现代API
- 支持可配置的tile大小（64x128x128 或 128x128x128）
- 使用Epilogue Visitor Tree (EVT)实现融合操作
- 调度策略根据tile大小和fast_accum选择：
  - `KernelTmaWarpSpecialized`
  - `KernelTmaWarpSpecializedCooperative`  
  - `KernelTmaWarpSpecializedFP8FastAccum`

#### 2. **SM100架构实现** (lines 310-478)
- 针对最新Blackwell架构优化
- 使用自动调度策略 `EpilogueScheduleAuto`

#### 3. **SM89架构实现** (lines 487-695)
- 使用CUTLASS 2.x的传统API
- 固定的threadblock配置（64x128x64）
- StreamK线程块调度

#### 4. **智能调度系统** (lines 698-967)
多层分发机制，根据运行时条件选择最优kernel：

```cpp
dispatch_fp8_rowwise_kernel_on_bias_dtype()
  └─> dispatch_fp8_rowwise_kernel_on_input_dtypes()  // 检查输入是E4M3还是E5M2
       └─> dispatch_fp8_rowwise_kernel_on_fast_accum()  // 选择累加模式
            └─> dispatch_fp8_rowwise_kernel_on_sm()     // 选择架构
                 └─> dispatch_fp8_rowwise_kernel_on_cluster_size_and_transpose()
                      └─> dispatch_fp8_rowwise_kernel_on_tile_size()
```

## 关键优化技术

1. **Tile大小自适应** (lines 721-747)
   - 根据SM数量避免wave量化
   - 小矩阵用小tile减少padding浪费

2. **矩阵转置处理** (lines 757-782)
   - 对于瘦矩阵自动转置以提高性能
   - 特殊处理64/192这样的奇数倍形状

3. **Swizzle优化** (lines 858-868)
   - 大矩阵(4096+)使用swizzle=8提高缓存命中率

4. **内存Padding** (lines 132-138, 323-328)
   - x_scale padding到256倍数，解决性能问题

5. **SM Carveout** (lines 265-269)
   - 预留SM给NCCL等后台操作

## 输入约束检查 (lines 969-1018)

严格的张量验证：
- 数据类型：`XQ`支持E4M3/E5M2，`WQ`必须E4M3
- 维度匹配：`scale_a`为[M,1]，`scale_b`为[1,N]
- 内存布局：要求连续的最内层维度
- Stride要求：确保正确的内存访问模式

## 应用场景

- **LLM推理**：量化Transformer模型的线性层
- **训练混合精度**：FP8前向传播 + BF16梯度
- **高吞吐服务**：减少带宽和计算成本

这是PyTorch中支持FP8计算的核心底层实现之一，体现了现代GPU编程中性能优化的复杂性。
