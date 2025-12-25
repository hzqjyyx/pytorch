# ConvolutionMM3d.cpp/h 核心功能

这两个文件实现了 **3D 卷积的 CPU 后端**，采用 **im2col + GEMM (矩阵乘法)** 的经典实现策略。

## 主要实现思路

### 1. Im2col 转换 (`compute_columns3d`)
将 3D 输入张量转换为列矩阵，使卷积运算可以表示为矩阵乘法：

- **特殊优化**：当 kernel=1x1x1、stride=1、padding=0、groups=1 时，直接将输入 view 为列矩阵
- **通用情况**：调用 `Unfold3dCopyCPU` 将输入展开为 `[batch, in_channels * kD * kH * kW, out_D * out_H * out_W]`
- 支持并行处理，按 batch 维度切分（粒度 `CONV3D_GRAIN_SALT=20`）

### 2. 前向传播 (`slow_conv3d_forward_out_cpu`)

**流程**：
1. **形状检查** (`slow_conv3d_shape_check`)：验证 kernel、stride、padding 合法性，计算输出尺寸
2. **权重转换** (`view_weight_2d`)：将 5D 权重 `[out_ch, in_ch, kD, kH, kW]` 重塑为 2D `[out_ch, in_ch*kD*kH*kW]`
3. **Im2col**：生成 `finput` 列矩阵
4. **偏置初始化**：若存在 bias，将输出初始化为 bias 值
5. **分组卷积 GEMM** (`slow_conv3d_update_output_frame`)：
   - 对每个 batch，计算 `output = weight × finput`
   - 使用 `cpublas::gemm_batched_with_stride` 处理分组卷积
   - 利用 Fortran 列优先转置技巧：`C = AB` ⟺ `Cᵀ = BᵀAᵀ`

**核心计算**：
```
output[batch, out_ch, D, H, W] = 
    weight[out_ch, in_ch*kD*kH*kW] × finput[batch, in_ch*kD*kH*kW, D*H*W] + bias
```

### 3. 分组卷积支持

- Groups 参数通过 `self.size(1) / weight.size(1)` 推断
- GEMM 分批处理：`cpublas::gemm_batched_with_stride` 自动处理分组维度

### 4. 数据类型调度

使用 ATen 的 `AT_DISPATCH_ALL_TYPES_AND2` 宏：
- 前向支持：`float, double, int, long, BFloat16, Half`
- 主要分发点：
  - `compute_columns3d` 中的 `Unfold3dCopyCPU`
  - `slow_conv3d_forward_out_cpu` 中的 GEMM 调用

---

## Backward/ROCm 相关（简略）

### Backward 实现
- **梯度输入**：`slow_conv3d_backward_update_grad_input_frame` 计算 `fgrad_input = weightᵀ × grad_output`，再通过 `Unfold3dAccCPU` 累加回原形状
- **梯度权重**：`slow_conv3d_backward_weight_frame` 计算 `grad_weight = grad_output × finputᵀ`，累加所有 batch
- **梯度偏置**：对 `grad_output` 按 `[0,2,3,4]` 维度求和

### ROCm
- 无 ROCm 特定代码，仅 CPU 实现
