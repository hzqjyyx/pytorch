这个文件实现了 3D 平均池化（Average Pooling 3D）的 CUDA 前向和反向传播内核。

## 核心前向传播实现

### 通用版本内核 (avg_pool3d_cuda_update_output)

**aten/src/ATen/native/cuda/AveragePool3d.cu:36-99**

每个 CUDA 线程负责计算输出的一个元素：

1. **索引计算**：通过 blockIdx/threadIdx 确定输出位置 (oCol, oRow, oFrame, slice)
2. **池化窗口确定**：
   - 起始位置：`tstart = oFrame * dT - padT` (时间维度)，`hstart = oRow * dH - padH` (高度)，`wstart = oCol * dW - padW` (宽度)
   - 结束位置：`tend = min(tstart + kT, input.size(1) + padT)` 等
   - 裁剪到有效范围：`tstart = max(tstart, 0)` 等

3. **归一化因子计算**：
   - 如果指定 `divisor_override`：使用该值
   - 如果 `count_include_pad=true`：使用整个池化窗口大小（包括 padding 部分）
   - 否则：只计算有效元素数量

4. **求和与平均**：三重循环遍历池化窗口内所有元素求和，最后除以归一化因子

### 优化版本内核（模板特化）

**aten/src/ATen/native/cuda/AveragePool3d.cu:104-168**

将最内层循环的宽度 `kW` 作为模板参数，编译器可以更好地优化。在主函数中通过 switch-case 为常见的 kW 值（1-7）调用特化版本。

## 主前向函数

**aten/src/ATen/native/cuda/AveragePool3d.cu:347-441**

```cpp
TORCH_IMPL_FUNC(avg_pool3d_out_cuda)
```

1. **参数提取**：从 IntArrayRef 中提取 kernel_size、stride、padding 参数
2. **维度处理**：如果输入是 5D (batch, channel, time, height, width)，reshape 成 4D (batch*channel, time, height, width) 统一处理
3. **网格配置**：
   - Block: `dim3(32, 8)` 
   - Grid: 根据输出大小计算，Z 维度最大 65535，超过则分批处理
4. **内核分派**：
   - kW ∈ [1,7]：调用模板特化版本
   - 其他：调用通用版本

## 关键设计细节

### 大规模数据处理
使用 `offsetZ` 机制处理超过 65535 的 Z 维度：
```cpp
while (totalZ > 0) {
    dim3 grid(..., totalZ > 65535 ? 65535 : totalZ);
    // 启动内核，传入 offsetZ
    totalZ -= 65535;
    offsetZ += 65535;
}
```

### 边界处理
池化窗口可能超出输入边界，通过 padding 参数控制：
- 先计算包含 padding 的范围
- 再裁剪到实际输入范围
- 空窗口直接输出 0

### 类型精度
使用 `accscalar_t = acc_type<scalar_t, true>` 进行累加，避免精度损失（如 half 类型累加时使用 float）。

---

**Backward 相关（简述）**：
- `avg_pool3d_single_backward_out_frame_stride1`: stride=1 且无 padding 的优化版本反向传播
- `avg_pool3d_cuda_update_grad_input_atomic`: 池化窗口重叠时使用原子操作累加梯度
- `avg_pool3d_cuda_update_grad_input`: 窗口不重叠时的反向传播
- `avg_pool3d_backward_out_cuda`: 主反向函数，根据 stride 和 kernel 参数选择合适的反向内核
