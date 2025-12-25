# AdaptiveAveragePooling3d.cpp 主要功能

这个文件实现了3D自适应平均池化操作，允许将任意大小的3D输入降采样到指定的输出尺寸。

## 核心实现

### 前向传播 (adaptive_avg_pool3d_out_frame)

**基本原理：** 对于输出中的每个位置，计算输入中对应区域的平均值。

```cpp
// 对每个输出位置 (ot, oh, ow)
istartT = start_index(ot, osizeT, isizeT);  // 计算输入起始位置
iendT = end_index(ot, osizeT, isizeT);      // 计算输入结束位置
kT = iendT - istartT;                        // 池化窗口大小

// 累加对应输入区域的所有值
sum = 0;
for (it in [0, kT))
  for (ih in [0, kH))
    for (iw in [0, kW))
      sum += input[istartT+it][istartH+ih][istartW+iw];

// 除以窗口大小得到平均值
output[ot][oh][ow] = sum / (kT * kH * kW);
```

**关键特性：**
- 使用 `start_index/end_index` 函数动态计算每个输出位置对应的输入范围
- 支持任意输入尺寸到任意输出尺寸的映射
- 并行化处理：在通道维度(sizeD)和批次维度(n)上使用 `at::parallel_for`

### 输入输出格式

**支持两种张量格式：**
1. **4D张量：** `[C, D, H, W]` - 无批次维度
2. **5D张量：** `[N, C, D, H, W]` - 有批次维度

其中：
- N: batch size
- C: channels (sizeD)
- D/H/W: depth/height/width (isizeT/isizeH/isizeW)

### 特殊优化 (adaptive_avg_pool3d_symint)

当输出尺寸为 `[1, 1, 1]` 时，直接使用 `input.mean({-1, -2, -3})` 计算全局平均：

```cpp
if (output_size[0] == 1 && output_size[1] == 1 && output_size[2] == 1) {
    return input.mean({-1, -2, -3}, keepdim=true);
}
```

对于 ChannelsLast3d 内存格式，还会调整输出步幅以保持内存布局。

### 数据类型支持

通过 `AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, ...)` 支持：
- float32
- float64
- float16 (half)
- bfloat16

### 主要API函数

1. **adaptive_avg_pool3d_out_cpu:** 使用预分配的输出张量
2. **adaptive_avg_pool3d_cpu:** 自动分配输出张量
3. **adaptive_avg_pool3d_symint:** 支持符号整数（用于动态形状），包含 1x1x1 优化

---

## ROCm 相关
- 文件中未包含 ROCm 特定代码（纯CPU实现）

## Backward 相关
- `adaptive_avg_pool3d_backward_out_frame`: 反向传播实现，将输出梯度均匀分配到对应的输入区域
- `adaptive_avg_pool3d_backward_out_cpu` / `adaptive_avg_pool3d_backward_cpu`: 反向传播的公共API
- 梯度计算公式：`gradInput[i] += gradOutput[o] / (kT * kH * kW)` 对所有映射到输出o的输入位置i
