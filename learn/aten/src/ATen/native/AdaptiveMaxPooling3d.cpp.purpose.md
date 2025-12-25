# AdaptiveMaxPooling3d.cpp 主要功能

这个文件实现了3D自适应最大池化操作的CPU版本。

## 核心概念

**自适应池化**：与普通池化不同，不是指定kernel size和stride，而是直接指定输出尺寸。算法会自动计算每个输出位置对应的输入区域。

## 主要实现

### 1. Meta函数 (aten/src/ATen/native/AdaptiveMaxPooling3d.cpp:18-65)

`adaptive_max_pool3d` meta函数负责：
- 输入验证：必须是4D (D×T×H×W) 或5D (B×D×T×H×W) 张量
- 输出形状设置：返回两个张量
  - 输出张量：池化结果
  - 索引张量：记录每个输出值对应的输入位置（用于反向传播）

### 2. 前向实现

核心函数 `adaptive_max_pool3d_single_out_frame` (aten/src/ATen/native/AdaptiveMaxPooling3d.cpp:96-170)：

```
对于每个输出位置 (ot, oh, ow)：
  1. 使用 start_index/end_index 计算对应的输入区域范围
  2. 在该3D区域内找最大值
  3. 记录最大值和其在输入中的索引
```

关键点：
- 使用 `start_index(ot, osizeT, isizeT)` 和 `end_index(ot, osizeT, isizeT)` 计算输入区域边界
- NaN处理：`if ((val > maxval) || std::isnan(val))` 确保NaN会被选中
- 并行化：使用 `at::parallel_for` 在深度维度(D)上并行

### 3. 实际执行入口

`adaptive_max_pool3d_out_cpu` (aten/src/ATen/native/AdaptiveMaxPooling3d.cpp:269-360)：
- 处理4D和5D输入的维度索引调整
- 使用 `AT_DISPATCH_FLOATING_TYPES_AND2` 支持多种数据类型 (float, double, bfloat16, half)
- 计算stride信息并调用核心实现函数

## 数据流示例

假设输入 5×7×9 (D×H×W)，输出 2×3×4：
- 对于输出位置 (0, 1, 2)
- 计算输入T维度范围：start_index(0, 2, 5) = 0, end_index(0, 2, 5) = 2
- 计算输入H维度范围：start_index(1, 3, 7) = 2, end_index(1, 3, 7) = 4  
- 计算输入W维度范围：start_index(2, 4, 9) = 4, end_index(2, 4, 9) = 6
- 在输入的 [0:2, 2:4, 4:6] 区域内找最大值

## 其他内容简述

**Backward相关**：
- `adaptive_max_pool3d_backward` meta函数：验证梯度张量维度
- `adaptive_max_pool3d_backward_single_out_frame`：根据保存的索引将梯度分配回输入位置
- 使用加法累积梯度：`gradInput_p_d[maxp] += gradOutput_p_d[...]`

**ROCm相关**：无（纯CPU实现）
