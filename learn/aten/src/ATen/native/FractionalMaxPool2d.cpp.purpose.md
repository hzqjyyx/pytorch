## FractionalMaxPool2d.cpp 主要功能

这个文件实现了二维分数最大池化（Fractional Max Pooling 2D）操作，是一种随机降采样技术。

### 核心概念

与传统最大池化不同，分数池化通过**随机采样**确定池化窗口位置，而非固定步长滑动。关键特点：
- 使用 `randomSamples` 参数为每个通道生成随机序列
- 每个通道有2个随机样本（宽度和高度各一个）
- 通过 `generate_intervals` 函数将输入空间随机划分为区间

### 输入输出处理

**支持的张量格式：**
- 3D: `[C, H, W]` (单样本)
- 4D: `[N, C, H, W]` (批次)

**输出两个张量：**
1. `output`: 池化后的值
2. `indices`: 记录每个输出位置对应输入中的最大值索引

### Meta函数（lines 20-126）

`fractional_max_pool2d` meta函数负责：
- 验证 `pool_size` 和 `output_size` 都是2元素数组
- 检查输出尺寸合法性：`outputH + poolSizeH - 1 <= inputH`
- 根据输入维度（3D/4D）设置输出形状
- 输出张量使用 `kLong` 类型存储索引

### 前向实现（lines 133-220）

**单批次处理** `fractional_max_pool2d_out_single_batch_frame`:
- 并行处理各通道（`parallel_for` over planes）
- 对每个通道：
  - 用随机样本生成宽度和高度的区间序列
  - 遍历输出位置，在对应池化窗口内找最大值
  - 特殊处理：`val > maxVal || std::isnan(val)` 确保NaN优先选中

**批次调度** `fractional_max_pool2d_out_frame`:
- 单批次直接调用单批次函数
- 多批次并行处理各样本

### CPU实现入口（lines 278-339）

`fractional_max_pool2d_out_cpu`:
- 确保输入和随机样本连续存储（`.contiguous()`）
- 使用 `AT_DISPATCH_FLOATING_TYPES_AND2` 支持多种浮点类型（包括BFloat16、Half）
- 空输出时提前返回

### 关键设计细节

1. **维度映射动态调整**：根据输入是3D还是4D，planeDim/heightDim/widthDim 相应递增
2. **随机性来源**：外部传入 `randomSamples` 张量，形状为 `[..., C, 2]`
3. **并行策略**：两级并行 - 批次级和通道级
4. **索引计算**：线性化二维坐标 `h * inputW + w`

---

**简要列出的其他内容：**
- Backward meta函数（lines 81-126）：验证梯度输出和索引形状，设置梯度输入形状
- Backward实现（lines 223-274, 341-393）：通过保存的索引将输出梯度累加回输入位置
- ROCm相关：文件中未包含ROCm特定代码，仅CPU实现
