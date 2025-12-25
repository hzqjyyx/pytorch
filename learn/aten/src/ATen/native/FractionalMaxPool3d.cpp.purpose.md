## 主要功能

这个文件实现了 3D 分数最大池化（Fractional Max Pooling）操作，用于在 3D 张量上进行非整数比例的下采样。

## 核心机制

**1. Meta 函数（元信息设置）**
`fractional_max_pool3d` 的 meta 函数（第21-94行）负责：
- 验证输入参数：pool_size 和 output_size 必须是 3 元素数组
- 处理 4D（无批次）和 5D（有批次）张量
- 检查池化窗口大小是否合法：确保 `outputT + poolSizeT - 1 < inputT`
- 分配输出张量：一个存储池化结果，一个存储最大值的索引位置

**2. 前向计算流程**

核心函数 `fractional_max_pool3d_out_single_batch_frame`（第102-169行）：

a. **生成随机间隔序列**（第119-124行）
   - 使用 `randomSamples` 为每个通道的 T/H/W 维度生成间隔序列
   - `generate_intervals` 将输入维度划分为输出维度个数的不等长区间

b. **并行处理每个通道**（第112行）
   - 使用 `at::parallel_for` 并行处理 `numPlanes` 个通道

c. **滑动窗口最大值计算**（第132-166行）
   - 三层嵌套循环遍历输出位置 (t, h, w)
   - 根据序列确定输入起始位置 `inputTStart/inputHStart/inputWStart`
   - 在 `poolSizeT × poolSizeH × poolSizeW` 窗口内找最大值
   - NaN 处理：`if (val > maxVal || std::isnan(val))` 确保 NaN 优先（第154行）
   - 记录最大值和其在输入中的线性索引

**3. 批次处理**（第172-206行）
`fractional_max_pool3d_out_frame` 处理多批次：
- 单批次直接调用单批次函数
- 多批次时并行处理每个批次

**4. CPU 实现入口**（第210-255行）
`fractional_max_pool3d_out_cpu`：
- 调用 `fractional_max_pool_check_shape` 验证形状
- 确保输入和随机样本连续
- 使用 `AT_DISPATCH_FLOATING_TYPES_AND2` 支持 float/double/bfloat16/half 类型

## 关键特性

- **分数池化**：通过随机采样生成不等长区间，实现非整数倍下采样
- **并行化**：通道级和批次级并行处理
- **索引记录**：返回最大值位置，用于反向传播
- **多精度支持**：支持多种浮点类型

---

**ROCm 相关**：无（此文件仅实现 CPU 版本）

**Backward 相关**：
- `fractional_max_pool3d_backward_out_single_batch_frame`（第260-287行）：根据前向保存的索引，将梯度累加回对应位置
- `fractional_max_pool3d_backward_out_cpu_template`（第322-387行）：反向传播模板函数，初始化 gradInput 为零后累加梯度
- 支持与前向相同的数据类型和批次处理策略
