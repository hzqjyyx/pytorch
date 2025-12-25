## Group Normalization 实现分析

**group_norm.h** 定义了两个函数指针类型用于分发机制：
- `forward_fn`: 前向计算的函数签名
- `backward_fn`: 反向梯度计算的函数签名
- 使用 `DECLARE_DISPATCH` 宏声明 `GroupNormKernel` 和 `GroupNormBackwardKernel` 作为分发桩

**group_norm.cpp** 包含以下主要函数：

**check_group_norm_inputs** (第28-59行)
- 验证 `num_groups > 0`
- 验证通道数 `C` 能被 `num_groups` 整除
- 验证权重和偏置的形状必须为 `[C]` 的一维张量

**native_group_norm** (第61-102行)
- 核心前向实现，处理可选的 gamma（权重）和 beta（偏置）参数
- 检查是否需要混合精度处理
- 创建输出张量 Y 和统计张量 mean、rstd（reciprocal standard deviation）
- 调用设备相关的 `GroupNormKernel` 完成实际计算

**group_norm** (第176-207行)
- 公共API，接收输入张量和规范化参数
- 计算张量形状参数：N（batch size）、C（channels）、HxW（空间维度乘积）
- 确保张量内存连续性，处理权重/偏置的连续化
- 调用 `native_group_norm_symint` 进行符号化计算

**math_group_norm** (第213-255行)
- 参考实现，基于 batch normalization 的数学等价形式
- 将输入重塑为 `[1, N*group, -1]`，调用 `native_batch_norm`
- 应用 affine 变换（权重乘法和偏置加法）
- 将均值和方差转换回原始数据类型并重塑为 `[N, group]`

---

### 主要功能总结

- **前向通道分组归一化**：独立对每个group计算均值和标准差进行归一化
- **输入验证**：确保group数量有效且整除通道数
- **混合精度支持**：处理权重/偏置与输入的数据类型不匹配情况
- **设备独立的分发机制**：使用 `GroupNormKernel` 分发到CPU/CUDA/其他后端
- **符号化形状支持**：通过 `sym_size` 和 `sym_numel` 支持动态形状
- **Affine变换**：可选的学习参数（gamma和beta）用于缩放和平移
