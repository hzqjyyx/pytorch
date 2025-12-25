# QTensor.cpp 核心功能解析

这个文件实现了 PyTorch 的张量量化/反量化核心操作，将浮点张量转换为低精度整数表示以节省内存和加速计算。

## 主要功能模块

### 1. 动态量化 (quantize_per_tensor_dynamic)
**位置**: lines 16-55

根据输入张量的实际数值范围自动计算量化参数：
- 支持 QInt8/QUInt8/Half 三种目标类型
- 自动计算输入的 min/max 值
- 根据数据类型设置量化范围（QInt8: -128~127, QUInt8: 0~255）
- 调用 `ChooseQuantizationParams` 自动选择 scale 和 zero_point
- 特殊处理：QNNPACK 引擎会忽略 reduce_range 参数

### 2. 静态量化 (quantize_per_tensor)
**位置**: lines 57-64, 66-73

使用预定义的量化参数：
- 接受显式的 scale/zero_point 参数
- 创建 `PerTensorAffineQuantizer` 执行实际量化
- 支持从 Tensor 形式的量化参数中提取标量值

### 3. 批量量化 (quantize_per_tensor_list_cpu)
**位置**: lines 75-89

高效处理多个张量的量化：
- 输入：张量列表 + 对应的 scales/zero_points 数组
- 对每个张量使用其对应的量化参数
- 返回量化后的张量向量

### 4. 逐通道量化 (quantize_per_channel)
**位置**: lines 91-99

为不同通道使用独立的量化参数：
- 典型应用：卷积层的权重，每个输出通道独立量化
- 参数：scales 和 zero_points 是张量，axis 指定通道维度
- 使用 `PerChannelAffineQuantizer` 实现

### 5. 反量化操作
**位置**: lines 101-115

两个实现路径：
- `dequantize_cpu_or_cuda`: 简单转换为 float32 (line 102)
- `dequantize_quantized`: 通过 Quantizer 对象执行完整反量化 (line 106)
- `dequantize_tensors_quantized_cpu`: 批量反量化

### 6. 量化参数查询接口
**位置**: lines 117-145

提供访问已量化张量参数的 API：
- `q_scale_quant` / `q_zero_point_quant`: Per-tensor 参数
- `q_per_channel_scales` / `q_per_channel_zero_points` / `q_per_channel_axis`: Per-channel 参数
- 所有函数都验证量化方案类型并进行 type casting

### 7. 张量构造与存储操作
**位置**: lines 147-182

- `make_per_channel_quantized_tensor_cpu`: 从已量化数据构造张量，使用 memcpy 复制底层数据
- `set_storage_quantized_`: 直接设置量化张量的存储、偏移和形状

### 8. 克隆与比较
**位置**: lines 189-267

**quantized_clone**: 
- 支持 PerTensorAffine 和 PerChannelAffine 两种方案
- 处理内存格式（Contiguous/Preserve）
- 保留量化参数

**equal_quantized_cpu**:
- 验证量化参数相等性（通过 Quantizer::equalTo）
- 比较形状和数据类型
- 使用 memcpy 进行字节级数据比较
- 特殊处理 QUInt4x2/QUInt2x4 的打包格式

### 9. 量化参数优化
**位置**: lines 270-395

**_choose_qparams_per_tensor**: 基础参数选择
- 计算全局 min/max
- 固定量化范围 0-255
- 返回 scale 和 zero_point

**choose_qparams_optimized**: 高级优化（核心算法）
- 使用贪婪搜索最小化量化误差：`L2_norm(x - fake_quant(x))`
- 分 bins 步进调整 min/max 边界
- 迭代比较左移/右移边界的量化损失
- 找到局部最优的量化范围

算法逻辑：
```
1. 初始化 xmin/xmax 为实际数据范围
2. 将范围划分为 n_bins 个区间
3. 循环：
   - 尝试左边界右移（cur_min + stepsize）
   - 尝试右边界左移（cur_max - stepsize）
   - 选择损失更小的方向
   - 保存全局最优值
4. 返回优化后的边界值
```

---

## 其他内容

- **ROCm 相关**: 无（此文件为 CPU 实现）
- **Backward/梯度**: 无（纯前向量化操作）
