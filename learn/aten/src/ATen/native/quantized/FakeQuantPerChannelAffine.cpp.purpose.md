# FakeQuantPerChannelAffine.cpp 主要功能分析

## 核心功能

该文件实现了**按通道仿真量化（Per-Channel Fake Quantization）** 机制，用于神经网络量化过程中的前向传播。

## 关键函数

### 1. `fake_quantize_per_channel_affine()` (32-42行)
- 对输入张量进行按通道仿真量化
- 调用带掩码版本获取量化结果，返回第一个输出（量化后的张量）

### 2. `fake_quantize_per_channel_affine_cachemask()` (44-107行)
- 核心前向实现
- **输入验证**：
  - scale 必须为 Float 类型，1维张量
  - zero_point 可为 Int/Float/Half 类型，1维张量
  - scale 和 zero_point 尺寸相同，且等于指定轴的大小
  - 检查量化范围有效性 (quant_min ≤ quant_max)

- **处理流程**：
  - 创建预期形状（除量化轴外为1，轴方向为通道数）
  - 使用 TensorIterator 重塑 scale 和 zero_point 以匹配输入形状
  - 构建两个迭代器分别计算量化输出和掩码
  - 调用 dispatch stub 执行具体计算

### 3. `fake_quantize_per_channel_affine_cachemask_backward()` (118-131行)
- 反向传播梯度计算（忽略）

### 4. `_fake_quantize_learnable_per_channel_affine()` (141-152行)
- 可学习参数版本的仿真量化
- 先对 zero_point 进行四舍五入和限幅处理
- 调用标准量化函数

### 5. `_fake_quantize_learnable_per_channel_affine_backward()` (154-258行)
- 可学习参数的梯度计算（忽略）

---

## 核心特性总结

- **按通道量化**：为张量的不同通道使用不同的 scale 和 zero_point
- **掩码生成**：记录超出量化范围的位置，用于梯度计算
- **形状自适应**：通过 unsafe_view 和 expand 自动调整参数维度
- **Dispatch 机制**：支持 CPU/CUDA 后端切换
- **梯度支持**：配置有学习参数版本的反向传播
