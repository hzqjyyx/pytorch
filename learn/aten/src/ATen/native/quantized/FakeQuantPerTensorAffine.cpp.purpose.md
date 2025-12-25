## 主要功能

该文件实现了 PyTorch 中**按张量仿真量化（Per-Tensor Affine Fake Quantization）** 的核心操作。

### 核心函数说明

**1. `fake_quantize_per_tensor_affine` (行 31-51)**
- 两个重载版本，均调用内部缓存掩码函数
- 第一个版本：标量 scale/zero_point 参数
- 第二个版本：张量 scale/zero_point 参数（支持学习）
- 返回仿真量化后的张量

**2. `fake_quantize_per_tensor_affine_cachemask` (行 69-90)**
- 执行仿真量化并保存掩码，用于降低反向传播的内存开销
- 创建输出张量 `Y` 和布尔掩码 `mask`
- 通过 `fake_quant_tensor_cachemask_stub` 分发到 CPU/CUDA 实现
- 返回量化张量和掩码元组

**3. `_fake_quantize_per_tensor_affine_cachemask_tensor_qparams` (行 92-110)**
- 支持张量形式的量化参数版本
- 添加 `fake_quant_enabled` 标志控制是否启用量化
- 同样分发到对应设备实现

**4. `fake_quantize_per_tensor_affine_cachemask_backward` (行 121-134)**
- 反向传播：梯度乘以预计算的掩码
- 简化计算，避免重新计算量化边界检查

**5. `_fake_quantize_learnable_per_tensor_affine` (行 148-159)**
- 支持可学习的 scale 和 zero_point
- 提取张量标量值并进行值域转换
- 调用标准仿真量化函数

**6. `_fake_quantize_learnable_per_tensor_affine_backward` (行 161-227)**
- 计算关于 X、scale、zero_point 的梯度
- 根据量化边界分段计算梯度
- 返回三元组：dX、dScale、dZeroPoint

---

- **用途**：模拟整数量化的效果，用于量化感知训练（QAT）
- **关键验证**：quant_min ≤ quant_max，zero_point 在有效范围内
- **分发机制**：使用 `DEFINE_DISPATCH` 支持多设备后端
- **优化**：缓存掩码减少反向传播内存占用（1 byte vs 理想 1 bit）
- **可学习性**：支持 scale 和 zero_point 梯度计算，实现端到端量化训练
