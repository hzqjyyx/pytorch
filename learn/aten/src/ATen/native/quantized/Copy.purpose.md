## 文件功能分析

### Copy.h
头文件声明了一个函数接口：
- `quantized_copy_from_float_()` - 将float类型张量的数据复制到量化张量中

### Copy.cpp
实现了量化张量的复制功能，核心逻辑包括：

**输入验证**：
- 源张量必须是float类型
- 自身和源张量必须同时满足：连续（contiguous）或NHWC格式连续
- 两个张量形状必须相同

**量化过程**：
根据量化方案（qscheme）选择不同的量化方式：
1. **Per-Channel量化**（kPerChannelAffine、kPerChannelAffineFloatQParams、kPerChannelSymmetric）
   - 调用 `quantize_tensor_per_channel_affine()` 
   - 使用per-channel的scale和zero_point进行量化

2. **Per-Tensor量化**（其他情况）
   - 调用 `quantize_tensor_per_tensor_affine()`
   - 使用全局的scale和zero_point进行量化

**关键限制**：
- 不支持非连续的量化子张量赋值（如 `qt1[:, 0] = t2[:, 0]`）

---

### 主要功能总结：
- 将float张量量化并复制到量化张量
- 支持per-channel和per-tensor两种量化方式
- 严格的张量形状和内存布局检查
- 原地操作，返回修改后的量化张量
