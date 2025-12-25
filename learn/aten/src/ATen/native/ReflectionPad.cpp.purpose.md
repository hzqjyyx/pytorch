# ReflectionPad.cpp 主要功能

这个文件实现了反射填充（Reflection Padding）操作，用于在张量边缘进行镜像对称的扩展。

## 核心概念

反射填充是一种边界填充策略，通过镜像翻转边缘像素来扩展张量。例如，对于序列 `[1, 2, 3, 4]`，左侧填充2个元素后变为 `[3, 2, 1, 2, 3, 4]`。

## 文件结构

### 1. Meta函数（at::meta命名空间）

负责输出形状推导和参数验证，不执行实际计算：

**reflection_pad1d (27-72行)**
- 处理1D张量填充
- 支持2D输入 `(C, W)` 或3D批次输入 `(N, C, W)`
- 验证：填充大小必须小于对应输入维度（`pad_l < input_w && pad_r < input_w`）
- 计算输出宽度：`output_w = input_w + pad_l + pad_r`

**reflection_pad3d (106-162行)**
- 处理3D张量填充
- 支持4D输入 `(C, D, H, W)` 或5D批次输入 `(N, C, D, H, W)`
- 验证每个维度的填充约束
- 计算输出的深度、高度、宽度

### 2. 实现函数（at::native命名空间）

**reflection_pad2d_out_template (212-267行)**
- 2D填充的核心模板函数
- 处理维度索引调整（批次模式下所有维度索引+1）
- 特殊处理：
  - 量化张量调整大小时不使用 `memory_format` 参数（260-261行）
  - 非量化张量使用 `suggest_memory_format()` 优化内存布局（263行）
- 调用 `reflection_pad2d_kernel(kCPU, ...)` 执行实际计算

**TORCH_IMPL_FUNC宏定义的函数**

- `reflection_pad1d_out_cpu` (312-315行)
- `reflection_pad3d_out_cpu` (371-377行)
- 调用对应的kernel dispatcher进行实际计算

**量化张量支持**
- `reflection_pad1d_out_quantized_cpu` (304-310行)
- `reflection_pad2d_quantized_cpu` (341-348行)
- 验证仅支持 `kPerTensorAffine` 量化方案
- 保持输入的量化参数（scale和zero_point）

### 3. Kernel Dispatchers（394-399行）

```cpp
DEFINE_DISPATCH(reflection_pad1d_kernel);
DEFINE_DISPATCH(reflection_pad2d_kernel);
DEFINE_DISPATCH(reflection_pad3d_kernel);
```

这些dispatcher会根据设备类型（CPU/CUDA）分发到具体的kernel实现。

## 关键验证逻辑

所有维度的填充都必须满足：
- `pad_size < input_size`（防止填充超出可反射的范围）
- `output_size >= 1`（确保输出有效）

## 内存优化

- 使用 `suggest_memory_format()` 推荐的内存布局（channels_last等）
- 量化张量特殊处理避免不必要的格式转换

---

**简要列出的其他功能：**
- Backward函数：`reflection_pad1d_backward`、`reflection_pad2d_backward`、`reflection_pad3d_backward` 用于梯度反向传播
- ROCm支持：通过相同的dispatcher机制支持AMD GPU
