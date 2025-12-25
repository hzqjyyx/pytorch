# ATen Quantizer 核心功能分析

这两个文件实现了 PyTorch 的量化张量（Quantized Tensor）系统，负责将浮点张量转换为低精度整数表示以及逆向转换。

## 核心概念

**量化器层次结构：**
- `Quantizer` (基类) → `UniformQuantizer` / `NonUniformQuantizer`
- `UniformQuantizer` → `AffineQuantizer`
- `AffineQuantizer` → `PerTensorAffineQuantizer` / `PerChannelAffineQuantizer`
- `PerChannelAffineQuantizer` → `PerChannelAffineFloatQParamsQuantizer`

**仿射量化公式：**
```
量化: Y = clamp(round(X / scale + zero_point), min, max)
反量化: X = (Y - zero_point) * scale
```

## 主要功能模块

### 1. 量化器工厂函数

**make_per_tensor_affine_quantizer** (aten/src/ATen/quantized/Quantizer.cpp:40-46)
- 创建统一 scale/zero_point 的量化器
- 参数: `scale`, `zero_point`, `scalar_type`

**make_per_channel_affine_quantizer** (aten/src/ATen/quantized/Quantizer.cpp:48-74)
- 创建每通道独立参数的量化器
- 根据 zero_points 类型选择：
  - 浮点型 → `PerChannelAffineFloatQParamsQuantizer`
  - 整数型 → `PerChannelAffineQuantizer`

### 2. 量化张量创建

**new_qtensor** (aten/src/ATen/quantized/Quantizer.cpp:107-161)
- 核心量化张量分配函数
- 处理特殊内存布局（sub-byte 类型如 QUInt4x2, QUInt2x4）
- 设备分支：
  - CUDA: `getCUDAHooks().getCUDADeviceAllocator()`
  - CPU: `getCPUAllocator()`
  - Accelerator: `GetAllocator(device.type())`
  - QNNPACK: 使用 `GetDefaultMobileCPUAllocator()`

**get_sub_byte_tensor_size** (aten/src/ATen/quantized/Quantizer.cpp:84-105)
- 计算 sub-byte 量化类型的实际存储大小
- QUInt4x2: 2 元素/字节
- QUInt2x4: 4 元素/字节
- 对最内层维度进行字节对齐

### 3. PerTensorAffineQuantizer 实现

**quantize** (aten/src/ATen/quantized/Quantizer.cpp:163-180)
- 输入检查：必须是 Float 类型
- 创建量化张量，保持内存格式
- 调用 `native::quantize_tensor_per_tensor_affine`

**dequantize** (aten/src/ATen/quantized/Quantizer.cpp:207-215)
- 创建新的 Float 张量
- 调用 `native::dequantize_tensor_per_tensor_affine`

**dequantize_out** (aten/src/ATen/quantized/Quantizer.cpp:193-205)
- 复用已有输出张量（避免分配）
- 要求输出张量连续且为 Float 类型

### 4. PerChannelAffineQuantizer 实现

**quantize** (aten/src/ATen/quantized/Quantizer.cpp:217-230)
- 使用 `scales_`, `zero_points_`, `axis_` 参数
- 调用 `native::quantize_tensor_per_channel_affine`

**应用场景说明** (aten/src/ATen/quantized/Quantizer.h:118-128)
- 主要用于权重的输出通道量化
- 不适合激活值的 per-channel 量化（处理器效率问题）

### 5. PerChannelAffineFloatQParamsQuantizer

**特殊性** (aten/src/ATen/quantized/Quantizer.h:178-191)
- zero_point 使用浮点数
- 量化公式: `Xq = (Xf - zero_point) * inv_scale`
- 优势：0 不需要精确表示时可获得额外精度

**quantize** (aten/src/ATen/quantized/Quantizer.cpp:268-280)
- 调用 `native::quantize_tensor_per_channel_float_qparams`
- 不保证特定内存格式（使用 `expect_contiguous()` 不带参数）

### 6. from_blob 系列函数

**from_blob_quantized_per_tensor_affine** (aten/src/ATen/quantized/Quantizer.cpp:320-382)
- 从已有内存创建量化张量
- 支持自定义 deleter 管理内存生命周期
- 两个重载：带 strides 和不带 strides（自动计算 C-contiguous strides）

**from_blob_quantized_per_channel_affine** (aten/src/ATen/quantized/Quantizer.cpp:384-429)
- 验证 channel 维度与 scales/zero_points 长度匹配
- 自动设置连续布局

### 7. 工具函数

**get_qtensorimpl** (aten/src/ATen/quantized/Quantizer.cpp:76-82)
- 获取量化张量的底层实现指针
- 检查：不支持 autograd，必须是量化类型

**set_quantizer_** (aten/src/ATen/quantized/Quantizer.cpp:316-318)
- 动态修改张量的量化器

**checkPerChannelParamDims** (aten/src/ATen/quantized/Quantizer.cpp:22-29)
- 验证 scales 和 zero_points 都是 1D 且元素数量匹配

### 8. UnknownQuantizer

**用途** (aten/src/ATen/quantized/Quantizer.h:23-27)
- 占位符量化器，用于两阶段量化流程
- 先分配张量，后由内核决定最终量化参数

**实现** (aten/src/ATen/quantized/Quantizer.cpp:431-448)
- 所有操作都触发断言失败
- 仅用于类型占位

## 关键设计细节

**内存管理：**
- 使用 `c10::make_intrusive` 创建量化器（引用计数）
- `intrusive_from_this()` 复用量化器实例

**类型安全：**
- 所有公共函数验证 `isQIntType(dtype)`
- scales 必须是浮点类型
- zero_points 可以是浮点或整数

**性能优化：**
- `expect_contiguous()` 避免不必要的拷贝
- `suggest_memory_format()` 保持原始内存布局
- dequantize_out 复用输出张量

**量化方案枚举：**
- `kPerTensorAffine`: 单一 scale/zero_point
- `kPerChannelAffine`: 每通道独立参数（整数 zero_point）
- `kPerChannelAffineFloatQParams`: 每通道独立参数（浮点 zero_point）

---

**ROCm 相关：** 文件中没有直接的 ROCm 特定代码，设备分支通过 `at::accelerator::isAccelerator` 统一处理加速器设备。

**Backward 相关：** `get_qtensorimpl` 明确检查 `!self.requires_grad()`，量化张量不支持自动求导。
