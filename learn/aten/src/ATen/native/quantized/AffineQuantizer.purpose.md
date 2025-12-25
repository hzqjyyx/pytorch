# AffineQuantizer 文件功能分析

## 核心功能

这两个文件实现了 PyTorch 的**仿射量化(Affine Quantization)**机制，负责在浮点数张量(float tensor)和量化张量(quantized tensor)之间进行转换。

## 主要组件

### 1. 量化操作 (Quantization)

将浮点数张量转换为量化张量，支持三种量化模式：

**Per-tensor affine quantization** (`quantize_tensor_per_tensor_affine` - 109-136行)
- 整个张量使用单一的 scale 和 zero_point
- 公式: `q = round(r/scale + zero_point)`
- 支持 sub-byte 类型(QUInt4x2, QUInt2x4)的特殊处理(128-130行)

**Per-channel affine quantization** (`quantize_tensor_per_channel_affine` - 138-171行)
- 每个通道(channel)使用独立的 scale 和 zero_point
- 需要指定量化的轴(axis)参数
- 适用于卷积层的权重量化

**Per-channel float qparams** (`quantize_tensor_per_channel_float_qparams` - 173-203行)
- 类似 per-channel，但使用浮点型的量化参数
- 支持 sub-byte 类型

### 2. 反量化操作 (Dequantization)

将量化张量转换回浮点数张量，对应三种反量化模式：

- `dequantize_tensor_per_tensor_affine` (205-228行)
- `dequantize_tensor_per_channel_affine` (230-262行)
- `dequantize_tensor_per_channel_float_qparams` (264-292行)

公式: `r = (q - zero_point) * scale`

### 3. 调度机制 (Dispatch Stubs)

**头文件定义** (AffineQuantizer.h 87-113行)
- 使用 `DECLARE_DISPATCH` 声明调度桩
- 定义函数指针类型(如 `quantize_tensor_per_tensor_affine_fn`)

**实现文件定义** (AffineQuantizer.cpp 6-13行)
- 使用 `DEFINE_DISPATCH` 实现调度桩
- 支持多设备后端(CPU, CUDA, PrivateUse1等)

调度机制允许根据设备类型(`rtensor.device().type()`)自动选择对应的实现。

### 4. 参数验证

匿名命名空间(15-107行)提供完整的参数检查：

- `checkRoundingMode`: 检查浮点数舍入模式(当前已禁用)
- `checkFloatTensor`: 验证输入是否为 float 类型
- `checkQuantizedTensor<T>`: 验证是否为正确类型的量化张量
- `checkZeroPoint<T>`: 验证 zero_point 在数据类型范围内
- `checkZeroPoints<T>`: 批量验证 zero_points 张量
- `checkSameDevice`: 确保张量在同一设备上
- `checkSameSize`: 确保形状相同
- `checkPerChannelParamsSize`: 验证 per-channel 参数长度与通道数匹配

## 设计模式

**策略模式**: 通过 dispatch stub 根据设备类型选择不同实现
**模板编程**: 使用 `AT_DISPATCH_QINT_*` 宏处理多种量化类型
**防御式编程**: 大量的 `TORCH_CHECK` 进行前置条件验证

## 支持的量化类型

- 标准类型: QInt8, QUInt8, QInt32
- Sub-byte 类型: QUInt4x2, QUInt2x4

---

**ROCm/Backward 相关**:
- CUDA 和 PrivateUse1 设备的 zero_point 检查延迟到设备函数中执行(153-156, 244-247行)
- Sub-byte 量化使用专门的 stub 处理
