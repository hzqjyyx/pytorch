# Copy.cpp/Copy.h 核心功能分析

## 架构概览

这是 ATen 张量复制操作的核心实现，提供了 `copy_()` (in-place) 和 `copy()` (out-of-place) 两个主要接口，通过多层优化路径处理不同设备、数据类型和内存布局的张量复制。

## 主要函数

### `copy_impl()` (line 139-311)
核心分发逻辑，按优先级处理：

1. **FBGEMM 快速路径** (152-195): FP32↔FP16 转换的 SIMD 优化
   - 条件：CPU 设备、连续内存、相同 shape
   - 使用 `fbgemm::FloatToFloat16_simd` / `Float16ToFloat_simd`
   - 并行化阈值：`GRAIN_SIZE`

2. **提前退出检查**：
   - 同一张量 (197)
   - Meta 张量处理 (202-215)
   - 相同数据视图 (257-268)

3. **设备分发** (224-227):
   - 不支持的设备 (如 XLA) → `_copy_from()` 重新分发
   - 支持的设备：CPU, CUDA, HIP, Vulkan, Metal, MPS, XPU

4. **量化处理** (229-242):
   - float → quantized: `quantized_copy_from_float_()`
   - quantized → quantized: 检查 qscheme 和 dtype 一致性
   - quantized → float: 禁止（需显式 dequantize）

5. **专用设备路径**:
   - Vulkan (244-250): `vulkan::ops::copy_()` 或 `vulkan_copy_()`
   - Metal (252-254): `metal_copy_()`
   - MPS (300-304): `mps_copy_()`

6. **转置优化** (295-298):
   - 条件：`copy_transpose_valid()` - CPU、连续输出、转置输入、≥3600 元素
   - 使用分块算法 `copy_same_type_transpose_()`

7. **通用路径** (309):
   - `TensorIterator` 构建 (271-277)
   - `copy_stub` 设备分发

### `copy_same_type_transpose_()` (73-126)
分块转置复制优化：

```
算法流程（BLOCK_SZ = 60/120）:
for R, C in blocks:
  1. 从 src 复制列到 buffer (memcpy)
  2. buffer 原地转置 (swap elements)
  3. 从 buffer 复制行到 dst (memcpy)
```

- 减少 cache miss，提升大矩阵转置性能
- 块大小根据数据类型调整（Byte: 120, 其他: 60）

### `copy_transpose_valid()` (47-57)
转置优化条件检查：
- 输出连续、输入 2D
- 输入步长 `[1, size(0)]` (列优先/转置)
- 相同 dtype、conj/neg 标志
- 最小尺寸 3600 元素

### `copy_()` (352-366)
用户 API，增加语义检查：
- ZeroTensor 不可变性检查 (356-358)
- ZeroTensor 源处理 (359-361)
- Named tensor 名称传播 (353, 364)
- 调用 `copy_impl()`

### `copy()` (320-336)
函数式版本，用于 functionalization：
- 克隆 self 保留步长：`clone_preserve_strides()`
- 特殊处理零存储情况 (329-330): `empty_strided()`
- 调用 `copy_()` 填充数据

### `_foreach_copy()` (338-350)
批量复制实现，遍历调用 `copy()`

### `copy_ignoring_overlaps()` (368-382)
CUDA 专用，禁用内存重叠检查：
- 用于 `CUDA_tensor_apply2` 的写入重叠场景
- 关闭 `set_check_mem_overlap`

## 分发机制

### Copy.h 接口
```cpp
using copy_fn = void (*)(TensorIterator&, bool non_blocking);
DECLARE_DISPATCH(copy_fn, copy_stub)
```

`copy_stub` 根据设备类型分发到：
- CPU: `aten/src/ATen/native/cpu/CopyKernel.cpp`
- CUDA: `aten/src/ATen/native/cuda/Copy.cu`
- 等

### 设备类型选择 (283-292)
优先级：`CUDA > HIP > MPS > XPU > device_type(0)`

## 关键优化点

1. **SIMD 向量化**: FBGEMM 半精度转换
2. **分块算法**: 转置复制的 cache 优化
3. **并行化**: `at::parallel_for` 处理大张量
4. **提前退出**: 多层检查避免无效复制
5. **类型分发宏**: `_AT_DISPATCH_CP_TYPES` 覆盖所有数据类型

## 边界情况处理

- 空张量 (279-281)
- 复数→实数警告 (306-308)
- 广播 shape 检查 (Meta 张量, 203-209)
- 不同设备间复制的非阻塞模式传递

---

**ROCm/HIP**: 设备类型 `kHIP` 支持，优先级与 CUDA 同级

**Backward**: 无梯度逻辑（`copy_` 是叶子操作，梯度在 autograd 层处理）
