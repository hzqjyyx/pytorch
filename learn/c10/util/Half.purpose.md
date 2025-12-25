## 核心功能

这两个文件实现了 PyTorch 的半精度浮点数（FP16）类型 `Half`，提供 16-bit 浮点数与标准 C++ 类型之间的转换和基本操作。

## Half.h 主要内容

### 1. 精度转换函数

**FP16 → FP32 (位表示)** `fp16_ieee_to_fp32_bits`
- 纯整数位运算，不使用浮点操作
- 处理流程：
  - 提取符号位、指数、尾数
  - 计算非规格化数的重规格化偏移 (renorm_shift)
  - 调整指数偏置：从 FP16 的 15 偏移到 FP32 的 127 (差值 0x70)
  - 特殊值处理：Inf/NaN 检测 (inf_nan_mask)、零检测 (zero_mask)

**FP16 → FP32 (值转换)** `fp16_ieee_to_fp32_value`
- x86 平台使用硬件指令 `_cvtsh_ss`
- 通用实现分两路处理：
  - **规格化数路径**：调整指数偏置 0xE0，然后乘以 2^(-112) 还原
  - **非规格化数路径**：构造指数为 126 的 FP32，减去 0.5 修正偏置
  - 根据 `two_w < 2^27` 选择路径

**FP32 → FP16** `fp16_ieee_from_fp32_value`
- x86 使用 `_cvtss_sh` 硬件舍入
- 通用实现：
  - 通过 `scale_to_inf * scale_to_zero` 预处理溢出/下溢
  - 动态调整 bias 处理非规格化边界
  - 提取 13 位后的指数和尾数组合

### 2. 平台优化

- **x86 F16C 扩展**：检测 `__F16C__` 宏启用硬件转换指令
- **ARM NEON**：
  - 使用原生 `float16_t` 类型
  - 编译器自动生成 `fcvt` 指令进行转换
  - 提供 `native_fp16_to_fp32_value` 等包装函数

### 3. Half 结构体

```cpp
struct alignas(2) Half {
  unsigned short x;  // 16-bit 存储
```

**构造函数**：
- 从位模式构造：`Half(bits, from_bits())`
- 从 float 隐式转换（定义在 Half-inl.h）
- 支持 CUDA `__half`、SYCL `sycl::half` 互转

**设计理念**：
- 算术操作转换为 FP32 执行（注释说明：大多数内核是内存受限，半精度硬件指令在某些 GPU 上效率不高）
- 计算密集型内核可直接使用 CUDA half intrinsics

## Half.cpp 内容

仅包含一个编译期检查：
```cpp
static_assert(std::is_standard_layout_v<Half>);
```
确保 `Half` 是标准布局类型，可安全地进行 memcpy 和跨语言交互。

## 关键实现细节

1. **指数偏置转换**：FP16 偏置 15，FP32 偏置 127，差值 112 (0x70)
2. **非规格化处理**：通过构造魔术常数 (126 << 23) + 减 0.5 实现
3. **特殊值保持**：通过位掩码确保 Inf/NaN 在转换中保持语义
4. **避免浮点依赖**：`fp16_ieee_to_fp32_bits` 可用于不支持浮点的环境

---

**ROCm/Backward 相关**：
- ROCm 要求 `Half()` 默认构造函数标记 `__host__ __device__`
- 支持 HIP 的 `__half` 类型转换
- 兼容旧版 SYCL 1.2.1 和 SYCL 2020
