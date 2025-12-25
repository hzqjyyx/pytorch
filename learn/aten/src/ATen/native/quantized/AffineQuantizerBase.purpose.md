# AffineQuantizerBase 核心功能解析

这两个文件实现了 PyTorch 量化的基础操作，提供了浮点数与量化整数之间的转换功能。

## 核心概念

**Affine Quantization（仿射量化）**采用线性映射：
- 量化公式：`Xq = Round(Xf / scale + zero_point)`
- 反量化公式：`Xf = scale * (Xq - zero_point)`

其中：
- `scale`：缩放因子，控制量化精度
- `zero_point`：零点偏移，确保浮点数 0 能精确映射到量化值
- `Round`：使用 `std::nearbyint`，采用 round-to-even（银行家舍入）策略

## 主要功能模块

### 1. 单值量化 (quantize_val)

**位置**：AffineQuantizerBase.cpp:38-54 (FBGEMM) / 123-139 (非FBGEMM)

两种实现路径：
- **USE_FBGEMM 路径**：调用 `fbgemm::Quantize`，利用优化的 SIMD 实现
- **通用路径**：手动实现量化逻辑
  ```cpp
  int64_t qvalue;
  float inv_scale = 1.0f / scale;
  qvalue = zero_point + Round(value * inv_scale);
  qvalue = std::clamp(qvalue, qmin, qmax);
  ```

支持的量化类型：`qint8`, `quint8`, `qint32`

### 2. ARM 特化版本 (quantize_val_arm)

**位置**：AffineQuantizerBase.cpp:75-86 / 142-163

针对 ARM 架构的优化实现，关键特性：
- 直接使用 `int32_t` 类型避免溢出
- 非 MSVC 编译器使用 `__builtin_add_overflow` 检测溢出
- 显式模板实例化 `uint8_t` 和 `int8_t` 版本

### 3. 向量量化 (quantize_vec)

**位置**：AffineQuantizerBase.cpp:57-69 (FBGEMM) / 166-176 (非FBGEMM)

批量量化浮点数组：
- **FBGEMM 路径**：直接调用 `fbgemm::Quantize`，支持 SIMD 加速
- **通用路径**：循环调用 `quantize_val`
- 模板参数 `precision` 控制位宽（默认 8，`qint32` 使用 32）

### 4. 反量化 (dequantize_val)

**位置**：AffineQuantizerBase.cpp:99-104 (FBGEMM) / 187-189 (非FBGEMM)

将量化整数还原为浮点数：
- **FBGEMM**：调用 `fbgemm::Dequantize`
- **通用**：`scale * (value.val_ - zero_point)`

### 5. 重量化 (requantize_val)

**位置**：AffineQuantizerBase.cpp:209-217

在两种量化参数间转换而不回到浮点：
```cpp
dequantize(src_scale, src_zero_point, src)
  → quantize(dst_scale, dst_zero_point, ...)
```

支持类型转换组合（9 种显式实例化，如 `qint8→quint8`）

### 6. 整数重量化 (requantize_from_int)

**位置**：AffineQuantizerBase.cpp:220-228

从 `int64_t` 累加结果直接量化到目标类型：
```cpp
quantize_down = zero_point + lrintf(src * multiplier)
```

用于高效的融合操作（如量化卷积后的累加结果）

### 7. 浮点参数量化 (quantize_val_float_qparams)

**位置**：AffineQuantizerBase.cpp:200-206

特殊场景：zero_point 为浮点数（如 embedding 量化）：
```cpp
zero_point = -Xmin / scale  // Xmin 是行最小值
qvalue = lrintf(value * inv_scale + zero_point)
```

## 设计细节

### 舍入策略
- 使用 `std::nearbyint` 而非 `std::round`
- Round-to-even 在 x86/ARM 上性能更优
- 与 SIMD 指令（`_mm512_cvtps_epi32`）行为一致

### 边界检查
- `checkZeroPoint` (AffineQuantizerBase.cpp:18-31) 验证 zero_point 在类型范围内
- 量化结果通过 `std::clamp` 确保在 `[qmin, qmax]` 内

### 条件编译
- `USE_FBGEMM`：优先使用 FBGEMM 库的优化实现
- `__ARM_NEON__`：ARM NEON 优化
- `__ANDROID__`：Android NDK 兼容性处理
- `_MSC_VER`：MSVC 编译器不支持 `__builtin_add_overflow`

### 显式模板实例化
文件末尾（230-291 行）显式实例化所有需要导出的模板函数，确保 API 可链接。

---

**ROCm/Backward 相关**：
- 该文件无 ROCm 特定代码
- 无反向传播逻辑（量化是前向操作，梯度通过 Straight-Through Estimator 处理）
