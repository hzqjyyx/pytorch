# QuantizedLinear.cpp 主要功能分析

这个文件实现了基于 FBGEMM 库的量化线性层操作，用于加速推理时的矩阵乘法运算。

## 核心功能

### 1. INT8 量化线性运算

**`fbgemm_linear_int8_weight_fp32_activation`** (49-182行)
- **输入处理**：接收 FP32 激活值和 INT8 量化权重
- **输入量化**：
  - 使用 `fbgemm::FindMinMax` 计算输入张量的最小/最大值
  - 通过 `fbgemm::ChooseQuantizationParams` 选择量化参数（scale 和 zero_point）
  - 将输入量化为 8-bit 无符号整数
- **矩阵运算**：
  - 使用 `PackAWithQuantRowOffset` 打包量化后的输入矩阵
  - 调用 `fbgemm::fbgemmPacked` 执行 uint8 × int8 矩阵乘法
  - 支持多线程并行计算（通过 `at::parallel_for`）
- **反量化与后处理**：
  - `ReQuantizeForFloat` 将 INT32 结果反量化为 FP32
  - 添加行偏移和列偏移以保证数值正确性
  - 加上偏置项

### 2. 权重预处理

**`fbgemm_linear_quantize_weight`** (225-290行)
- **量化权重**：
  - 将 FP32 权重量化为 INT8（有符号）
  - 计算并返回量化参数（scale, zero_point）
- **列偏移计算**：
  - `CalcColOffsetsTranspose` 计算权重矩阵的列偏移
  - 公式：`col_offsets[i] = sum(column[i]) - B_zero_point * K`
  - 这些偏移在推理时用于校正量化误差

**`fbgemm_pack_quantized_matrix`** (292-324行)
- 将 INT8 量化权重打包成 `fbgemm::PackBMatrix` 格式
- 优化内存布局以提升缓存命中率和向量化效率

### 3. FP16 量化线性运算

**`fbgemm_pack_gemm_matrix_fp16`** (377-407行)
- **权重饱和处理**：
  - `HandleWeightsSaturation` 检查权重是否在 FP16 表示范围内 [5.96e-8, 65504]
  - 超出范围的值会被截断到边界值
- **打包权重**：
  - 将 FP32 权重转换并打包为 `fbgemm::PackedGemmMatrixFP16` 格式

**`fbgemm_linear_fp16_weight_fp32_activation`** (409-455行)
- 输入为 FP32 激活值，权重为预打包的 FP16 格式
- 调用 `fbgemm::cblas_gemm_compute` 执行混合精度矩阵乘法
- 输出 FP32 结果并加上偏置

### 4. 辅助函数

**`RawUint16ToFp16`** (328-344行)
- 手动实现 FP16 到 FP32 的转换
- 解析符号位、指数位、尾数位并重建浮点数

**`CheckAndSaturate`** (346-357行)
- 通用的饱和检查模板函数
- 将超出范围的值限制在 [-max_val, max_val] 内

## 编译配置

- **USE_FBGEMM**：所有实现都在此宏保护下
- 未启用 FBGEMM 时，所有函数返回错误提示
- 运行时检查 `fbgemm::fbgemmSupportedCPU()` 确保 CPU 支持必要的指令集

## 类型系统集成

- 使用 `cpp_custom_type_hack` 将 FBGEMM 的 C++ 对象包装为 PyTorch Tensor
- 注册 `PackBMatrix<int8_t>` 和 `PackedLinearWeightFp16` 为 Caffe2 已知类型
- 支持在 TorchScript 中序列化和传递这些打包格式

---

**其他内容（简略）**：
- 所有公开 API 都标记为 deprecated，将在未来版本移除
- ROCm 相关：无
- Backward 相关：无（此文件仅包含推理操作）
