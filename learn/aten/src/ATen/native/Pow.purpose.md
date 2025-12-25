# Pow.h 和 Pow.cpp 的主要功能

## Pow.h 的核心内容

定义了整数幂运算的实现函数 `powi` 和 `powi_impl`：

- `powi_impl`: 使用二进制分解算法计算整数的整数次幂（快速幂），支持负数基数
- `powi`: 包装函数，处理无符号和有符号整数的不同情况
  - 无符号整数：直接调用 `powi_impl`
  - 有符号整数：处理负指数特殊情况
    - 指数为负时，只有基数为 1 或 -1 才有非零结果
    - 基数为 1：返回 1
    - 基数为 -1：根据指数奇偶性返回 ±1
    - 其他情况：返回 0（截断结果）

- 声明两个调度函数的接口：`pow_tensor_tensor_stub` 和 `pow_tensor_scalar_stub`

## Pow.cpp 的核心内容

### Meta 函数（形状和数据类型推导）

- `pow(Tensor, Tensor)`: 使用二元借用操作构建输出
- `pow(Tensor, Scalar)`: 检查整数不能有负指数，进行类型提升
- `pow(Scalar, Tensor)`: 设置输出形状和数据类型

### 实现函数（实际计算）

- `pow_Tensor_Tensor_out`: 调用 tensor-tensor 调度函数
- `pow_Tensor_Scalar_out`: 优化处理 exp=0（返回1）和 exp=1（复制base）的特殊情况，否则调用 tensor-scalar 调度函数
- `pow_Scalar_out`: base=1 时填充1，否则重新调度为 Tensor-Tensor 版本

### float_power 函数组

统一处理浮点幂运算，自动转换为 `kComplexDouble` 或 `kDouble`：

- `float_power_out`: 三个重载版本（Tensor-Tensor, Tensor-Scalar, Scalar-Tensor）
- `float_power`: 三个重载版本（返回新 Tensor）
- `float_power_`: 两个重载版本（原位操作）

---

## 功能总结

- **整数幂算法**: 快速幂实现（O(log n)）
- **多态支持**: 处理 Tensor-Tensor、Tensor-Scalar、Scalar-Tensor 三种组合
- **类型系统**: 自动类型推导和转换（NumPy 兼容）
- **优化**: 特殊值快速路径（exp=0/1）
- **浮点统一**: `float_power` 强制浮点/复数输出
- **调度**: 通过 stub 调度到不同后端实现（CPU/CUDA 等）
