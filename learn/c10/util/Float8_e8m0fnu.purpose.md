Float8_e8m0fnu 是一个 8 位浮点数类型的实现，遵循 OCP MX 格式规范。

**二进制格式结构：**
- 8 个指数位（exponent bits）
- 0 个尾数位（mantissa bits）
- 0 个符号位（sign bits）
- 总共 8 位用于表示指数

**核心功能：**

1. **Float32 到 Float8_e8m0fnu 的转换** (`fp8e8m0fnu_from_fp32_value`)
   - 从 32 位浮点数的位表示中提取指数部分
   - 处理特殊情况：NaN 和 ±Inf 映射到 e8m0 的 NaN
   - 实现 RNE（Round to Nearest, ties to Even）舍入逻辑
   - 使用 Guard、Round、Sticky 位进行精确舍入控制

2. **Float8_e8m0fnu 结构体**
   - 单字节对齐存储 (`alignas(1)`)
   - 支持从位表示构造 (`from_bits()`)
   - 支持与 float 的隐式转换
   - 提供 NaN 检测方法 (`isnan()`)
   - 支持输出流操作符

**关键实现细节：**

- 由于 e8m0 格式只有指数位，LSB 表示隐含的尾数位（正常数为 1，非正常数为 0）
- 舍入逻辑中，当 Guard 位为 1 时，根据 Round 位、Sticky 位和 LSB 决定是否进位
- 指数溢出时已提前返回，确保加 1 不会越界

**文件组成：**

- **Float8_e8m0fnu.h**：声明类型定义和转换函数，包含内联实现文件
- **Float8_e8m0fnu.cpp**：编译期断言，验证 Float8_e8m0fnu 为标准布局类型

**主要特点：**

- 极致的位存储密度（8 位表示浮点数）
- 精确的舍入控制（RNE 算法）
- 支持 CPU 和 GPU 执行（`C10_HOST_DEVICE` 宏）
