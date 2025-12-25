## Float8_e5m2fnuz 8位浮点数类型实现

这两个文件定义了 PyTorch 中的 Float8_e5m2fnuz 类型，这是一种 8 位浮点数格式。

### 二进制格式结构
```
s eeeee mm
```
- 1 个符号位
- 5 个指数位
- 2 个尾数位
- 偏置值：16
- 无穷大表示：不支持
- 负零：不支持
- NaN：仅当符号位为1、其余为0时

### 核心转换函数

**fp8e5m2fnuz_from_fp32_value()** - float32 转换为 float8
- 处理溢出：值 ≥ 65536.0f 转换为 NaN (0x80)
- 处理下溢：小于 2^-15 的值转换为次正规数（subnormal），零时返回0（无负零）
- 舍入逻辑：使用偏置和尾数奇偶性实现舍入
- 符号位提取与处理：保留原始符号位

### 类型定义

**Float8_e5m2fnuz 结构体**
- 对齐要求：1 字节
- 存储：uint8_t x
- 支持从位模式构造：Float8_e5m2fnuz(bits, from_bits())
- 支持 float 隐式转换
- 方法：isnan()、isinf()

### 关键特性

- **基于论文**：https://arxiv.org/pdf/2206.02915.pdf
- **运算方式**：通过转换为 float32 执行算术运算，然后转换回 float8
- **输出支持**：重载 operator<< 用于流输出
- **跨设备**：使用 C10_HOST_DEVICE 宏支持 CPU 和 GPU

---

• 定义 8 位浮点数类型 Float8_e5m2fnuz，用于模型压缩和加速
• 实现 float32 到 float8 的精确转换，处理溢出、下溢和舍入
• 不支持无穷大和负零，NaN 用特定位模式表示
• 支持跨 CPU/GPU 执行，提供类型转换和流输出能力
