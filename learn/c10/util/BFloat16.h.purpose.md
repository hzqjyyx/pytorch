**BFloat16.h 主要功能：**

- **定义 BFloat16 数据类型**：一个 16 位浮点数结构体，采用 1 位符号 + 8 位指数 + 7 位尾数的布局

- **位级转换函数**：
  - `f32_from_bits()` - 将 16 位整数转换为 32 位浮点数
  - `bits_from_f32()` - 将 32 位浮点数转换为 16 位整数

- **舍入处理**：`round_to_nearest_even()` 函数实现 BFloat16 的舍入逻辑，包括 NaN 的处理

- **类型转换**：
  - 支持从 float 隐式转换到 BFloat16
  - 支持从 BFloat16 显式转换回 float
  - 支持 CUDA 原生 `__nv_bfloat16` 类型的互转

- **输出流支持**：重载 `operator<<` 以便直接打印 BFloat16 值（转为 float 输出）
