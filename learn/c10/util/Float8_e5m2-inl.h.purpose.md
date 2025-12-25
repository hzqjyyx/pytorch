## Float8_e5m2-inl.h 文件分析

这个文件定义了 PyTorch 中 8 位浮点数类型 `Float8_e5m2` 的内联实现。

**核心组件：**

- **格式定义**（12-14行）：5 位指数，2 位尾数，偏置为 15
- **构造与转换**（20-27行）：从 float 构造、隐式转换回 float
- **特殊值检查**（31-37行）：`isnan()` 和 `isinf()` 方法
- **算术运算**（41-92行）：Float8_e5m2 之间的加减乘除操作（含赋值版本）
- **混合类型运算**（94-221行）：
  - 与 float 的算术运算
  - 与 double 的算术运算
  - 与 int 和 int64_t 的算术运算
- **标准库支持**（230-282行）：std::numeric_limits 特化，提供 min/max/epsilon/infinity/NaN 等值

**关键特点：**

- 所有函数都用 `C10_HOST_DEVICE` 标注，支持 CPU 和 GPU 执行
- 算术运算通过转换为 float 执行再转回 Float8_e5m2
- 除法操作标注 `__ubsan_ignore_float_divide_by_zero__` 以忽略除零检查
- 比较操作依赖隐式 float 转换而非直接实现

**主要功能清单：**

• 8 位浮点数类型的完整算术运算支持
• CPU/GPU 双平台兼容
• 与 float/double/int 等类型的自动转换和混合运算
• 符合 C++ 标准库接口（numeric_limits 特化）
• 特殊浮点值（NaN、Inf）的识别和标准化表示
