这个文件定义了 `Float8_e4m3fnuz` 浮点数类型的内联实现，支持与其他数值类型的互操作。

**主要功能模块：**

- **构造和转换**：从 float 构造，隐式转换回 float
- **特殊值检查**：`isnan()` 方法判断 NaN 值（0b10000000）
- **基础算术**：Float8_e4m3fnuz 之间的 +、-、*、/ 和复合赋值运算
- **与 float 混合运算**：Float8_e4m3fnuz 与 float 的四则运算，返回 float 结果
- **与 double 混合运算**：Float8_e4m3fnuz 与 double 的四则运算，返回 double 结果
- **与整数混合运算**：Float8_e4m3fnuz 与 int 和 int64_t 的四则运算
- **std::numeric_limits 特化**：定义 Float8_e4m3fnuz 的数值限制（精度、范围、指数等）

**关键特性：**
- 所有函数标记为 `C10_HOST_DEVICE`，支持 CPU 和 GPU 执行
- 除法操作忽略浮点异常（`__ubsan_ignore_float_divide_by_zero__`）
- 不直接定义比较运算，依赖隐式转换为 float 实现
