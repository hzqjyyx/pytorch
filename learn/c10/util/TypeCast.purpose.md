TypeCast 文件提供了 C++ 类型转换的安全机制，特别是处理可能导致未定义行为的转换情况。

**核心功能：**

• **类型转换适配器** - `static_cast_with_inter_type` 模板根据源类型和目标类型选择合适的转换策略

• **复数类型处理** - `needs_real` 和 `maybe_real` 用于在复数到实数转换时提取实部

• **布尔转换** - `maybe_bool` 处理复数到布尔的转换，使用 `real() || imag()` 而非简单强制转换

• **uint8 特殊处理** - 通过中间转换到 `int64_t` 来解决负浮点数到无符号整数的未定义行为，提高跨编译器一致性

• **Float8 格式支持** - 针对 Float8_e4m3fn、Float8_e5m2 等低精度格式到 complex<Half> 的专门转换实现

• **溢出检测** - `checked_convert` 函数在转换前检测数值溢出，溢出时调用 `report_overflow` 抛出异常

• **编译器诊断抑制** - 使用 `__ubsan_ignore_undefined__` 和编译器诊断指令来管理关于隐式转换的警告

• **跨设备支持** - `C10_HOST_DEVICE` 宏支持在 CPU 和 GPU 代码中使用
