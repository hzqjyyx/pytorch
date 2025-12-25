这个文件为 BFloat16 和 Half（fp16）类型提供数学函数重载。

**核心机制：**
- 通过 `is_reduced_floating_point_v<T>` 模板约束，使标准库的数学函数能处理低精度浮点类型
- 转换流程：低精度浮点 → float → 计算 → 转回低精度浮点

**主要功能清单：**
- 三角函数：acos, asin, atan, sin, cos, tan
- 双曲函数：sinh, cosh, tanh, atanh
- 指数对数：exp, expm1, log, log10, log2, log1p
- 舍入函数：ceil, floor, nearbyint, trunc
- 特殊函数：erf, erfc, lgamma, sqrt, rsqrt, abs, pow, fmod
- 判断函数：isfinite
- 邻近值：nextafter（自定义实现，基于 musl 库）

**设计特点：**
- 利用 SFINAE（enable_if_t）在编译时选择正确的重载
- nextafter 使用位级操作直接操作浮点数表示
- MSC_VER + CUDACC 特殊处理 pow 函数的精度问题
