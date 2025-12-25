# complex_math.h/cpp 主要功能

这两个文件为 PyTorch 的 `c10::complex<T>` 类型提供完整的数学函数支持，作为跨平台（CPU/CUDA/HIP）的统一接口。

## 核心架构

**条件编译策略**：所有函数通过预处理器指令选择实现路径
- `__CUDACC__` 或 `__HIPCC__`：GPU 路径，委托给 `thrust::` 实现
- CPU 路径：优先使用标准库 `std::complex<T>`，但对有缺陷的实现提供自定义版本

## 主要功能分类

### 1. 指数和对数函数（complex_math.h:8-47）
- `exp()`, `log()`, `log10()`, `log2()`
- `log1p()` (293-341)：专门处理 log(1+z)，针对不同平台优化
  - macOS/CUDA：使用极坐标形式避免数值误差
  - 其他 CPU：基于 NumPy 的改进算法
- `expm1()` (343-358)：计算 exp(z)-1，使用三角恒等式减少误差

### 2. 幂函数（complex_math.h:49-152）
- `sqrt()`：平方根，对某些标准库提供自定义实现
- `pow()` 系列：6 个重载版本支持 complex-complex、complex-scalar、scalar-complex 及混合类型

### 3. 三角函数（complex_math.h:154-222）
- 基础：`sin()`, `cos()`, `tan()`
- 反三角：`asin()`, `acos()`, `atan()`
- `acos()` 对 libc++ 提供自定义实现

### 4. 双曲函数（complex_math.h:224-290）
- 基础：`sinh()`, `cosh()`, `tanh()`
- 反双曲：`asinh()`, `acosh()`, `atanh()`

## 自定义实现细节（complex_math.cpp）

针对 `_LIBCPP_VERSION` 或不支持 C99 的 `__GLIBCXX__`：

### compute_csqrt() (20-48)
修复标准库在以下场景的数值问题：
1. **纯虚数**（real=0）：`sqrt(iy) = v(1+i·sign(y))`，其中 `v=√(|y|/2)`
2. **实部≥0**：`t = √((real+|z|)/2)`，结果为 `(t, imag/(2t))`
3. **实部<0**：`t = √((-real+|z|)/2)`，结果为 `(|imag|/(2t), t·sign(imag))`
4. **特殊值**：委托标准库处理 inf/NaN

### compute_cacos() (55-70)
基于 W. Kahan 1986 论文公式：
```
acos(z).real = 2·atan2(√(1-z).real, √(1+z).real)
acos(z).imag = asinh((√(conj(1+z))·√(1-z)).imag)
```
避免 libc++ 使用极坐标导致的数值误差（arg 接近 0, π/2, π, 3π/4 时）

## 命名空间导出（complex_math.h:362-406）
将所有函数同时导出到全局命名空间和 `std::` 命名空间，实现与标准库的无缝集成。

---

**简要提及**：
- ROCm 支持：通过 `__HIPCC__` 宏使用 HIP/thrust 后端
- 向后兼容：为缺少 C99 复数支持的旧版 GCC libstdc++ 提供 fallback
