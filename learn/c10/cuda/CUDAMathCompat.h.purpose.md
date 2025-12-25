## CUDAMathCompat.h 文件分析

这个文件定义了一套在 CUDA 和 HIP 平台上通用的数学函数兼容层。

**核心机制：**
- 通过条件编译宏 `__MATH_FUNCTIONS_DECL__` 统一声明方式
- 在 CUDA RTC（运行时编译）模式下，使用 `C10_HOST_DEVICE` 不加 inline
- 其他 CUDA 编译环境下，使用 `inline C10_HOST_DEVICE`
- 所有函数定义在 `c10::cuda::compat` 命名空间内

**包装的数学函数对（浮点数和双精度）：**

- `abs()` → `fabsf()`/`fabs()`
- `exp()` → `expf()`/`exp()`
- `ceil()` → `ceilf()`/`ceil()`
- `copysign()` → `copysignf()`/`copysign()`（含 CPU 路径保护）
- `floor()` → `floorf()`/`floor()`
- `log()` → `logf()`/`log()`
- `log1p()` → `log1pf()`/`log1p()`
- `max()`/`min()` → `fmaxf()`/`fminf()` 等
- `pow()` → `powf()`/`pow()`
- `sincos()` → `sincosf()`/`sincos()`
- `sqrt()`/`rsqrt()` → `sqrtf()`/`rsqrtf()` 等
- `tan()`/`tanh()` → `tanf()`/`tanhf()` 等
- `normcdf()` → `normcdff()`/`normcdf()`

**主要作用：**

- 提供统一的数学函数接口，屏蔽 CUDA 和 HIP 的差异
- 简化 GPU 内核代码中的函数调用
- 处理浮点精度重载，避免代码中显式调用 `fabsf` vs `fabs` 等
