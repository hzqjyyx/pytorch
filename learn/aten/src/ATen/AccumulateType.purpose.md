AccumulateType 定义了一套类型系统，用于在数值计算中选择合适的中间计算精度。

**核心概念：**
当执行 reduce、matmul 等操作时，为了避免舍入误差累积，PyTorch 会在比输入/输出更高精度的类型上进行中间计算。例如 float16 输入可能用 float32 累积。

**类型映射规则（AccumulateType.h 第 34-40 行）：**
- bool → bool
- 浮点数：
  - CUDA 上：float（除非输入已是 double，则保持 double）
  - CPU 上：double
  - MPS/XPU：float
- 整数：int64_t

**具体实现：**

AccumulateType.h 中用宏定义了各设备的映射表：
- `ACC_TYPE` 基础宏：为特定类型和设备定义 AccumulateTypeDevice 特化
- `CPU_ACC_TYPE`、`CUDA_ACC_TYPE` 等：设备特定宏
- 覆盖所有标量类型（BFloat16、Half、Float8、int8_t、complex 等）

AccumulateType.cpp 提供了两个运行时接口：
- `toAccumulateType(ScalarType, DeviceType)`：根据输入类型和设备返回累积类型
- `toAccumulateType(ScalarType, bool is_cuda)`：便利重载

**关键特点：**

- CUDA 优化：避免双精度（性能开销大），大多数浮点用 float 累积
- MPS 对齐：类似 CUDA 策略，double 也用 float 累积
- XPU 保守：double 保持 double
- CPU 精度优先：float 提升到 double，提供最高精度

**关键设计决策：**

- 编译时接口（模板）+ 运行时接口（函数）并存
- 设备类型参数化而非 is_cuda 布尔值，支持 MPS/XPU 等新设备
- 复数类型特殊处理：Complex<Half> → Complex<float> 等
