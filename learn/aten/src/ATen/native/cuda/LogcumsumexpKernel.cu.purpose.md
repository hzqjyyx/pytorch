## LogcumsumexpKernel.cu 功能分析

这个文件实现了 CUDA 上的 **logcumsumexp**（对数累积求和指数）操作，用于数值稳定地计算累积的对数求和指数。

### 核心组件

**1. 复数 min/max 辅助函数** (第 15-28 行)
- `_logcumsumexp_minmax`: 处理复数类型的 min/max 比较
- 只比较实部，并正确处理 NaN 值

**2. 标量 log-add-exp 助手** (第 30-45 行)
- `_log_add_exp_helper` (浮点版本): 数值稳定的 log(exp(x) + exp(y)) 计算
- 使用 log1p 和 min/max 技巧避免浮点溢出
- 特殊处理 NaN 和无穷大情况

**3. 复数 log-add-exp 助手** (第 73-102 行)
- `_log_add_exp_helper` (复数版本): 扩展到复数域
- 手动实现复数指数函数 `_fast_build_exp` 和 `_fast_build_exp_inf`
- 加快编译速度（避免使用标准库复数运算）
- 处理实部无穷大的特殊情况

**4. 主启动函数** (第 104-122 行)
- `launch_logcumsumexp_cuda_kernel`: CUDA 核心启动器
- 支持浮点数、复数、半精度和 BFloat16 数据类型
- 使用 `scan_dim` 执行沿指定维度的扫描操作

### 关键特点

- **数值稳定性**: 通过 log-sum-exp 技巧避免溢出和下溢
- **复数支持**: 完整的复数类型处理
- **NaN 处理**: "感染性 NaN" 的正确传播
- **编译优化**: 手动实现复数运算以加快编译

### 要点总结

- 实现 logcumsumexp 的 CUDA 核心
- 支持浮点和复数数据类型
- 使用 log-sum-exp 数值稳定化技巧
- 通过手动复数指数实现加快编译
- 正确处理 NaN、无穷大等边界情况
- 依赖 `ScanKernels.h` 和 `ScanUtils.cuh` 进行扫描操作
