这个文件提供了针对不同数值类型优化的数学工具函数，主要解决两个问题：

## 核心功能

### 1. 类型安全的 NaN 检测 (`_isnan`)

提供了比 `std::isnan` 更高效的 NaN 检测，针对不同类型有专门实现：

- **整数类型**：直接返回 `false`，避免无意义的类型转换
- **浮点类型**：在 CUDA 环境使用 `::isnan`，否则用 `std::isnan`
- **复数类型**：检查实部和虚部是否有 NaN
- **半精度类型** (Half, BFloat16)：转换为 float 后检查
- **8位浮点类型** (Float8_e5m2, Float8_e4m3fn, Float8_e5m2fnuz, Float8_e4m3fnuz)：调用各自的 `isnan()` 方法

注意：第 56-58 行有重复的 BFloat16 非模板版本定义。

### 2. 类型安全的无穷大检测 (`_isinf`)

类似 `_isnan` 的设计模式：

- **整数类型**：直接返回 `false`
- **浮点类型**：CUDA 环境用 `::isinf`，否则用 `std::isinf`
- **Half/BFloat16**：转换为 float 后检查
- **Float8_e5m2**：调用 `isinf()` 方法
- **Float8_e4m3fn/fnuz 和 Float8_e5m2fnuz**：直接返回 `false`（这些格式不支持无穷大）

### 3. CUDA 优化的数学函数

为 GPU 计算提供快速近似版本：

- **`exp`**：CUDA/HIP 环境使用 `__expf` 快速近似
- **`log`**：CUDA/HIP 环境使用 `__logf` 快速近似
- **`log1p`**：CUDA/HIP 环境用 `__logf(1.0f + x)` 近似（注释提到会损失精度，因为没有 `__log1pf`）
- **`tan`**：CUDA/HIP 环境使用 `__tanf` 快速近似

所有函数都有 double 类型的特化版本，使用标准库实现以保持精度。

## 设计特点

- 使用 SFINAE 和 `std::enable_if_t` 进行编译期类型分发
- `C10_HOST_DEVICE` 宏确保函数可在 CPU 和 GPU 上运行
- 针对性能关键路径（整数类型的 NaN/Inf 检测）避免不必要的计算
- 在 GPU 上优先使用快速近似函数以提升带宽利用率

---

**ROCm 相关**：
- 支持 HIP runtime（第 3-5 行）
- `__HIP_ARCH__` 宏用于 HIP 设备代码路径

**Backward 相关**：无
