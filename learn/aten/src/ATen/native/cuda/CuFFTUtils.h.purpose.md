根据我刚才读取的文件内容，让我为你解释 `CuFFTUtils.h` 的主要功能：

## 文件概览
这是一个 CUDA FFT（快速傅里叶变换）工具头文件，位于 PyTorch 的 ATen（张量库）中。

## 主要功能

### 1. **常量定义**
```
constexpr int max_rank = 3;
```
定义最大维度为 3，考虑到批处理维度和复数维度，总共最多支持 5 维数据。

### 2. **错误枚举转换函数** (`_cudaGetErrorEnum`)
- 将 cuFFT 库返回的错误代码转换为可读的字符串
- 支持 15+ 种 cuFFT 错误类型，例如：
  - `CUFFT_SUCCESS`：成功
  - `CUFFT_ALLOC_FAILED`：内存分配失败
  - `CUFFT_INVALID_PLAN`：无效的 FFT 计划
  - `CUFFT_NOT_IMPLEMENTED`：功能未实现
  - 等等
- 对于未知错误，返回错误代码的数字表示

### 3. **错误检查宏** (`CUFFT_CHECK`)
```cpp
static inline void CUFFT_CHECK(cufftResult error)
```
- 检查 cuFFT 操作的返回值
- 如果出现错误（返回值不是 `CUFFT_SUCCESS`），则：
  - 调用 `_cudaGetErrorEnum()` 获取错误描述
  - 使用 `TORCH_CHECK` 抛出异常，终止执行

## 用途
这是一个工具类文件，为 PyTorch 中的 FFT 操作提供：
- **统一的错误处理机制**
- **易于阅读的错误消息**
- **代码复用**（避免在多个 FFT 实现中重复错误处理代码）

通常在其他 CUDA FFT 相关的源文件中被包含和使用。
