## CUDA 异常处理机制

这两个文件实现了 PyTorch 中 CUDA 错误的检查和异常处理系统。

### 核心功能

**CUDAException.h** 定义了一系列宏和异常类，用于捕获和处理 CUDA 操作的错误：

- **CUDAError 类** (22-24行)：继承自 `c10::Error`，表示 CUDA 框架级别的异常

- **C10_CUDA_CHECK 宏** (27-37行)：最常用的宏，执行 CUDA 操作并检查返回值。如果出错，调用 `c10_cuda_check_implementation()` 进行错误处理和报告

- **C10_CUDA_CHECK_WARN 宏** (39-46行)：记录 CUDA 警告而不抛出异常

- **C10_CUDA_IGNORE_ERROR 宏** (52-58行)：显式忽略 CUDA 错误，防止错误堆积

- **C10_CUDA_CLEAR_ERROR 宏** (61-64行)：清空 CUDA 错误队列

- **C10_CUDA_KERNEL_LAUNCH_CHECK 宏** (69行)：专门用于检查核函数启动是否成功

- **TORCH_DSA_KERNEL_LAUNCH 宏** (73-84行)：支持设备端断言的核函数启动宏，包含异常信息追踪

**CUDAException.cpp** 实现了错误检查的核心逻辑：

- **c10_cuda_check_implementation() 函数** (11-44行)：
  - 检查 CUDA 错误码和设备端断言失败状态
  - 如果没有错误，直接返回（C10_LIKELY 优化快速路径）
  - 清空最后的 CUDA 错误以防止重复报告
  - 构建详细的错误消息，包含 CUDA 错误说明和后缀信息
  - 如果启用了设备端断言，会获取设备断言失败的详细信息
  - 最后通过 `TORCH_CHECK(false, message)` 抛出异常

### 错误处理流程

1. CUDA 操作通过宏包装（如 `C10_CUDA_CHECK`）
2. 宏捕获返回值和源代码位置信息
3. 调用 `c10_cuda_check_implementation()` 验证
4. 若出错：收集错误消息 → 检查设备断言 → 抛出异常
5. 若正常：快速路径直接返回

---

**主要职责：**
- 统一的 CUDA 错误检查机制
- 详细的错误报告和诊断信息
- 设备端断言的集成支持
- 灵活的错误处理选项（检查、警告、忽略、清空）
- 核函数启动的安全验证
