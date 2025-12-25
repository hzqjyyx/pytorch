## verbose_wrapper 文件分析

**verbose_wrapper.h** 定义了两个导出的 API 函数：
- `_mkl_set_verbose(int enable)` - 控制 MKL (Math Kernel Library) 的详细日志输出
- `_mkldnn_set_verbose(int level)` - 控制 MKLDNN (Intel MKL-DNN) 的详细日志输出

**verbose_wrapper.cpp** 提供了这两个函数的具体实现：

1. **`_mkl_set_verbose`** (14-25行)
   - 当编译时启用 MKL (`AT_MKL_ENABLED()`) 时，调用 `mkl_verbose()` 设置 MKL 的详细级别
   - 返回值：成功返回 1，失败返回 0
   - 当 MKL 未启用时直接返回 0

2. **`_mkldnn_set_verbose`** (27-33行)
   - 当编译时启用 MKLDNN (`AT_MKLDNN_ENABLED()`) 时，委托给 `at::native::set_verbose()`
   - 返回设置结果
   - 当 MKLDNN 未启用时返回 0

## 核心功能总结

- **目的**：为 PyTorch 的 ATen 库提供统一的接口来控制底层优化库（MKL 和 MKLDNN）的日志输出
- **条件编译**：根据构建配置决定是否真正调用底层库的 verbose 函数
- **封装**：隐离上层代码对具体库的依赖，提供一致的返回值约定
