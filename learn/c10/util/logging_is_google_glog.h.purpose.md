这个文件是 PyTorch 中用于配置 Google glog 日志库集成的头文件。

**主要功能：**

- **条件编译包含**：根据是否在 CUDA 编译器（`__CUDACC__`）或最小化 glog 模式（`C10_USE_MINIMAL_GLOG`）下，决定是否包含 `glog/stl_logging.h`

- **STL 容器日志支持**：在非 CUDA 环境下，使用 glog 的 STL 日志功能，允许直接输出 vector、map、set 等容器到日志

- **CUDA 兼容性处理**：在 CUDA 编译器中，由于 nvcc 不支持 STL 日志，提供空实现的容器输出操作符重载（返回 ostream 不做任何操作）

- **Torch 检查宏定义**：
  - 包装 glog 的 `CHECK_*` 宏为 `TORCH_CHECK_*` 形式（EQ、NE、LE、LT、GE、GT）
  - 定义调试模式下的 `TORCH_DCHECK_*` 宏
  - 在优化模式（`NDEBUG`）下，使 `DCHECK_*` 宏生成零代码

- **指针检查宏**：`TORCH_CHECK_NOTNULL` 和 `TORCH_DCHECK_NOTNULL` 用于非空指针验证

- **文件行号日志**：`LOG_AT_FILE_LINE` 宏支持自定义日志源位置信息，用于通用警告/错误处理函数
