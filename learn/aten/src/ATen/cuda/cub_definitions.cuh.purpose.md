• **版本检查**：通过 `CUB_VERSION` 宏检测 CUDA Unified Backend (CUB) 库的版本

• **特性条件编译**：根据 CUB 版本提供宏定义，条件启用特定功能：
  - `CUB_SUPPORTS_NV_BFLOAT16()`：CUB >= 1.13 时启用 bfloat16 排序支持
  - `USE_GLOBAL_CUB_WRAPPED_NAMESPACE()`：检测命名空间包装支持（CUDA >= 11.5）
  - `CUB_SUPPORTS_UNIQUE_BY_KEY()`：CUB >= 1.16 时启用 UniqueByKey 操作
  - `CUB_SUPPORTS_SCAN_BY_KEY()`：CUB >= 1.15 时启用按键扫描
  - `CUB_SUPPORTS_FUTURE_VALUE()`：CUB >= 1.15 时启用 FutureValue 支持

• **兼容性适配**：为不同版本的 CUB 库提供统一接口，使 PyTorch 能够在多个 CUDA 环境下编译运行
