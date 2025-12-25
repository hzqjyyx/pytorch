**文件功能概述：**

- **量化引擎枚举定义** - 定义了 `QEngine` 枚举类，用于选择运行量化操作的后端引擎
  - 包含 5 个引擎选项：NoQEngine、FBGEMM、QNNPACK、ONEDNN、X86

- **常量别名** - 为每个枚举值提供简化的常量别名（kNoQEngine、kFBGEMM 等）

- **字符串转换函数** - `toString()` 函数将枚举值转换为可读的字符串表示

- **同步要求** - 注释强调需要与 PyTorch 的 `torch/backends/quantized/__init__.py` 中的 `get_qengine_id()` 保持同步

- **类型定义** - 使用 `uint8_t` 作为底层类型，确保枚举大小紧凑
