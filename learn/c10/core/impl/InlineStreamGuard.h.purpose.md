## InlineStreamGuard.h 文件分析

这个文件定义了三个 RAII 类来管理 CUDA/设备流的状态：

**InlineStreamGuard**
- 在构造时将当前设备切换到流对应的设备，并将该设备的默认流设置为指定的流
- 析构时恢复原始设备和流的状态
- 禁用拷贝和移动操作
- `reset_stream()` 方法可以更改当前流（如果设备不同则同时更改设备）
- 提供 `original_stream()`、`current_stream()` 和 `current_device()` 等查询方法

**InlineOptionalStreamGuard**
- InlineStreamGuard 的可选版本，允许创建未初始化的守卫
- 可以在任何时候通过 `reset_stream()` 初始化
- 通过 `reset()` 方法可以恢复到未初始化状态
- 适用于流可能为空的场景

**InlineMultiStreamGuard**
- 管理多个流的守卫，允许在不同设备上设置不同的流
- 构造时接收流数组，逐个交换每个流
- 析构时依次恢复所有原始流
- 验证所有流必须来自同一设备类型

**核心特性：**
- 基于 RAII 模式确保资源正确管理
- 跟踪原始设备/流以便正确恢复
- 通过模板参数 `T` 支持不同的守卫实现
- 仅用于测试的特殊构造函数（VirtualGuardImpl）

**主要用途：**
- 确保在特定流上执行操作
- 自动管理设备和流的上下文切换
- 防止设备/流上下文泄漏
