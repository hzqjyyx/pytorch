## 文件功能分析

### 核心职责
这两个文件实现了PyTorch中的**线程本地存储（TLS）机制**，用于管理Python torch function模式的全局状态。

### 关键组件

**PythonTorchFunctionTLS 结构体** (`PythonTorchFunctionTLS.h:10-30`)
- 包含两个核心私有成员：
  - `disabled_state_`: 枚举类型，记录torch function的禁用状态（ENABLED/SUBCLASSES_DISABLED/ALL_DISABLED）
  - `stack_`: 存储SafePyObject智能指针的向量，维护用户定义模式的堆栈

**静态接口方法** (`PythonTorchFunctionTLS.cpp`)
- `push_onto_stack()`: 将模式入栈
- `pop_stack()`: 弹出栈顶模式并返回
- `get_stack_at(idx)`: 获取指定索引处的模式引用
- `stack_len()`: 返回当前堆栈深度
- `set_disabled_state()` / `get_disabled_state()`: 管理禁用状态
- `set_state()` / `get_state()`: 获取/设置完整的TLS状态

**工具函数** (`PythonTorchFunctionTLS.cpp:44-52`)
- `torch_function_mode_enabled()`: 检查torch function模式是否启用（状态非全禁用 且 堆栈非空）
- `torch_function_all_disabled()`: 检查是否完全禁用

### 主要特点

- **线程安全**: 使用thread_local关键字确保每个线程有独立的pythonTorchFunctionState实例
- **堆栈管理**: 支持模式的嵌套和层级管理
- **状态控制**: 提供细粒度的禁用状态控制（全局禁用/子类禁用/启用）

### 功能概览

- torch function模式的线程本地生命周期管理
- 维护模式嵌套堆栈
- 提供禁用/启用状态的动态控制机制
- 支持Python对象的安全包装存储
