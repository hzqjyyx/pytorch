## Vitals 系统概述

Vitals 是 PyTorch 的一个诊断和监控系统，用于跟踪运行时的关键指标。

### 核心概念

**TorchVitalAttr** - 单个属性的容器
- 存储字符串值
- 提供 `operator<<` 用于追加数据
- 提供 `write()` 方法用于直接赋值
- 当 Vitals 启用时才实际存储数据

**TorchVital** - 命名的指标组合
- 包含多个属性（attrs）的 `unordered_map`
- 析构时如果启用会输出所有属性到 stdout
- 通过 `create()` 方法创建或获取属性

**APIVitals** - 全局单例接口
- 通过字符串名称访问 Vitals（供 Python API 调用）
- `setVital()` - 设置特定的属性值
- `readVitals()` - 读取所有已记录的指标为字符串
- 内部维护 `name_map_` 存储所有 TorchVital 对象

### 启用机制

- 通过 `TORCH_VITAL` 环境变量控制
- `torchVitalEnabled()` 检查该环境变量是否设置（非空）
- 全局 `VitalsAPI` 对象在初始化时会设置默认的 CUDA 指标

### 宏定义便利工具

- `TORCH_VITAL_DECLARE/DEFINE` - 声明和定义全局 TorchVital 对象
- `TORCH_VITAL(name, attr)` - 快速访问指定的属性

### 主要特性

- **性能考虑** - 禁用时几乎没有开销，因为返回静态的空属性对象
- **单例模式** - APIVitals 通过删除拷贝/移动构造确保唯一性
- **灵活的值写入** - 支持任何可 `<<` 输出流的类型

---

### 功能要点

- 追踪运行时关键指标（如 CUDA 使用情况）
- 通过环境变量动态启用/禁用
- 对象销毁时自动输出记录的指标
- 支持 C++ 和 Python API 访问
- 禁用时性能开销极小
