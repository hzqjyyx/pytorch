## Padding.h 文件功能分析

**核心功能：**
- 定义 padding 操作的分发接口（dispatch stubs）
- 为反射填充（reflection padding）和复制填充（replication padding）的前向和反向传播提供声明

**关键组件：**

- **padding_fn 函数指针类型** (line 8)
  - 签名：`void (*)(const Tensor&, const Tensor&, IntArrayRef)`
  - 用于定义具体的 padding 核心实现

- **Dispatch 声明**
  - reflection_pad1d/2d/3d_kernel：反射填充的 1D/2D/3D 前向实现
  - replication_pad1d/2d/3d_kernel：复制填充的 1D/2D/3D 前向实现

- **check_valid_input 模板函数** (line 28-59)
  - 验证输入张量的维度合法性
  - 支持批处理模式（batch + spatial dims）和非批处理模式
  - 检查：
    - padding 大小是否为 `2 * dim`
    - 输入维度是 `dim+1` 或 `dim+2`
    - 允许批大小为 0，但其他维度必须非零

**主要特点：**
- 头文件提供接口声明，具体实现在其他源文件
- 支持多维度 padding 操作（1D/2D/3D）
- 分别处理反射和复制两种填充方式
