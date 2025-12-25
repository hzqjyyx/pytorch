## MathBitsFallback.h 主要功能

这个文件实现了一个通用的 fallback 机制，用于处理具有自逆（self-inverse）特性且使用 Tensor Bit（通过 DispatchKey 实现）来维护状态的操作。

### 核心设计

**MathOpFallback 结构体**提供了一个可继承的基类，目前支持两种 math bit：
- **Conjugate bit**：在调用 `.conj()` 时设置
- **Negative bit**：在调用 `.conj().imag` 时设置

### 处理逻辑（fallback_impl 方法）

将不同类型的操作进行反糖化处理：

1. **非原地操作**：直接将所有输入具体化（materialize）后调用原操作
2. **原地操作**：将 `x.op_(args)` 转化为 `x.conj_().op_(args).conj_()`
3. **out= 操作**：将 `op(x, args, out=y)` 转化为 `y.copy_(op(x, args))`

### 关键处理步骤

- **Alias 分析**：检查参数是否为可写（mutable）来判断操作类型
- **Mutable 输入克隆**：有 math bit 的可写输入必须被克隆，防止内存共享时的数据错误
- **Stack 操作**：通过修改 JIT stack 中的参数来实现 bit 的具体化
- **输出回写**：对于原地操作，将修改后的张量复制回原始输入

### 关键约束

- `clone` 和 `copy_` 必须正确理解和处理 math bit 的语义
- 不支持多个可写张量都有 bit 设置
- 可写 TensorList 中不能包含有 bit 设置的张量

### 核心特点

- 主要功能点：
  - 通过 DispatchKey 实现 Tensor 级别的 bit 状态管理
  - 自动处理 math bit 的具体化流程
  - 支持非原地、原地、out= 三种操作模式
  - 避免内存别名导致的数据错误
  - 减少为每个 math bit 操作手动编写 fallback 的工作量
