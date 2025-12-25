## 文件功能分析

这两个文件定义和实现了 PyTorch functorch 中的自动微分（AD）解释器，用于处理梯度变换（grad、vjp）和前向模式自动微分（jvp）。

### 核心概念

**GradInterpreterPtr** 和 **JvpInterpreterPtr** 是轻量级包装器，封装了 Interpreter 对象，专门处理 AD 变换的操作调度。

### 主要流程

1. **materializeGradWrappers** (行 31-52)
   - 将张量包装在 TensorWrapper 中以追踪梯度信息
   - 禁用 FuncTorchDynamicLayerFrontMode 确保操作通过正确的层级传播
   - 不可变张量标记为 `is_immutable=true`

2. **checkForInvalidMutationOnCaptures** (行 10-29)
   - 检验在梯度变换中是否有对捕获张量的原地操作
   - 若发现非法原地修改，抛出错误提示用户重写函数

3. **autogradBasedTransformProcess** (行 62-81)
   - 对当前层级的操作进行梯度包装物质化
   - 设置 dispatch key TLS 并调用操作的 boxed 版本
   - 用于处理梯度/jvp 的本地变换逻辑

4. **autogradBasedTransformSendToNext** (行 83-202)
   - 核心的多层级梯度处理逻辑，6 步过程：
     1. 复制参数到栈
     2. 解包当前层级的 TensorWrapper
     3. 调用操作（支持可选的梯度模式守卫）
     4. 包装输出张量
     5. 刷新原始参数的元数据（处理原地修改的尺寸/步幅变化）
     6. 清理栈上的临时参数副本
   - 使用 bitset 追踪哪些输出别名到不可变输入

### 关键设计

- **TensorWrapper** 跟踪张量的梯度变换层级和不可变性
- **别名追踪**：通过 bitset 检测输出是否别名到不可变输入，影响包装策略
- **元数据刷新**：原地操作可能改变尺寸/步幅，需要重新同步 wrapper 的元数据
- **梯度模式守卫**：对应 grad/jvp 恢复之前的自动求导状态（如 no_grad）

---

### 关键功能点

- 实现梯度变换（Grad）和前向微分（Jvp）的操作解释器
- 验证梯度变换中禁止对捕获张量的原地修改
- 通过 TensorWrapper 在 functorch 变换栈中追踪张量的梯度信息
- 处理多层级变换间的张量包装/解包和元数据同步
- 使用 bitset 优化追踪输出的别名关系以决定包装策略
- 支持梯度模式恢复（no_grad 等）跨变换层级的正确传播
