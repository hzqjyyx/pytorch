## CPUFallback 核心功能

这两个文件实现了 PyTorch 的 **CPU 回退机制**，允许在其他设备（如 XLA、MPS 等后端）没有实现某个算子时，自动将计算回退到 CPU 执行。

### 工作流程

**Step 1: 输入转换 (CPU化)**
- 遍历所有输入参数，识别三类需要处理的输入：
  - `Tensor` 单个张量
  - `TensorList` 张量列表
  - `OptionalTensorList` 可选张量列表
- 将所有非CPU设备的张量批量转换到CPU（通过 `at::_to_cpu`）
- 记录原始设备信息（`tgt_device`）用于后续恢复
- 特殊处理：跳过 undefined 张量，避免不必要的转换

**Step 2: CPU执行**
```cpp
op.redispatchBoxed(c10::DispatchKeySet(cpu_dispatch_key), stack);
```
使用 CPU DispatchKey 重新分发算子调用，执行CPU实现。

**Step 3: 处理可变别名 (Mutable Aliases)**
检查输入参数的 `AliasInfo`，如果标记为 `isWrite()`：
- 对于单个张量：通过 `_copy_from_and_resize` 将CPU结果拷贝回原始设备张量
- 对于张量列表：逐个元素拷贝回原始列表
- 目的：确保原地修改操作（如 `add_`）正确反映到原始设备

**Step 4: 输出转换 (恢复设备)**
根据输出的别名类型分三种情况：

1. **可变别名输出** (`isWrite()`)：
   - 直接将原始输入张量放回栈上（因为已经在Step 3更新过）
   - 需要匹配输入输出的 `AliasInfo` 来找到对应关系

2. **不可变别名输出** (View 操作，`!isWrite()`)：
   - ⚠️ **无法正确处理** - 发出警告或错误
   - 原因：View 操作要求共享存储，但跨设备无法共享
   - 见 `CPUFallback.cpp:199` 的详细说明

3. **普通输出** (无别名)：
   - 将CPU输出张量拷贝回原始设备（`.to(*tgt_device)`）

### 关键辅助函数

**`to_cpu` 模板函数** (line 24-59)
- 处理 `vector<Tensor>` 和 `vector<optional<Tensor>>`
- 过滤出已定义的张量，批量调用 `at::_to_cpu`
- 保持 undefined 张量的原始状态

**`compute_target_device`** (line 61-77)
- 从输入参数中推断目标设备
- 优先使用第一个 Tensor 参数的设备
- 其次遍历 TensorList 找到第一个有效张量的设备

**`validate_tensor_list`** (line 79-88)
- 检查张量列表是否包含至少一个已定义的张量

### Header 文件功能

**`cpu_fallback` 函数声明** (line 14-15)
```cpp
void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack, 
                  bool error_on_views = false,
                  c10::DispatchKey cpu_dispatch_key = c10::DispatchKey::CPU);
```
- `error_on_views`：控制遇到 View 操作时是报错还是警告

**`_call_fallback_fn` 模板结构** (line 19-38)
- 提供类型安全的 fallback 调用接口
- 通过算子名称动态查找 schema
- 使用 `BoxedKernelWrapper` 进行类型转换
- 支持 SymInt 模式（符号整数追踪）

---

### ROCm/Backward 相关
- 无 ROCm 特定代码
- 无 Backward 特定代码（这是运行时调度机制，不涉及梯度计算）
