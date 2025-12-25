# Interpreter.cpp/h 主要功能

## 核心概念

**Interpreter Stack (解释器栈)**: functorch 的调度系统基于一个解释器栈，历史上称为 "DynamicLayerStack"。每个解释器负责读取传入的算子并按特定语义执行。

## 支持的变换类型 (TransformType)

- **Vmap**: 向量化映射，批处理操作
- **Grad**: 反向模式自动微分 (vjp)
- **Jvp**: 正向模式自动微分
- **Functionalize**: 函数化变换

## Interpreter 结构

采用 **非虚函数的"子类化"设计**，通过 `std::variant` 存储不同类型的元数据：

```cpp
typedef std::variant<
  int64_t,
  GradInterpreterMeta,
  JvpInterpreterMeta,
  VmapInterpreterMeta,
  FunctionalizeInterpreterMeta
> InterpreterMeta;
```

每种解释器有对应的元数据结构：
- `VmapInterpreterMeta`: 存储 `batchSize_` 和 `randomness_`
- `GradInterpreterMeta`: 存储 `prevGradMode_`
- `JvpInterpreterMeta`: 存储 `prevFwdGradMode_`
- `FunctionalizeInterpreterMeta`: 存储 `functionalizeAddBackViews_`

## 核心方法

### 1. process(op, stack)
根据解释器类型分发到对应的 `processImpl`，负责：
- 接收算子句柄和参数栈
- 按解释器语义执行操作（如 VmapInterpreter 调用批处理规则）

### 2. sendToNextInterpreter(op, stack, grad_special_case)
将处理后的操作发送到栈中的下一个解释器

### 3. INTERPRETER_DISPATCH 宏
通过 `switch-case` 实现类型分发，避免虚函数开销

## DispatchKey 管理

### keysForEnteringDynamicLayer(key)
返回进入动态层时需要的 DispatchKey 集合：
- Vmap: `FuncTorchBatched`, `BatchedNestedTensor`
- Grad/Jvp: autograd 相关的 key + `ADInplaceOrView`
- Functionalize: `Functionalize` key

### keysToExcludeWhenEnteringDynamicLayer(key)
计算进入动态层时需要排除的 key，确保只有当前层的 key 被激活

### setup_dispatch_key_tls(key, also_include)
设置线程本地存储 (TLS) 的 dispatch key 集合，通过修改 `excluded_` 和 `included_` 来控制调度行为

## 生命周期管理

- `is_alive_`: 共享指针跟踪解释器是否活跃（是否在变换的执行过程中）
- `savedLocalDispatchKeySet_`: 保存和恢复进入/退出层时的 dispatch key 状态

## 辅助功能

- `sanityCheckStack()`: 验证栈中的 tensor 没有被 wrapper 或 batched impl 包装
- `foreachTensorInplace()`: 对栈中指定范围的 tensor 应用函数
- `get_all_dynlayer_keyset()`: 获取所有动态层相关的 dispatch key，排除 autocast 和 VmapMode

## 随机性控制 (RandomnessType)

仅用于 Vmap：
- **Error**: 调用随机函数时报错
- **Same**: 跨批次的随机性相同
- **Different**: 跨批次的随机性不同

---

**ROCm/Backward 相关**: 无直接涉及，主要是调度框架的通用实现
