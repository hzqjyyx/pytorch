# PyTorch Dispatcher 核心机制

## 主要功能

Dispatcher 是 PyTorch 的动态调度系统，负责在运行时根据张量类型、设备和其他上下文信息，将算子调用路由到正确的内核实现。

## 核心组件

### 1. Dispatcher 类（单例）

**职责**：全局算子注册表和调度协调器

**关键数据结构**：
- `operators_`: 存储所有已注册算子的链表
- `operatorLookupTable_`: 算子名称到 OperatorHandle 的哈希表（使用 LeftRight 实现无锁读取）
- `backendFallbackKernels_`: 每个 DispatchKey 的后备内核数组
- `libraries_`: 命名空间到调试信息的映射
- `listeners_`: 算子注册/注销的监听器列表

**核心方法**：

```cpp
// 算子查找
findOp()           // 按名称查找算子
findSchema()       // 查找带 schema 的算子
findSchemaOrThrow() // 查找失败则抛异常

// 算子调用
call()             // 主调用路径（模板化，支持类型安全调用）
callBoxed()        // boxed 调用（通过 IValue Stack）
redispatch()       // 内核中的重新调度

// 注册管理
registerDef()      // 注册算子 schema
registerImpl()     // 注册特定 DispatchKey 的实现
registerFallback() // 注册后备内核
registerLibrary()  // 注册 TORCH_LIBRARY 命名空间
```

### 2. OperatorHandle

**职责**：算子的不透明句柄，提供访问算子元数据和调用的接口

**关键成员**：
- `operatorDef_`: 指向 OperatorDef 的指针（快速访问）
- `operatorIterator_`: 链表迭代器（用于快速清理）

**功能**：
```cpp
operator_name()              // 获取算子名称
schema()                     // 获取函数签名
hasKernelForDispatchKey()    // 检查是否有特定 key 的内核
typed<FuncType>()            // 转换为类型化句柄
callBoxed()                  // 执行 boxed 调用
```

### 3. TypedOperatorHandle<Return(Args...)>

**职责**：类型安全的算子句柄，支持编译时类型检查

**核心方法**：
```cpp
call(Args... args)           // 类型安全的直接调用
redispatch(DispatchKeySet, Args...) // 类型安全的重新调度
```

## 调度流程

### 主调用路径 `Dispatcher::call()`

```
1. 提取 DispatchKeySet
   └─ 从参数中推断（张量设备、dtype、autograd 状态等）

2. 查找内核
   └─ op.operatorDef_->op.lookup(dispatchKeySet)
   └─ 按优先级查找第一个匹配的内核

3. 性能分析（可选）
   └─ 检查 RecordFunction callbacks
   └─ 如果启用，调用 callWithDispatchKeySlowPath()

4. 执行内核
   └─ kernel.call<Return, Args...>(op, dispatchKeySet, args...)
```

### Boxed 调用路径 `callBoxed()`

用于 Python 绑定和动态调用：
- 参数通过 `Stack*`（IValue 向量）传递
- 支持运行时类型检查
- 性能开销相对较大

### Redispatch 机制

内核可以调用 `redispatch()` 将控制权传递给下一个 DispatchKey：
- 不触发新的 RecordFunction
- 跳过当前 key，查找 DispatchKeySet 中下一个优先级的内核
- 用于实现层叠功能（如 autograd → 设备 → 实现）

## 注册机制

### 生命周期管理

所有注册返回 `RegistrationHandleRAII`：
- 析构时自动注销
- 使用 `guard_` 检查 Dispatcher 是否仍然存活
- 支持静态和动态注册

### OperatorDef 引用计数

```cpp
def_count           // 仅 def() 调用计数
def_and_impl_count  // def() + impl() 总计数
```

当 `def_count` 降至 0：触发注销监听器
当 `def_and_impl_count` 降至 0：完全删除算子

### Python 模块存根（PyStub）

机制：将算子与 Python 模块关联
- `registerPythonModule()`: 注册 Python 实现位置
- `throwIfHasPythonModule()`: 检查并提示用户导入模块
- 用于惰性加载和更好的错误消息

## 性能关键设计

### 1. 内联调用路径
```cpp
C10_ALWAYS_INLINE_UNLESS_MOBILE Return Dispatcher::call(...)
```
- 移动端避免代码膨胀
- 桌面端完全内联以减少开销

### 2. 单例优化
```cpp
static Dispatcher& s = realSingleton();  // 缓存引用
```
- 避免每次调用的函数开销

### 3. LeftRight 并发控制
- 读操作无锁（写时拷贝语义）
- 写操作串行化（mutex 保护）
- 适合读多写少场景

### 4. 快速查找
- `operatorDef_` 直接指针避免迭代器解引用开销
- `operatorIterator_` 用于 O(1) 清理

## 调试和追踪

### 环境变量
```cpp
TORCH_SHOW_DISPATCH_TRACE=1  // 打印调度决策
```

### 嵌套追踪
```cpp
thread_local dispatch_trace_nesting_value_  // 跟踪调用深度
DispatchTraceNestingGuard                    // RAII 嵌套管理
```

输出示例：
```
[call] op=[aten::add], key=[CPU]
  [redispatch] op=[aten::add], key=[AutogradCPU]
```

## 多解释器支持

### 等待机制
```cpp
waitForDef()   // 等待 schema 注册
waitForImpl()  // 等待实现注册
```

- 用于 torchdeploy/multipy 多解释器场景
- 通过 `cond_var_` 条件变量同步
- 2 秒超时保护

### 守卫机制
```cpp
struct Guard {
  std::atomic<bool> alive;  // Dispatcher 是否存活
  std::mutex mutex;         // 保护并发修改
}
```

- 回调函数持有 `shared_ptr<Guard>`
- 在 Dispatcher 销毁后安全失败

## 性能分析集成

### RecordFunction 集成
```cpp
callWithDispatchKeySlowPath()  // 慢路径，包含性能分析
```

流程：
1. 检查是否需要输入（`guard.needsInputs()`）
2. Box 参数到 IValue 数组
3. 调用 `runRecordFunction()` 记录事件
4. 执行内核
5. 如果需要输出，捕获并记录返回值

### Autograd 序列号
```cpp
sequenceNumberForRunningRecordFunction()
```
- 关联前向传播和反向传播事件
- 仅当 DispatchKeySet 包含 autograd keys 时记录

---

## 简要列出的其他内容

**ROCm 相关**：
- `GemmHipblaslt.h` / `GemmRocblas.h` - AMD GPU GEMM 内核
- ROCm 特定的可调优算子实现

**Backward/Autograd 相关**：
- 自动微分通过 DispatchKey::Autograd* 实现
- `sequenceNumberForRunningRecordFunction()` 关联前向/反向
- Autograd keys 在 DispatchKeySet 中优先级较高
