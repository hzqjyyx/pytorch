# RecordFunction 核心功能分析

这是 PyTorch 的性能分析和观察者(observer)框架，用于在运算执行时插入回调函数，支持 profiling、tracing 等功能。

## 核心架构

### 三层回调管理系统

**1. GlobalCallbackManager (全局单例)**
- 管理进程级全局回调，所有线程共享
- 使用版本号(`version_`) + 互斥锁实现线程安全
- 任何修改(add/remove/enable/disable)都会递增版本号
- 典型用途：后台监控、全局 profiling

**2. LocalCallbackManager (线程局部单例)**
- 每个线程独立维护的回调管理器
- 缓存全局回调的快照，通过版本号判断是否需要重建
- 合并全局回调 + 线程局部回调
- 典型用途：特定代码段的 profiling/tracing

**3. CacheEntry (按 RecordScope 缓存)**
- 为每种 scope (FUNCTION, BACKWARD_FUNCTION, TORCHSCRIPT_FUNCTION 等) 维护独立缓存
- 存储当前应执行的回调及其采样状态
- 实现高效的采样机制

### RecordScope 类型

定义在 `record_function.h:24-45`，包括：
- `FUNCTION`: C10/ATen 算子、autograd 节点
- `TORCHSCRIPT_FUNCTION`: TorchScript 函数/方法
- `USER_SCOPE`: 用户自定义 scope
- `STATIC_RUNTIME_OP/MODEL`: 静态运行时
- `CUSTOM_CLASS`, `BUILD_FEATURE`, `KERNEL_FUNCTION_DTYPE`, `LITE_INTERPRETER` 等

## 采样机制 (Sampling)

### 几何分布采样优化

传统方法每次调用都需要随机数生成，成本高。优化策略：

**核心思想** (`record_function.cpp:85-100`)：
- Bernoulli 分布的失败次数服从几何分布
- 预先采样下次触发需要的调用次数 (`tries_left_`)
- 每次调用只需递减计数器，无需生成随机数

**实现细节**：
```cpp
// CacheEntry::sampleTries() - 行357-363
std::geometric_distribution<int>(p)(*generator_) + 1
```

**Scope 分片** (`record_function.cpp:94-100`)：
- 为每个 RecordScope 维护独立计数器
- 使用 `sampling_countdown_` 统一管理，值为所有回调中最小的 `tries_left_`
- 达到阈值时触发 `rebuildActiveCallbacks()`

### 采样状态转换

在 `CacheEntry::rebuildActiveCallbacks()` (行323-355)：
- `tries_left_ < 0`: 非采样回调，始终执行
- `tries_left_ == 0`: 采样命中，执行并设置 `sampling_countdown_ = 1`
- `tries_left_ > 0`: 等待中，更新 `sampling_countdown_` 为最小值

## RecordFunction 生命周期

### 初始化

1. 构造函数接收 `RecordScope` 或 `StepCallbacks`
2. 通过 `getStepCallbacks(scope)` 获取当前激活的回调列表
3. 检查全局版本号，必要时重建缓存
4. 如果需要 ID 追踪，分配唯一 handle

### 执行流程

**Before 阶段** (`record_function.cpp:727-763`)：
```cpp
before(name/schema, sequence_nr)
  → 存储函数名/schema
  → 检查是否为 NCCL metadata 收集
  → runStartCallbacks()
    → tryRunCallback<is_start=true>() 依次执行
    → 返回 ObserverContext 存入 ctx_
  → invalidateInputs() (防止 callback 外访问)
```

**End 阶段** (`record_function.cpp:553-561`)：
```cpp
end() 或 ~RecordFunction()
  → 检查 called_start_callbacks_
  → tryRunCallback<is_start=false>() 传入 ObserverContext
  → 清空 callbacks_
```

### 输入输出管理

- `inputs_` / `kwinputs_`: 仅在 start callback 内有效
- `inputs_valid_` (debug 模式): 防止非法访问
- `outputs_`: 通过 `setOutputs()` 设置，在 end callback 可用
- `needs_inputs_` / `needs_outputs_`: 标志位控制数据收集开销

## 关键特性

### 线程安全

- 全局回调：读用原子操作(`version_.load`)，写用互斥锁
- 线程局部回调：无需锁(thread_local 存储)
- 版本号机制实现乐观并发控制

### 异步支持

- `is_async_` 标志支持异步操作
- `threadId()` 记录启动线程，end callback 可在不同线程执行
- `fwd_thread_id_` 关联前向和反向函数

### Schema 支持

`fn_` 使用 `std::variant<std::string, schema_ref_t>`:
- 普通字符串：用户自定义 scope
- FunctionSchema 引用：算子调用，可获取完整签名信息
  - `operator_name()` 返回 OperatorName
  - `operator_schema()` 返回完整 schema (拷贝开销大)
  - `num_inputs()` / `num_outputs()` 自动从 schema 推断

### 宏便利接口

```cpp
RECORD_FUNCTION(fn, inputs)                    // 默认 FUNCTION scope
RECORD_USER_SCOPE(fn)                          // 用户 scope
RECORD_TORCHSCRIPT_FUNCTION(mn, inputs)       // TorchScript
RECORD_FUNCTION_WITH_INPUTS_OUTPUTS(...)      // 同时记录输入输出
```

## API 摘要

### 回调注册
- `addGlobalCallback()` / `addThreadLocalCallback()`: 返回 CallbackHandle
- `removeCallback()` / `disableCallback()` / `reenableCallback()`: 通过 handle 管理
- `clearCallbacks()`: 清除所有

### 控制开关
- `enableRecordFunction(bool)`: 线程局部开关
- `RecordFunctionGuard`: RAII 风格临时启用/禁用
- `DisableRecordFunctionGuard`: 便捷禁用守卫

### 查询状态
- `hasCallbacks()` / `hasGlobalCallbacks()` / `hasThreadLocalCallbacks()`
- `isRecordFunctionEnabled()`

### 分布式 Profiling
- `setDefaultNodeId()` / `getDefaultNodeId()`: 节点 ID
- `is_nccl_meta_`: NCCL 元数据收集标志

---

**ROCm 相关**：代码中提及 `hipblaslt`, `rocblas` (在 tunable 目录)，但主文件无 ROCm 特定逻辑

**Backward 相关**：
- `fwd_thread_id_` 字段关联前向/反向函数
- `RecordScope::BACKWARD_FUNCTION` 枚举值
- `sequence_nr_` 配合线程 ID 用于前后向关联
