# OperatorEntry 核心功能分析

## 主要职责

OperatorEntry 是 PyTorch Dispatcher 系统中的核心数据结构，负责管理**单个算子**的所有注册信息和运行时分发逻辑。

## 核心组件

### 1. 算子元数据存储
- `name_`: 算子名称 (namespace + operator name)
- `schema_`: 算子的函数签名（参数、返回值类型）
- `tags_`: 算子标签（非移动端）
- `is_observed_`: 是否需要被 RecordFunction 观测

### 2. 内核注册表 (kernels_)
存储所有已注册的内核实现，按 DispatchKey 组织：
- 数据结构：`flat_hash_map<DispatchKey, AnnotatedKernelContainer>`
- 移动端：每个 DispatchKey 只保留一个内核（`array<AnnotatedKernel, 1>`）
- 非移动端：保留内核历史记录（`list<AnnotatedKernel>`），支持 Jupyter 环境的重复注册

### 3. 分发表 (dispatchTable_)
- 类型：`array<KernelFunction, num_runtime_entries>`
- 作用：运行时快速查找的核心表，索引对应 DispatchKey
- 不变式：`dispatchTable_[key] == kernels_[key].front()`

### 4. 分发键提取器 (dispatchKeyExtractor_)
- 从 Tensor 参数中提取当前操作应该使用的 DispatchKey
- 维护 fallthrough 信息

## 关键操作流程

### Schema 注册 (registerSchema)
```
1. 验证 schema 未注册
2. 检查已有内核的推断 schema 是否与注册 schema 一致
3. 更新 dispatchKeyExtractor
4. 保存 schema 和 tags
```

### 内核注册 (registerKernel)
```
1. 验证 C++ 签名一致性
2. 检查推断 schema 与注册 schema 是否匹配
3. 将内核添加到 kernels_ 容器
   - CatchAll 注册重定向到 CompositeImplicitAutograd
   - 覆盖已有内核时发出警告（Meta key 除外）
4. 调用 updateDispatchTable_ 更新分发表
```

### 分发表计算 (computeDispatchTableEntryWithDebug)

这是最复杂的逻辑，按优先级查找内核：

**优先级顺序：**
1. **直接注册**: 检查是否有直接注册到该 DispatchKey 的内核
2. **别名键回退** (按优先级):
   - (2.1) CompositeExplicitAutogradNonFunctional - 推理期所有非函数式后端
   - (2.2) CompositeExplicitAutograd - 推理期所有后端
   - (2.3) CompositeImplicitAutogradNestedTensor - 嵌套张量专用
   - (2.3) CompositeImplicitAutograd - 训练+推理通用（有条件）
     - AutogradOther + 有后端注册 → 返回 ambiguousAutogradOtherKernel
     - 无后端内核时才使用
   - (2.4) Autograd - autograd 键专用
   - (2.5) FuncTorchBatchedDecomposition - batched 键专用
3. **后端 fallback**: 使用 Dispatcher 全局注册的后端回退
4. **缺失内核**: 返回 missingKernel()，触发错误

### 分发表更新策略

**updateDispatchTableEntry_**: 单个 DispatchKey 更新
- 计算该 key 的内核
- 更新 dispatchTable 对应索引
- 更新 dispatchKeyExtractor 的 fallthrough 标记

**updateDispatchTable_**: 单个 key + 关联 keys 更新
- 处理 runtime key set 中的所有 key
- 别名键（Composite*/Autograd）注册时同步更新 Undefined
- 后端键注册时同步更新对应的 Autograd 键（见 Note [Refresh Runtime Autograd entries]）

**updateDispatchTableFull_**: 完整更新
- 遍历所有 runtime keys
- 在 OperatorEntry 构造时调用，捕获已注册的后端 fallback

## 运行时查找 (lookup)

```cpp
const KernelFunction& lookup(DispatchKeySet ks) const {
  idx = ks.getDispatchTableIndexForDispatchKeySet();
  kernel = dispatchTable_[idx];
  // 优先检查 unboxed kernel（性能优化）
  if (!kernel.isValidUnboxed() && !kernel.isValid())
    reportError(...);
  return kernel;
}
```

## C++ 签名验证

- `cpp_signature_`: 常规内核的 C++ 签名
- `sym_cpp_signature_`: SymInt 内核的 C++ 签名
- 防止同一算子的不同内核使用不兼容的 C++ 签名
- 通过 `assertSignatureIsCorrect<FuncType>()` 在编译时验证

## 错误处理

**reportError**: 
- 调用 Python 自定义错误回调（如果有）
- Undefined key: 提示没有 Tensor 参数
- 其他 key: 提示该后端不支持，列出可用后端

## 调试支持

- `dumpState()`: 打印规范状态（注册的原始信息）
- `dumpComputedTable()`: 打印计算后的分发表（所有 runtime keys）
- `checkInvariants()`: 验证内部不变式
- 每个 AnnotatedKernel 包含 `debug` 字符串（注册位置信息）

## ROCm/HIP 相关
- 支持 ROCm 后端的 DispatchKey
- ROCblas/Hipblaslt 内核注册

## Backward/Autograd 相关
- AutogradCPU, AutogradCUDA, AutogradOther 等 autograd 键
- CompositeImplicitAutograd 在训练时的特殊处理
- 后端键与 autograd 键的联动更新机制
- ambiguousAutogradOtherKernel 用于未明确指定后端的 autograd 场景
