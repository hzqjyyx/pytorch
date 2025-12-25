## 核心功能

这两个文件定义了 PyTorch 的 **dispatch system（调度系统）** 的核心概念，用于实现运行时的多态分发机制。

### DispatchKey - 调度键

`DispatchKey` 是一个枚举类型，标识了算子可能被分发到的不同"层级"或"处理器"。每个 `DispatchKey` 代表一个特定的功能或后端：

**分类体系：**

1. **Backend Components（后端组件）** - 硬件/设备后端
   - `CPUBit`, `CUDABit`, `XPUBit`, `IPUBit`, `MPSBit`, `HPUBit`, `VEBit`, `MTIABit` 等
   - `XLABit`, `LazyBit`, `MetaBit`
   - `PrivateUse1/2/3Bit` - 用户自定义后端

2. **Functionality Keys（功能键）** - 张量特性或操作模式
   - **布局相关**: `Dense`, `Sparse`, `SparseCsr`, `NestedTensor`, `Quantized`
   - **模式/包装器**: `Python`, `Fake`, `Functionalize`, `Tracer`
   - **变换**: `Conjugate`, `Negative`, `ZeroTensor`
   - **批处理**: `Batched`, `VmapMode`, `FuncTorchBatched`
   - **其他**: `BackendSelect`, `Named`, `DeferredInit`

3. **Per-Backend Functionality Keys（每后端功能键）**
   - 某些功能可以针对每个后端定制实现
   - 如 `CPU`, `CUDA`, `SparseCPU`, `SparseCUDA`, `QuantizedCPU` 等
   - 映射关系：`toRuntimePerBackendFunctionalityKey(Dense, CUDABit) → CUDA`

4. **Alias Keys（别名键）** - 映射到多个运行时键
   - `Autograd` - 映射到所有后端的 autograd 键
   - `CompositeImplicitAutograd` - 组合操作，隐式支持自动微分
   - `CompositeExplicitAutograd` - 组合操作，显式处理自动微分
   - `FuncTorchBatchedDecomposition`

### BackendComponent - 后端位组件

表示 `DispatchKeySet` 中的后端位，用于与功能键组合确定具体的调度目标。

### 核心函数

**toString()** - 序列化
- `toString(DispatchKey)` - 将调度键转为字符串，支持动态组合键的命名
- `toString(BackendComponent)` - 后端组件转字符串

**parseDispatchKey()** - 反序列化
- 从字符串解析出 `DispatchKey`
- 支持 `PrivateUse1` 自定义后端的运行时名称替换（通过正则表达式）

**类型转换**
- `toBackendComponent(DispatchKey)` - 提取调度键的后端部分
- `toBackendComponent(DeviceType)` - 设备类型转后端组件
- `toFunctionalityKey(DispatchKey)` - 提取调度键的功能部分
- `toRuntimePerBackendFunctionalityKey()` - 组合功能键和后端生成运行时键

**getAutogradKeyFromBackend()** - 获取后端对应的 autograd 键

### 设计要点

**位图编码**
- `DispatchKeySet` 使用 64 位位图
- 低 ~12 位：后端组件（最多 16 个后端）
- 高位：功能键
- 优先级通过位位置隐式定义（高位优先）

**Per-Backend 功能机制**
- `Dense`, `Sparse`, `SparseCsr`, `Quantized`, `NestedTensor`, `AutogradFunctionality` 支持每后端定制
- 在 `DispatchKeySet` 中只占一个位，但在运行时算子表中占 `(后端数量)` 个槽位
- 范围标记：`StartOfDenseBackends` → `EndOfDenseBackends` 等

**顺序依赖**
- 后端功能键必须按照 `BackendComponent` 的枚举顺序排列
- Meta 后端必须在最后（TLS 触发 meta kernel）
- `EndOfRuntimeBackendKeys` 标记运行时键结束

**别名键机制**
- 不直接调用，仅在构建算子表时填充
- 优先级低于运行时键
- 用于简化注册（如注册到 `Autograd` 自动填充到所有 `Autograd*` 键）

---

### 其他提及内容（简略）

- **Autograd 相关**: `ADInplaceOrView`, `AutogradOther`, `AutogradFunctionality`, `AutogradNestedTensor`, `Autograd*` 系列（CPU/CUDA/XLA/等）
- **Autocast 相关**: `AutocastCPU`, `AutocastCUDA`, `AutocastXLA`, `AutocastMPS`, `AutocastMTIA`, `AutocastXPU`, `AutocastIPU`, `AutocastHPU`, `AutocastPrivateUse1`
- **FuncTorch 相关**: `FuncTorchBatched`, `FuncTorchVmapMode`, `FuncTorchGradWrapper`, `FuncTorchDynamicLayerFrontMode`, `FuncTorchDynamicLayerBackMode`
- **测试键**: `TESTING_ONLY_GenericWrapper`, `TESTING_ONLY_GenericMode`
- **其他**: `PythonDispatcher`, `PreDispatch`, `PythonTLSSnapshot`, `CustomRNGKeyId`, `MkldnnCPU`
