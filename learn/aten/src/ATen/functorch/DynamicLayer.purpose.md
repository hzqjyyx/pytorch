# DynamicLayer 核心功能分析

这两个文件实现了 functorch 的**解释器堆栈（interpreter stack）**，用于管理函数式变换（functional transforms）的组合和执行。

## 核心概念

**DynamicLayer** 是 functorch 变换的容器，每个 DynamicLayer 持有一个 Interpreter 对象，代表一种具体的变换类型：

- `TransformType::Vmap` - 向量化映射
- `TransformType::Grad` - 梯度计算
- `TransformType::Jvp` - 前向模式自动微分
- `TransformType::Functionalize` - 函数化（去除原地操作）

## 堆栈管理机制

### TLS（Thread Local Storage）实现
DynamicLayer.cpp:85-142

使用线程局部存储管理变换堆栈：
- `FuncTorchTLS` 类继承自 `FuncTorchTLSBase`，存储 `dynamicLayerStack`
- `getRawFunctorchTLS()` 懒初始化并返回当前线程的 TLS 对象
- 支持深拷贝以便跨线程传递状态

### 堆栈操作 API
DynamicLayer.cpp:219-267

- `pushDynamicLayer()` - 压入新变换层，返回 layerId
- `popDynamicLayer()` - 弹出栈顶层
- `initAndPushDynamicLayer()` - 创建并压入新层，同时设置生命周期标记
- `popDynamicLayerAndDeleteMetadata()` - 弹出并标记为失效

堆栈为空时自动禁用 `FuncTorchDynamicLayerFrontMode` 和 `FuncTorchDynamicLayerBackMode` dispatch keys。

## 调度机制（Dispatcher Integration）

### 双模式调度
DynamicLayer.cpp:406-430, 475-493

通过两个 dispatch keys 实现变换的嵌套组合：

**FrontMode** (`dynamicLayerFrontFallback`)：
- 保存当前 LocalDispatchKeySet
- 解包已失效的 GradWrapper
- 调用栈顶 Interpreter 的 `process()` 方法处理操作

**BackMode** (`dynamicLayerBackFallback`)：
- 恢复保存的 LocalDispatchKeySet
- 临时移除栈顶层（WithoutTop guard）
- 调用 `sendToNextInterpreter()` 将操作传递给下一层

特殊处理：`lift_fresh` 和 `alias` 操作使用 `dynamicLayerBackGradSpecialCase` 避免别名检查。

## 生命周期管理

### Life Handles
DynamicLayer.cpp:168-178

每个变换层有一个 `shared_ptr<bool>` 作为生命周期句柄：
- TensorWrapper 持有该句柄的引用
- 当变换作用域结束时，句柄被标记为 false
- 逃逸的 Tensor 通过查询句柄判断变换是否仍然活跃

### 死亡检测与解包
DynamicLayer.cpp:277-294

- `isDeadTensorWrapper()` - 检查 Tensor 是否包含失效的 wrapper
- `unwrapIfDead()` - 如果 wrapper 已失效则解包返回内部值

## 工具函数

### Tensor 批量处理
DynamicLayer.cpp:296-346

`foreachTensorInplace()` 和 `foreachTensorInplaceWithFlag()`：
- 遍历操作参数中的所有 Tensor（包括嵌套在 List 中的）
- 对每个 Tensor 应用变换函数
- 处理 `Tensor?[]`、`TensorList` 等复杂类型

### Schema 分析
DynamicLayer.cpp:361-389

- `isInplaceOp()` - 判断操作是否为原地操作（首参数可变且被返回）
- `findAliasedOutput()` - 查找与输入参数存在别名关系的输出索引

## 安全检查

DynamicLayer.cpp:95-126

在 functorch 变换活跃时禁止某些操作：
- `checkSupportsCppAutogradFunction()` - 禁止 C++ autograd::Function
- `checkSupportsInplaceRequiresGrad()` - 默认禁止 `requires_grad_()`
- `checkSupportsRetainGrad()` - 禁止 `retain_grad()`

可通过 `setInplaceRequiresGradAllowed()` 等 API 临时启用。

---

**忽略内容简述：**
- ROCm 特定实现细节
- Backward pass 的梯度传播逻辑
