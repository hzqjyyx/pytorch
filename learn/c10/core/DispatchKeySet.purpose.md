# DispatchKeySet 核心功能理解

## 概念定位

`DispatchKeySet` 是 PyTorch 调度系统中的位集合（bitset）类型，用 64 位整数表示一组 `DispatchKey`，每个 tensor 持有自己的 `DispatchKeySet`。Dispatcher 通过将所有输入 tensor 的 keyset 按位或（OR）合并，决定调度到哪个具体实现。

## 核心设计：功能位 + 后端位的组合表示

### 为何不是 1:1 映射？

如果每个 `DispatchKey` 直接对应一个 bit，会导致组合爆炸：
- 后端数量：CPU, CUDA, XLA, ... (~12 个)
- 功能类型：Dense, Sparse, SparseCsr, Quantized, Autograd, ... (~5+ 个)
- 全组合需要：12 × 5 = 60+ 个独立 key

这会快速耗尽 64 位空间。

### 解决方案：分层表示

将 64 位划分为两个区域：

```
|<--- 后端位 --->|<------- 功能位 ------->|
 num_backends bits  num_functionality_keys bits
```

**三类 DispatchKey：**

1. **Building block keys（构建块）**
   - 后端构建块：`BackendComponent::CPUBit`, `CUDABit`, ...
   - 功能构建块：`DispatchKey::Dense`, `Sparse`, `AutogradFunctionality`, ...
   - 特点：直接对应 bitset 中的某一位

2. **Runtime keys（运行时键）**
   - 通过组合形成，如 `DispatchKey::CPU` = `Dense` 功能位 + `CPUBit` 后端位
   - 对应算子表中的实际槽位
   - 包括：
     - 每后端功能实例：`CPU`, `CUDA`, `SparseCPU`, `AutogradCUDA`, ...
     - 非可定制功能：`Functionalize`, `FuncTorchBatched`
     - 非可定制后端：`FPGA`, `MAIA`

3. **Alias keys（别名键）**
   - 如 `DispatchKey::Autograd` 映射到一组实际的 autograd 键

## 核心函数剖析

### DispatchKeySet 构造（DispatchKeySet.h:204-248）

```cpp
constexpr explicit DispatchKeySet(DispatchKey k)
```

**三种情况处理：**

1. `k == Undefined` → `repr_ = 0`

2. `k <= EndOfFunctionalityKeys`（功能位）
   ```cpp
   repr_ = 1ULL << (num_backends + static_cast<uint8_t>(k) - 1)
   ```
   只设置功能位，如 `Functionalize`, `AutogradOther`

3. `k <= EndOfRuntimeBackendKeys`（运行时键）
   ```cpp
   functionality_val = 1ULL << (num_backends + functionality_k - 1)
   backend_val = 1ULL << (backend_k - 1)
   repr_ = functionality_val + backend_val
   ```
   同时设置功能位和后端位，如 `DispatchKey::CPU` 设置 `Dense` + `CPUBit`

### 优先级查询（DispatchKeySet.h:428-435）

```cpp
DispatchKey highestPriorityTypeId() const
```

**逻辑：**
1. 找到最高优先级功能位 `highestFunctionalityKey()`
2. 如果是每后端功能（per-backend functionality），结合最高后端位返回组合键
3. 否则直接返回功能键

例如：keyset 有 `AutogradFunctionality + CPUBit + CUDABit`
- 最高功能位：`AutogradFunctionality`
- 最高后端位：`CUDABit`（假设 CUDA 优先级高于 CPU）
- 返回：`DispatchKey::AutogradCUDA`

### 算子表索引计算（DispatchKeySet.h:482-493）

```cpp
int getDispatchTableIndexForDispatchKeySet() const
```

**优化的热路径代码：**
```cpp
functionality_idx = (repr_ >> num_backends).indexOfHighestBit()
offset_and_mask = offsetsAndMasks()[functionality_idx]
backend_idx = ((repr_ & offset_and_mask.mask) >> 1).indexOfHighestBit()
return offset_and_mask.offset + backend_idx
```

**工作原理：**
- 每个功能位有一个预计算的 `(offset, mask)`
- `offset`：该功能在算子表中的起始位置
- `mask`：用于提取该功能是否需要后端索引（每后端功能 mask = full_backend_mask，其他为 0）
- 最终索引 = 功能基础偏移 + 后端偏移

### 别名展开（DispatchKeySet.cpp:64-85）

```cpp
DispatchKeySet getRuntimeDispatchKeySet(DispatchKey t)
```

**关键映射：**
```cpp
case DispatchKey::Autograd:
  return autograd_dispatch_keyset | full_backend_mask
case DispatchKey::CompositeImplicitAutograd:
  return math_dispatch_keyset  // backend + autograd 功能
case DispatchKey::CompositeExplicitAutograd:
  return backend_dispatch_keyset  // 仅 backend 功能
case DispatchKey::CompositeExplicitAutogradNonFunctional:
  return non_functional_backend_dispatch_keyset  // 排除 XLA/Lazy
```

用于算子注册时将别名键展开为实际运行时键集合。

### 迭代器（DispatchKeySet.h:524-614 & .cpp:178-257）

**复杂逻辑：**
- 遍历所有功能位（从低到高优先级）
- 对每个**每后端功能位**，再遍历所有后端位
- 非每后端功能位直接返回功能键

**状态机：**
```cpp
next_functionality_  // 下一个要检查的功能位
next_backend_        // 下一个要检查的后端位
current_dispatchkey_idx_
current_backendcomponent_idx_
```

**operator++ 三种情况：**
1. 当前功能是每后端，但没后端位 → 跳过此功能，递增
2. 当前功能是每后端，还有其他后端 → 保持功能位，递增后端位
3. 当前功能是每后端，后端耗尽 OR 非每后端功能 → 递增功能位，重置后端位

## 关键预定义 KeySet（DispatchKeySet.cpp & .h）

### backend_dispatch_keyset (line 9-10 .cpp)
```cpp
autogradother_backends | DispatchKeySet(DispatchKey::Dense)
```
所有映射到后端的键，`CompositeExplicitAutograd` 映射到此集合。

### non_functional_backend_dispatch_keyset (line 26-32 .cpp)
```cpp
backend_dispatch_keyset
  .remove(DispatchKey::Sparse)
  .remove_backend(BackendComponent::XLABit)
  .remove_backend(BackendComponent::LazyBit)
```
排除 XLA/Lazy 的后端集合，用于非函数式分解（会产生别名的分解不应发送给函数式后端）。

### math_dispatch_keyset (line 48-57 .cpp)
```cpp
backend_dispatch_keyset | autograd_dispatch_keyset 
  | DispatchKey::NestedTensor | DispatchKey::Functionalize
```
`CompositeImplicitAutograd` 映射到此集合，包含后端 + autograd 功能。

### autograd_dispatch_keyset (line 650-654 .h)
```cpp
{AutogradFunctionality, AutogradOther, AutogradNestedTensor}
```
注意：**不包含后端位**（见 Note [autograd_dispatch_keyset Does Not Include Backend Bits]）。原因：从 keyset 中移除键只能移除功能位，不能移除后端位（见 Note [Removing keys from DispatchKeySet Only Affects Functionality Keys]）。

## 辅助工具函数

### getBackendKeySetFromAutograd (DispatchKeySet.cpp:114-148)
```cpp
AutogradCPU → DispatchKeySet(DispatchKey::CPU)
AutogradCUDA → DispatchKeySet(DispatchKey::CUDA)
AutogradNestedTensor → NestedTensor | full_backend_mask
```

### getAutogradRelatedKeySetFromBackend (DispatchKeySet.h:830-861)
反向映射：
```cpp
CPUBit → ADInplaceOrView | AutogradCPU
CUDABit → ADInplaceOrView | AutogradCUDA
```

### isBackendDispatchKey (DispatchKeySet.cpp:34-43)
判断是否为后端调度键：
```cpp
t != Undefined 
&& !isAliasDispatchKey(t)
&& t != NestedTensor  // 显式排除
&& backend_dispatch_keyset.has(t)
```

---

## ROCm/Backward 相关内容
- ROCm: HIP 现在有独立后端位，可以有自己的 `AutogradHIP` 键（不再映射到 `AutogradOther`）
- Backward: `AutogradFunctionality` 功能位用于标识需要自动微分的操作，`AutogradCPU/CUDA/XLA` 等是其每后端实例
