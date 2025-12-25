# Resize.cpp 和 Resize.h 核心功能

## 主要职责

这两个文件实现了 PyTorch 张量的**动态内存调整和形状变更**功能，是张量操作的底层基础设施。

## 核心组件

### 1. 输出张量调整 (`resize_output` 系列)

**Resize.cpp:24-48, Resize.h:28-38**

处理带 `out` 参数的操作中输出张量的自动调整：

- `_resize_output_check()`: 检查是否需要调整，如果输出张量已有数据但形状不匹配，发出 **deprecation warning**（未来会变成错误）
- `resize_output()` / `resize_output_symint()`: 执行实际调整，CPU 张量走快速路径直接调用 `native_resize_`，其他设备通过调度

**设计意图**：逐步禁止非空输出张量的隐式调整，要求用户显式调用 `t.resize_(0)`

### 2. 存储层调整 (`resize_bytes` 系列)

#### CPU 存储 (Resize.cpp:92-107)
```cpp
void resize_bytes_cpu(StorageImpl* storage, size_t size_bytes)
```
- 分配新内存
- 拷贝旧数据到新存储（取 `min(old_size, new_size)`）
- 替换存储指针

#### Meta 设备 (Resize.cpp:162-165)
```cpp
void resize_bytes_meta(StorageImpl* storage, c10::SymInt size_bytes)
```
- 仅更新元数据，不实际分配内存（用于符号形状推导）

#### 跨设备统一接口 (Resize.cpp:278-322)
`resize_bytes_nocuda()` 处理除 CUDA 外的所有设备类型：
- **CPU/Meta**: 调用对应函数
- **XPU/HPU/MTIA**: 通过创建临时张量+`resize_`间接实现，处理数据指针未变化的特殊情况
- **PrivateUse1**: 调用自定义钩子

### 3. 张量维度调整 (`resize_impl` 系列)

**核心模板函数** (Resize.cpp:198-226):
```cpp
TensorImpl* _resize_impl_(
    TensorImpl* self,
    ArrayRef<T> size,
    at::OptionalArrayRef<T> stride,  // 可选步幅
    bool resize_storage)
```

执行流程：
1. **快速路径**：形状和步幅未变则直接返回
2. **设置元数据**：
   - 有步幅 → 调用 `set_sizes_and_strides()`
   - 无步幅 → 调用 `generic_set_sizes_contiguous()`（假设连续布局）
3. **计算所需存储大小**：考虑 `storage_offset`
4. **条件性调整存储**：`resize_storage=true` 时扩展底层内存

### 4. 用户级 API

#### `resize_()` (Resize.cpp:237-276)
```cpp
const Tensor& resize_(const Tensor& self, IntArrayRef size, 
                      std::optional<MemoryFormat> optional_memory_format)
```

- 调用 `_resize_impl_` 调整大小
- 支持 `MemoryFormat`（`Contiguous`/`ChannelsLast`，但不支持 `Preserve`）
- **确定性模式**：如果启用 `deterministicAlgorithms`，新分配的内存会被确定性填充
- 命名张量的特殊处理

#### `resize_as_()` (Resize.cpp:138-159)
```cpp
const Tensor& resize_as_(const Tensor& self, const Tensor& the_template, 
                         std::optional<MemoryFormat> optional_memory_format)
```

调整为目标张量的形状，**内存格式处理规则**：
- 无 `memory_format` 参数：
  - 形状相同 → 保持原步幅
  - 形状不同 → 设为连续
- `MemoryFormat::Preserve` → 继承模板的内存格式建议
- 其他格式 → 应用指定格式
- 传播命名维度信息

### 5. 辅助工具 (Resize.h)

#### 存储边界检查 (Resize.h:83-124)
```cpp
checkInBoundsForStorage(size, stride, storage_offset, data_type, new_storage)
```
验证 `(size, stride, offset)` 组合不会越界访问存储

#### 存储设置检查 (Resize.h:126-177)
```cpp
checkSetStorage(result, storage, storage_offset, size, stride)
```
- 验证步幅长度与维度数匹配
- 检查存储偏移非负
- 设备一致性检查（存储和张量必须在同一设备）
- 处理形状/步幅未变时的偏移验证

#### 步幅设置 (Resize.h:184-203)
```cpp
setStrided(self, size, stride, storage_offset)
```
- 拒绝负步幅
- 边界检查后设置新的形状/步幅/偏移

## 关键设计点

1. **模板化泛型实现**：通过 `T` 支持 `int64_t` 和 `c10::SymInt`（符号形状）
2. **分层架构**：存储层 → 实现层 → 用户 API 层
3. **快速路径优化**：CPU 非子类张量绕过调度直接操作
4. **内存安全**：多层边界检查、设备一致性验证
5. **兼容性处理**：为旧行为发出警告，逐步迁移到严格模式

---

### ROCm/Backward 相关内容（简要列出）
- 无 ROCm 特定代码（CUDA 调整在 `cuda/Resize.h` 中）
- 无反向传播逻辑（调整操作不需要梯度追踪）
