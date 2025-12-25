# c10/core/StorageImpl 核心功能

## StorageImpl 类的作用

StorageImpl 是 Tensor 的底层数据缓冲区抽象，负责管理实际的内存分配。这是从 Torch7 继承来的概念，虽然社区希望移除它，但目前仍是核心组件。

**核心不变式**：一个 Storage 应该唯一拥有一个 data pointer。两个非空 data pointer 别名当且仅当它们来自同一个 Storage。

## 主要数据成员

- `DataPtr data_ptr_`：智能指针，管理实际内存和删除器
- `SymInt size_bytes_`：字节大小，支持符号整数（用于动态形状）
- `bool size_bytes_is_heap_allocated_`：标记 SymInt 是否堆分配
- `bool resizable_`：是否可调整大小
- `Allocator* allocator_`：内存分配器指针
- `impl::PyObjectSlot pyobj_slot_`：Python 对象关联槽
- `std::unique_ptr<StorageExtraMeta> extra_meta_`：额外元数据（如自定义错误消息）

## 关键访问控制机制

### 三层防护标志

1. **`throw_on_immutable_data_ptr_`**：访问 `data_ptr()`/`data()` 时抛出异常
2. **`throw_on_mutable_data_ptr_`**：访问 `mutable_data_ptr()`/`mutable_data()` 时抛出异常（用于 FakeTensor）
3. **`warn_deprecated_on_mutable_data_ptr_`**：访问可变指针时发出弃用警告

性能优化：所有检查由单个 `has_mutable_data_ptr_check_` 布尔值统一守护，避免多次条件判断。

### FakeTensor 专用错误处理

- `throwNullDataPtrError()`：在 torch.compile 中阻止 FakeTensor 访问 data_ptr
- `warnDeprecatedDataPtr()`：非编译场景下发出弃用警告
- `throw_data_ptr_access_error()`：支持自定义错误消息（通过 `extra_meta_`）

## Copy-on-Write (COW) 支持

- `is_cow()`：检查 data_ptr 是否为 COW 类型
- `maybe_materialize_cow()`：在写入前触发拷贝
- `set_data_ptr_no_materialize_cow()`：替换指针时避免不必要的物化

## 自定义 StorageImpl 扩展机制

### 注册与查找（StorageImpl.cpp）

```cpp
static std::array<StorageImplCreateHelper, at::COMPILE_TIME_MAX_DEVICE_TYPES> StorageImplCreate;
```

- `SetStorageImplCreate(DeviceType, fptr)`：为特定设备类型注册自定义构造函数
  - 仅允许 `PrivateUse1` 设备类型（白名单机制）
  - 防止重复注册
- `GetStorageImplCreate(DeviceType)`：获取已注册的构造函数

### 工厂函数

`make_storage_impl()` (c10/core/StorageImpl.cpp:77)：
1. 检查是否有设备特定的自定义构造函数
2. 如果有，调用自定义函数
3. 否则使用标准 StorageImpl 构造函数
4. 根据 data_ptr 是否为空选择不同的构造重载

## 核心方法

### 大小管理
- `nbytes()`：返回字节数（要求非堆分配的 SymInt）
- `sym_nbytes()`：返回符号整数大小
- `set_nbytes(size_t/SymInt)`：设置大小

### 数据访问
- `data()`/`data_ptr()`：只读访问（带不可变检查）
- `mutable_data()`/`mutable_data_ptr()`：可写访问（带 COW 物化 + 完整检查）
- `_mutable_data_ptr_no_checks()`：绕过所有检查的内部接口

### 内存管理
- `set_data_ptr()`：替换 data_ptr（会物化 COW）
- `UniqueStorageShareExternalPointer()`：共享外部指针（仅 use_count==1 时）
- `reset()`：清空数据和大小
- `release_resources()`：清理资源（intrusive_ptr_target 接口）

### 设备相关
- `device_type()`/`device()`：获取设备信息
- `allocator()`/`set_allocator()`：管理分配器
- `received_cuda_`：标记是否从其他进程接收 CUDA 内存

## 违反不变式的场景及后果

使用 `at::from_blob` 创建非拥有 StorageImpl 会导致：
- 删除器错误（普通删除器假设唯一所有权）
- Python deepcopy 重复数据（基于 storage 相等性而非指针相等性）
- 版本计数失效（VC 跟踪在 storage 级别）

---

**ROCm/Backward 相关**：
- 文件中未直接涉及 ROCm 特定逻辑
- `received_cuda_` 标志可能影响多进程 CUDA 内存处理，但无明确反向传播代码
