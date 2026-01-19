`OpaqueTensorImpl.h` 定义了一个特殊的 TensorImpl 子类，用于表示"不透明"的张量——这些张量不支持传统的步长（stride）访问和指针算术操作。

## 核心设计理念

这是一个模板类 `OpaqueTensorImpl<OpaqueHandle>`，其中 `OpaqueHandle` 是一个泛型类型，用于存储后端特定的张量句柄。这种设计允许 PyTorch 支持那些内存布局不符合标准连续内存模型的张量实现（例如某些硬件加速器的专有格式）。

## 主要限制

**禁用的操作：**
- `set_size()` - 不允许修改维度大小
- `set_stride()` - 不允许设置步长
- `set_storage_offset()` - 不允许设置存储偏移
- `data()` - 不支持直接数据指针访问（通过 `set_storage_access_should_throw()` 实现）
- `resize_()` 等元数据修改操作

这些限制确保了外部代码无法对不透明张量进行不安全的内存操作。

## 关键实现

**构造函数（23-38行）：**
```cpp
OpaqueTensorImpl(
    at::DispatchKeySet key_set,
    const caffe2::TypeMeta data_type,
    c10::Device device,
    OpaqueHandle opaque_handle,
    c10::IntArrayRef sizes,
    bool is_non_overlapping_and_dense = true)
```
- 接收后端句柄 `opaque_handle`
- 设置 `CustomStrides` 策略
- 标记存储访问应抛出异常
- 只存储 sizes，不存储 strides

**浅拷贝机制（73-113行）：**
实现了两个 `shallow_copy_and_detach()` 重载，用于创建共享相同 `opaque_handle_` 的新 TensorImpl 实例。这对于 autograd 和张量视图操作至关重要。

**句柄访问（133-139行）：**
- `opaque_handle()` - 只读访问
- `unsafe_opaque_handle()` - 可变访问（标记为 unsafe 提醒使用者注意）

## 使用场景

适用于需要将外部数据结构（如 MKL-DNN 的 memory object、Metal 的 buffer、自定义硬件的张量表示）包装为 PyTorch Tensor 的情况，同时保持 PyTorch 的类型系统和调度机制的完整性。

---

**ROCm/Backward 相关：**
- 无直接 ROCm 特定代码
- 无 backward 相关实现（这是前向张量表示层）
- 版本计数器（`version_counter`）用于 autograd 追踪，但具体 backward 逻辑在其他层实现
