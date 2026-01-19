这个文件提供了从外部内存数据创建 PyTorch Tensor 的接口，核心是 `TensorMaker` 类和多个 `from_blob` 重载函数。

## 核心组件

### TensorMaker 类 (lines 22-101)
流式 API 构建器，用于从外部数据构造 Tensor：

```cpp
at::Tensor tensor = at::for_blob(data, sizes)
    .strides(strides)
    .context(context, deleter)
    .options(...)
    .make_tensor();
```

**配置方法：**
- `strides()` - 设置步幅
- `storage_offset()` - 设置存储偏移
- `deleter()` - 设置数据释放函数
- `context()` - 设置上下文指针及其删除器
- `target_device()` - 设置目标设备
- `options()` - 设置 TensorOptions（dtype、device 等）
- `resizeable_storage()` - 标记存储可调整大小
- `allocator()` - 设置自定义分配器

**私有成员：**
- `data_` - 外部数据指针
- `sizes_` - Tensor 形状
- `ctx_` - 使用 unique_ptr 管理的上下文，默认使用 `noopDelete`（line 8）

### from_blob 函数重载 (lines 107-165)
提供 5 个便捷重载，覆盖常见使用场景：

1. **完整版本** (lines 107-120): `data + sizes + strides + deleter + options + target_device`
2. **带偏移** (lines 122-137): 额外支持 `storage_offset`
3. **无步幅** (lines 139-150): `data + sizes + deleter + options + target_device`
4. **基础版本** (lines 152-158): `data + sizes + strides + options`
5. **最简版本** (lines 160-165): `data + sizes + options`

## 设计特点

**内存管理：**
- 支持自定义 deleter 函数，在 Tensor 销毁时调用
- 支持 context 指针 + deleter 模式，用于传递额外清理信息
- 默认 `noopDelete` 表示不接管内存所有权

**灵活性：**
- 流式 API 避免函数重载爆炸
- 所有配置项都是可选的
- 支持从 CPU/CUDA 等不同设备的外部内存创建 Tensor

**典型使用场景：**
- 包装 NumPy 数组、C++ 数组等外部数据
- 零拷贝数据交换
- 自定义内存管理策略

---

**ROCm/Backward 相关：** 无
