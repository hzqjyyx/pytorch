## SizesAndStrides 类的主要功能

`SizesAndStrides` 是一个针对 PyTorch 张量的尺寸和步幅信息的优化容器类。它采用混合存储策略来提高性能。

### 核心设计

- **混合存储方案**：对于维度数 ≤ 5 的情况，使用内联存储（inline storage）；超过 5 维则使用堆外存储（out-of-line storage）
- **内存布局**：
  - 1 个 `size_t` 存储当前维度数
  - 内联时：10 个 int64_t（前 5 个存尺寸，后 5 个存步幅）
  - 堆外时：指向动态分配的数组的指针

### 主要方法

**数据访问**：
- `sizes_data()` / `strides_data()`：获取尺寸/步幅数据指针
- `size_at(idx)` / `stride_at(idx)`：带边界检查的访问
- `sizes_arrayref()` / `strides_arrayref()`：返回 `IntArrayRef` 视图

**数据设置**：
- `set_sizes(IntArrayRef)`：批量设置尺寸
- `set_strides(IntArrayRef)`：批量设置步幅

**内存管理**：
- `resize(newSize)`：快速路径处理内联情况，慢速路径处理内联↔堆外转换
- `resizeSlowPath()`：处理以下三种场景：
  - 堆外 → 内联（收缩）
  - 内联 → 堆外（扩张）
  - 堆外 → 堆外（重新分配）

**生命周期**：
- 复制构造/赋值：分别处理内联和堆外情况
- 移动构造/赋值：高效转移所有权，源对象清零
- 析构：释放堆外存储

### 关键优化点

- **`C10_LIKELY` 宏**：标记内联路径为热路径
- **直接 malloc/free/realloc**：避免高层抽象开销
- **`memmove` 替代 memcpy**：在堆外调整时处理内存重叠

### Bullet Points

- 张量尺寸和步幅的高效容器，优化小维度张量的访问性能
- 5 维以内使用栈上存储，超过 5 维使用动态分配
- 提供尺寸和步幅的随机访问、迭代器接口和数组视图
- `resize()` 支持内联↔堆外自动转换，需要复杂的内存管理逻辑
- 完整的复制/移动语义支持
- 使用低层 C 内存操作函数确保最小开销
