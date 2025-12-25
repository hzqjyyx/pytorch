# SymbolicShapeMeta 核心功能解析

## 主要职责

管理张量的符号化形状元数据，支持动态形状（symbolic shapes）下的延迟计算和线程安全的属性缓存。

## 核心数据结构

### 基础元数据（不可变部分）
- `sizes_`: 张量维度大小（SymDimVector）
- `strides_`: 张量步长（SymDimVector）
- `storage_offset_`: 存储偏移量（SymInt）
- `strides_valid_`: 步长是否有效（如稀疏张量无步长）

### 派生属性（延迟计算 + 缓存）
使用 `mutable` + `std::mutex` + `std::atomic` 实现线程安全的懒初始化：

- `numel_`: 元素总数
- `is_contiguous_`: 是否连续存储
- `is_channels_last_contiguous_`: 是否 channels-last 连续（NHWC）
- `is_channels_last_3d_contiguous_`: 是否 3D channels-last 连续（NDHWC）
- `is_channels_last_`: 是否 channels-last 布局
- `is_channels_last_3d_`: 是否 3D channels-last 布局
- `is_non_overlapping_and_dense_`: 是否无重叠且稠密

## 关键机制

### 1. 延迟计算模式
```cpp
const SymInt& numel() const {
    if (!has_numel()) {  // 检查 atomic 标志位
        init_numel();     // 首次访问时计算并缓存
    }
    return numel_;
}
```

### 2. 线程安全策略
- **读取**：原子操作检查 `available_` 标志位
- **写入**：`std::scoped_lock` 保护可变成员的初始化
- **拷贝构造**：锁住源对象的 `mutables_` 后复制派生属性（c10/core/SymbolicShapeMeta.cpp:9-27）

### 3. 符号化计算优化
`normalize_sym_sizes_strides()` (c10/core/SymbolicShapeMeta.cpp:30-73) 实现智能调度：
- 查找 heap-allocated 的 SymNode 作为基础节点
- 如果所有值都有 hint（具体值），退化到普通计算
- 否则统一用 SymNode 进行符号计算

宏定义的两种计算模式：
- `DEFINE_EAGER_SYMBOOL_COMPUTE`: 直接用 fallback 函数计算（如 contiguity）
- `DEFINE_SYMBOOL_COMPUTE`: 优先用 SymNode 的方法（如 `is_non_overlapping_and_dense`），失败则 fallback

### 4. 短路优化
特殊维度下的计算利用已缓存属性加速（c10/core/SymbolicShapeMeta.cpp:129-185）：
```cpp
// dim=4 时检查 non-overlapping-dense
if (definitely_true(is_contiguous())) return true;
if (definitely_true(is_channels_last_contiguous())) return true;
return is_contiguous() | is_channels_last_contiguous() | compute_non_overlapping_and_dense();
```

## 维度特化逻辑

- **dim=4**: 检查 NCHW 或 NHWC 连续性
- **dim=5**: 
  - channels-last-3d（NCDHW → NDHWC）
  - channels-last-2d（5D 张量按 2D 规则判断，c10/core/SymbolicShapeMeta.cpp:150-157）
  - 三种连续性或运算（c10/core/SymbolicShapeMeta.cpp:166-178）
- **其他维度**: 仅检查标准连续性

## 性能优化设计

1. **原子标志位**: 避免每次都加锁检查
2. **双检查锁**: setter 函数先无锁检查 `has_foo()` 再加锁设置（c10/core/SymbolicShapeMeta.cpp:187-243）
3. **刷新接口**: `refresh_numel()` 和 `refresh_contiguous()` 批量重置标志位而不加锁
4. **假设接口**: `assume_*()` 方法允许非 const 环境直接设置属性跳过计算

## 不支持的操作

- 移动构造/赋值：禁用（避免处理原子变量和互斥锁的移动语义）
- 拷贝赋值：禁用（防止运行时意外的深拷贝开销）

---

**其他特性简述**：
- 支持通过 `assume_*()` 方法注入已知属性（如从 Python 传入的确定性信息）
- 注释提到避免符号计算过慢的问题（c10/core/SymbolicShapeMeta.cpp:122-127）
- 所有计算函数都返回 `SymBool` 而非 `bool`，保持符号化系统的完整性
