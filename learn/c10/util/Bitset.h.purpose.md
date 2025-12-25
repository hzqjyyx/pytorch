这个文件定义了 `c10::utils::bitset` 类，一个64位的简化位集实现。主要特点：

**核心功能：**
- 基于 `long long int`（64位）实现固定大小的位集
- 支持按索引设置/取消设置/查询单个位
- 提供查找第一个设置位的功能

**关键方法：**
- `set(size_t index)` - 设置指定索引的位
- `unset(size_t index)` - 取消设置指定索引的位
- `get(size_t index)` - 查询指定索引的位状态
- `is_entirely_unset()` - 检查是否所有位都未设置
- `for_each_set_bit(Func&& func)` - 遍历所有已设置的位并执行回调函数

**实现细节：**
- `find_first_set()` 私有方法用于查找第一个设置位的索引（1-索引），返回0表示无位被设置
- 平台特化实现：MSVC 使用 `_BitScanForward64`/`_BitScanForward`，其他平台使用 GCC 内置函数 `__builtin_ffsll`
- 支持相等性比较操作符

**设计理由：**
- 存在的唯一原因是 `std::bitset` 缺少 `find_first_set()` 方法
- 若无特殊需求，推荐优先使用 `std::bitset`

**主要特性总结：**
- 64位固定大小
- 跨平台支持（MSVC/x64/ARM64/x86、POSIX）
- constexpr 支持（除 `for_each_set_bit` 外）
- 轻量级位操作工具类
