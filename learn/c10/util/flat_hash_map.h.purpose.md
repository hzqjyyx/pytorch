`c10/util/flat_hash_map.h` 是一个第三方开源哈希表实现，来自 [skarupke/flat_hash_map](https://github.com/skarupke/flat_hash_map)，PyTorch 做了一些修复和适配。这个文件实现了高性能的 `flat_hash_map` 和 `flat_hash_set` 容器。

## 核心设计

**Robin Hood Hashing**

使用 Robin Hood hashing 策略处理冲突。每个槽位记录 `distance_from_desired`（距离理想位置的偏移），插入时通过"劫富济贫"策略：如果新元素的偏移距离大于当前槽位元素的偏移，就交换两者并继续插入被置换的元素。这种方式减少了探测链长度的方差，提升查找性能。

**数据结构**

`sherwood_v3_entry<T>` 是基本存储单元：
- `distance_from_desired`: int8_t，负数表示空槽，非负数表示已占用且记录偏移距离
- `value`: union 中的实际值，避免默认构造
- `special_end_value`: 哨兵标记数组末尾

`sherwood_v3_table` 是核心哈希表模板，维护：
- `entries`: 槽位数组
- `num_slots_minus_one`: 槽位数减一（用于快速取模）
- `hash_policy`: 哈希策略（prime/power_of_two/fibonacci）
- `max_lookups`: 最大探测长度（基于 log2(bucket_count)）
- `_max_load_factor`: 默认 0.5

## 主要操作

**查找 (find)**

1. 计算 hash 得到起始索引
2. 从起始位置线性探测，比较 `distance_from_desired`
3. 如果当前槽位的距离小于探测距离，说明元素不存在（Robin Hood 特性保证）
4. 否则继续探测直到找到或确认不存在

**插入 (emplace)**

1. 先尝试查找，存在则返回
2. 否则调用 `emplace_new_key`：
   - 检查是否需要扩容（负载因子超限或探测距离达到上限）
   - 找到空槽直接插入
   - 遇到已占用槽且新元素偏移更大，交换并继续插入被置换元素
   - 形成插入链，直到找到空槽

**删除 (erase)**

1. 销毁目标元素
2. 向后查找非理想位置的元素，依次前移填补空洞
3. 维护 `distance_from_desired` 正确性

**扩容 (rehash)**

1. 分配新的更大数组（默认 2x 扩容）
2. 重新插入所有元素
3. 更新哈希策略参数

## 哈希策略

**prime_number_hash_policy**
- 使用质数作为桶数量
- 通过预定义质数表选择合适大小
- 索引计算：`hash % prime`

**power_of_two_hash_policy**
- 桶数量为 2 的幂
- 索引计算：`hash & (size - 1)`（位运算更快）

**fibonacci_hash_policy** (默认)
- 桶数量为 2 的幂
- 索引计算：`(11400714819323198485ull * hash) >> shift`
- 利用黄金比例的乘法哈希，分布更均匀

## 容器接口

**flat_hash_map<K, V>**
- 标准 map 接口：`operator[]`, `at()`, `insert()`, `emplace()`
- `insert_or_assign()`: 插入或更新值
- 比较操作符 `==`, `!=`

**flat_hash_set<T>**
- 标准 set 接口
- 特化的 `emplace()` 处理各种参数类型

## 性能优化

- 使用 union 延迟构造，避免默认初始化开销
- 紧凑内存布局（flat，数组连续存储）
- 缓存友好的线性探测
- 低负载因子（0.5）减少冲突
- `SKA_NOINLINE` 宏标记大函数防止过度内联

## PyTorch 特定修改

- 用 `uint64_t` 替换 `size_t` 确保 32 位兼容
- 添加 GCC/Clang 诊断 pragma 抑制警告
- 修复模板兼容性问题（GCC5）
- 使用 `if constexpr` 替代模板元编程
- 集成 `c10/macros/Macros.h` 的编译器诊断工具

---

**忽略的内容：**
- 无 ROCm 相关内容
- 无 Backward 相关内容
