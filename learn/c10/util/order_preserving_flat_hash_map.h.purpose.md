这是一个基于开源 flat_hash_map 实现的、**保持插入顺序的哈希表**数据结构。

## 核心设计

基于 Robin Hood 哈希算法的扁平哈希表，通过**双向链表**维护插入/删除顺序：

```cpp
template <typename T>
struct sherwood_v3_entry {
  sherwood_v3_entry<T>* prev = nullptr;  // 链表前驱
  sherwood_v3_entry<T>* next = nullptr;  // 链表后继
  int8_t distance_from_desired = -1;     // Robin Hood 距离
  union { T value; };                     // 实际存储的值
};
```

- **哈希表部分**：用 `entries` 数组存储，通过 `distance_from_desired` 实现 Robin Hood probing
- **顺序维护**：`sentinel` 作为链表哨兵，所有元素按插入顺序串联

## 关键机制

### 1. 插入保序 (c10/util/order_preserving_flat_hash_map.h:900-960)

```cpp
emplace_new_key() {
  // 标准 Robin Hood 插入逻辑
  if (current_entry->is_empty()) {
    current_entry->emplace(...);
    append_to_list(current_entry);  // 追加到链表尾部
  } else {
    // displacement 时维护不变式:
    // - result.current 指向新插入值的哈希槽位
    // - to_insert 保存被挤占的旧值
    swap(to_insert, current_entry->value);
    // ...继续探测...
    swap_positions(current_entry, result.current);  // 交换链表位置
  }
}
```

核心不变式：新元素总是插入到链表**末尾**，即使发生 Robin Hood displacement。

### 2. 删除保序 (c10/util/order_preserving_flat_hash_map.h:644-661)

```cpp
erase(const_iterator to_erase) {
  remove_from_list(current);         // 从链表摘除
  current->destroy_value();
  
  // 后移元素填补空洞
  for (EntryPointer next = current + 1; !next->is_at_desired_position(); ) {
    current->emplace(next->distance_from_desired - 1, std::move(next->value));
    replace_linked_list_position(next, current);  // 更新链表指针
    next->destroy_value();
  }
}
```

删除后需要处理：
1. 链表摘除节点
2. Robin Hood 回填（后续元素前移）
3. 同步更新被移动元素的链表位置

### 3. 哈希策略

提供三种策略 (c10/util/order_preserving_flat_hash_map.h:1015-2029)：

- **`prime_number_hash_policy`**: 取模质数（默认，200+ 个预定义质数）
- **`power_of_two_hash_policy`**: 按位与掩码（`hash & (size-1)`）
- **`fibonacci_hash_policy`**: 斐波那契乘法哈希

### 4. 迭代器

```cpp
iterator begin() { return sentinel->next; }  // 返回链表头
iterator end()   { return sentinel; }        // 返回哨兵

templated_iterator& operator++() {
  current = current->next;  // 沿链表前进，非哈希数组
  return *this;
}
```

迭代器遍历的是**链表**而非哈希数组，保证顺序性。

## 公开接口

### `order_preserving_flat_hash_map<K, V>` (c10/util/order_preserving_flat_hash_map.h:2037)

标准 map 接口：
- `operator[]`, `at()`, `find()`, `count()`
- `emplace()`, `insert()`, `insert_or_assign()`
- `erase()`, `clear()`, `rehash()`, `reserve()`

特性：
- 迭代顺序 = 插入顺序
- 相同 key 重复插入不改变顺序
- O(1) 平均查找/插入/删除

### `flat_hash_set<T>` (c10/util/order_preserving_flat_hash_map.h:2153)

set 容器，基于相同的底层表实现。

## 性能优化

- **扁平内存布局**：数组存储减少指针追踪
- **SKA_NOINLINE 宏**：防止 `emplace_new_key` 过度内联膨胀代码
- **Small load factor (0.5)**：减少冲突，提升 Robin Hood 效率
- **Max lookups**：探测距离上限 = log2(bucket_count)，避免病态链

---

**特殊点**:
- ROCm/HIP 相关：无（纯标准 C++ 实现）
- Backward pass：无关（数据结构层）
