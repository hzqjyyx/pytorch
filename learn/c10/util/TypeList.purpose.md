## c10/util/TypeList.h

这是一个编译期类型列表操作库，提供类似函数式编程的类型元编程工具。核心功能：

### 1. 基础类型 `typelist<T...>`
- 编译期类型容器，不可实例化（`delete` 构造函数）
- 纯粹用于模板元编程的类型计算

### 2. 基本操作
- **`size`**: 获取类型列表长度
- **`concat`**: 连接多个类型列表，递归实现
- **`reverse`**: 反转类型列表顺序

### 3. 与 std::tuple 互转
- **`to_tuple_t`**: 将 `typelist<int, string>` 转为 `std::tuple<int, string>`
- **`from_tuple_t`**: 反向转换

### 4. 访问元素
- **`head`**: 取第一个类型
- **`head_with_default`**: 取第一个类型，空列表返回默认类型
- **`last`**: 取最后一个类型
- **`element<Index>`**: 按索引访问，递归实现（每次 `Index-1` 并移除 `Head`）
- **`take<N>`/`drop<N>`**: 取前/后 N 个类型，通过 `std::index_sequence` 批量索引

### 5. 过滤与查询
- **`filter`**: 按类型特征过滤（如 `std::is_reference`）
- **`count_if`**: 统计满足条件的类型数量
- **`contains`**: 检查是否包含某类型，用 SFINAE 递归实现
- **`all`**: 所有类型满足条件（用 `std::conjunction`）
- **`true_for_any_type`**: 任一类型满足条件（用 `std::disjunction`）
- **`find_if`**: 返回第一个满足条件类型的索引

### 6. 映射
- **`map`**: 对每个类型应用类型变换（如 `std::add_lvalue_reference_t`）
- **`map_types_to_values`**: 将类型列表映射为运行时值的 tuple
  ```cpp
  // 示例：将类型映射为 sizeof 值
  auto sizes = map_types_to_values<typelist<int, bool>>(
    [](auto t) { return sizeof(typename decltype(t)::type); }
  ); // -> std::tuple<size_t, size_t>{4, 1}
  ```

### 7. 实现技巧
- **SFINAE**: `contains` 用 `std::enable_if_t` 区分匹配/不匹配分支
- **递归模板**: 多数操作通过递归特化实现（如 `concat`、`reverse`）
- **辅助类型**: `false_t`/`false_higher_t` 用于延迟 `static_assert`（避免主模板立即失败）
- **`type_` wrapper**: `map_types_to_values` 中用 `type_<T>` 将类型包装为值传递给 lambda

### 8. 典型应用场景
- 泛型库中的参数包处理
- 编译期类型约束检查
- 根据类型列表生成代码或数据结构

## c10/util/TypeList.cpp

仅包含头文件 `#include <c10/util/TypeList.h>`，无实际实现代码。所有功能都在头文件中通过模板实现。

---

**其他提及内容**:
- ROCm/Backward: 无
- 依赖: `c10/util/TypeTraits.h`（提供 `is_type_condition`、`is_instantiation_of` 等辅助工具）
