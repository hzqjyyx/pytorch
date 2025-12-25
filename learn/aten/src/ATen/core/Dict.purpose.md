## ATen/core/Dict 核心功能

这是 PyTorch 的类型安全字典容器实现，用于在 C++ 层提供类似 Python dict 的功能。

### 主要组件

**DictImpl (detail namespace)**
- 底层实现类，使用 `ska_ordered::order_preserving_flat_hash_map` 保持插入顺序
- 存储 `IValue` 到 `IValue` 的映射，支持运行时类型信息
- 实现了引用计数（继承自 `intrusive_ptr_target`）
- 提供相等性比较：检查键值类型匹配、大小相等、所有键值对内容相等（不关心顺序）

**Dict<Key, Value> 模板类**
- 指针语义：拷贝后共享底层存储
- 支持的键类型受限（`int64_t`, `string`, `double`, `complex<double>`, `bool`, `Tensor`）
- 提供标准容器接口：`begin()`, `end()`, `size()`, `empty()`, `clear()`
- 插入操作：`insert()` 不覆盖已存在键，`insert_or_assign()` 会覆盖
- 查询操作：`at()`, `find()`, `contains()`
- 类型安全：通过模板在编译期检查键值类型

**迭代器系统**
- `DictIterator<Key, Value, Iterator>`: 封装底层迭代器，隐藏实现细节
- `DictEntryRef<Key, Value, Iterator>`: 字典条目的引用，提供 `key()`, `value()`, `setValue()` 方法
- 支持 forward iterator 语义
- 通过 `to<T>()` 将 `IValue` 转换为具体类型

**哈希与相等性**
- `DictKeyHash`: 为 `IValue` 提供哈希函数
- `DictKeyEqualTo`: 为 `IValue` 提供相等性比较

### 设计特点

1. **类型系统双重性**: 同时支持泛型 `Dict<IValue, IValue>` (GenericDict) 和类型化版本（如 `Dict<int64_t, string>`）
2. **Copy-on-Write 语义**: 提供 `copy()` 深拷贝，但默认是浅拷贝（共享存储）
3. **Python 语义**: `operator==` 实现值相等比较，`is()` 方法实现身份比较
4. **顺序保持**: 使用 order-preserving hash map 保持插入顺序
5. **类型标签可变**: `unsafeSetKeyType/ValueType` 用于 unpickler 场景，运行时修改类型标签

### Backward/ROCm 相关
- 无 backward 或 ROCm 特定代码
- 纯容器实现，不涉及梯度计算或设备特定逻辑
