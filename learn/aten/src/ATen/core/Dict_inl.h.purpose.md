这个文件是 PyTorch 的字典容器实现的内联定义文件。主要内容：

**核心结构：**
- `DictKeyHash`: 为不同类型的 IValue 提供哈希函数
  - 支持的类型：int、string、double、complex double、bool、tensor、device
  - Tensor 类型使用 TensorImpl 指针的哈希值
  - 不支持的类型抛出运行时错误

- `DictKeyEqualTo`: 定义字典键的相等比较逻辑
  - Tensor 使用恒等比较（`is`），遵循 Python 行为
  - 其他类型使用 `_fastEqualsForContainer` 进行快速比较

**模板类 Dict<Key, Value> 的主要方法：**
- 构造函数：默认构造、指定键值类型构造、移动语义构造
- 容器操作：`begin()`、`end()`、`empty()`、`size()`、`clear()`
- 插入/删除：`insert()`、`insert_or_assign()`、`erase()`
- 查询：`find()`、`contains()`、`at()`
- 其他：`copy()`、`reserve()`、`keyType()`、`valueType()`
- 比较：`operator==`、`operator!=`、`is()`（恒等比较）

**转换工具（impl 命名空间）：**
- `toTypedDict()`: GenericDict → Dict<Key, Value>（带类型检查）
- `toGenericDict()`: Dict<Key, Value> → GenericDict

**关键特点：**

- 所有修改操作（insert、erase、clear）声明为 `const`，实现通过 impl_ 指针修改（逻辑常量性）
- 内部使用 `intrusive_ptr<DictImpl>` 管理内存（引用计数）
- 支持类型安全的泛型字典和无类型的通用字典
- 比较时先比较恒等性再比较值（性能优化）
