这个文件是 PyTorch 中 `List<T>` 模板类的内联实现文件。

## 主要功能

该文件定义了一个通用的类型化列表容器 `List<T>`，用于在 PyTorch 的 JIT 编译器中存储和管理同一类型的元素集合。核心包括：

**构造函数：**
- 支持从 `intrusive_ptr` 创建（移动和拷贝构造）
- 支持从 `ArrayRef<T>` 和初始化列表创建
- 特殊的 `TypePtr` 构造用于 `List<IValue>` 和 `List<Future>`

**类型转换：**
- `toTypedList()` 将通用列表转换为类型化列表，支持协变性检查
- `toList()` 反向转换为通用列表

**容器操作：**
- 基础操作：`size()`, `empty()`, `clear()`, `reserve()`
- 访问：`operator[]`, `get()`，支持常量和非常量引用
- 修改：`set()`, `push_back()`, `pop_back()`, `insert()`, `erase()`
- 迭代：`begin()`, `end()`
- 高级：`append()`, `emplace()`, `emplace_back()`, `extract()`, `resize()`

**引用语义：**
- `ListElementReference<T, Iterator>` 提供列表元素的代理引用
- 支持引用赋值、相等比较和交换操作

**工具方法：**
- `copy()` 深拷贝列表
- `vec()` 转换为 `std::vector<T>`
- `use_count()` 获取引用计数
- `elementType()` 获取元素类型指针
- `operator==`, `operator!=`, `is()` 比较和身份检查

---

**关键特性总结：**

- 模板化的类型安全列表容器
- 基于 `intrusive_ptr` 的引用计数管理
- 支持 `IValue`（动态类型值）和具体类型混合使用
- 提供 STL 兼容的容器接口
- 支持常量和非常量操作的细致控制
