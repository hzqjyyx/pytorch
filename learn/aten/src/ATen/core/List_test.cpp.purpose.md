这个文件是 PyTorch 中 `c10::List` 类的单元测试文件，测试了类似 `std::vector` 的动态列表容器的各种功能。

## 主要测试内容

### 1. 两种 List 类型的测试
- **IValueBasedList**: 基于 `List<string>` 的测试，用于测试可以存储在 IValue 中的类型
- **NonIValueBasedList**: 基于 `List<int64_t>` 的测试，用于测试原始类型

### 2. 基本容器操作
- **容量查询**: `empty()`, `size()`, `clear()`
- **元素访问**: `get()`, `extract()`, `operator[]`
- **元素修改**: `set()` (拷贝/移动版本)
- **删除操作**: `erase()`, `pop_back()`
- **调整大小**: `resize()`, `reserve()`

### 3. 插入操作
- `insert()`: 在指定位置插入元素 (lvalue/rvalue)
- `emplace()`: 在指定位置原位构造元素
- `push_back()`: 尾部添加元素
- `emplace_back()`: 尾部原位构造元素

### 4. 迭代器功能
- **基础迭代**: `begin()`, `end()`, 范围 for 循环
- **递增/递减**: 前缀/后缀 `++`, `--`
- **算术运算**: `+=`, `-=`, `+`, `-`
- **解引用和赋值**: `*iter = value`
- **比较操作**: `==`, `!=`
- **迭代器距离**: `end() - begin()`

### 5. 拷贝与移动语义
- **拷贝构造/赋值**: 创建共享存储的引用（引用类型行为）
- **移动构造/赋值**: 原对象保持不变（浅拷贝）
- **显式拷贝**: `copy()` 方法创建独立存储

### 6. 引用类型行为测试
```cpp
// List 是引用类型：拷贝后共享底层存储
List<int64_t> list1;
List<int64_t> list2(list1);  // list2 和 list1 共享存储
list1.push_back(3);
// list2.size() 也变成 1
```

### 7. 类型安全和特殊访问模式
- **字符串访问**: `List<std::string>` 通过常量引用访问
- **可选类型访问**: `List<std::optional<std::string>>` 返回 `std::optional<std::reference_wrapper<const std::string>>`
- **Tensor 访问**: `List<at::Tensor>` 通过常量引用访问
- **类型转换**: `toList()` 和 `toTypedList<T>()` 的往返转换，包括错误类型转换的异常测试

### 8. 边界条件和异常处理
- 访问不存在的位置抛出 `std::out_of_range`
- 空列表的迭代器行为
- 单元素列表的删除操作

### 关键设计特点
- **引用语义**: 默认拷贝共享存储，需要 `copy()` 获得独立副本
- **双重实现**: 同时支持 IValue 类型和非 IValue 类型
- **STL 兼容**: 提供类似 `std::vector` 的接口
- **类型转换**: 支持泛型 List 和类型化 List 之间的转换

ROCm 和 Backward 相关：无
