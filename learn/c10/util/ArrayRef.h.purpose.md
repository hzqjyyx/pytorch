## c10/util/ArrayRef.h 功能分析

**ArrayRef** 是一个常量数组引用包装器，提供对连续内存中数组的非所有权视图。

### 核心设计
- 仅存储两个成员：指向数据的指针 (`Data`) 和元素个数 (`Length`)
- 不拥有底层数据，仅引用外部缓冲区中的数据
- 设计为平凡可复制类型，按值传递
- 始终提供 const 迭代器（只读访问）

### 主要功能特性

- **构造方式多样**
  - 单个元素、指针+长度、指针范围
  - SmallVector、std::vector、std::array、C 数组、initializer_list
  - 通过 SFINAE 支持任何提供 `.data()` 和 `.size()` 的容器

- **迭代和访问**
  - `begin()/end()` 和反向迭代器
  - `operator[]` 和 `at()` 方法
  - `front()` 和 `back()` 获取端点元素
  - `empty()` 检查、`size()` 获取大小、`data()` 获取指针

- **切片操作**
  - `slice(N, M)` 取 M 个元素从位置 N 开始
  - `slice(N)` 舍弃前 N 个元素

- **比较和转换**
  - `equals()` 元素级相等性检查
  - `operator==` 和 `operator!=` 重载（支持与 ArrayRef、vector 的比较）
  - `vec()` 转换为 std::vector
  - `allMatch()` 检查是否所有元素满足谓词

- **便利函数**
  - `makeArrayRef()` 模板函数族用于隐式构造 ArrayRef

- **安全防护**
  - 调试断言检查 nullptr 和非零长度的非法组合
  - 禁止从临时值赋值（删除赋值运算符）
  - 运行时检查边界访问 (`at()`)、切片边界 (`slice()`)、首尾访问 (`front()`/`back()`)

### 别名与相关定义

- `IntArrayRef` ← `ArrayRef<int64_t>` （用于整数数组）
- `IntList` ← 已弃用别名
