`c10/util/strong_type.h` 是一个第三方强类型封装库（来自 rollbear/strong_type 项目），用于在 C++ 中创建类型安全的包装类型。

## 核心设计

**基础模板类 `type`** (74-137行)：
```cpp
template <typename T, typename Tag, typename ... M>
class type : public modifier<M, type<T, Tag, M...>>...
```
- `T`: 底层值类型
- `Tag`: 用于区分不同强类型的标签（即使底层类型相同）
- `M...`: 修饰符参数包，通过 CRTP 继承添加功能

**值访问接口**：
- `value_of()` 成员函数和友元函数：提取包装的底层值
- 支持左值、const 左值、右值引用的重载

## 修饰符系统

通过可组合的修饰符为强类型添加特定能力：

**基础特性**：
- `default_constructible`: 支持默认构造
- `equality`: 相等性比较 (`==`, `!=`)
- `ordered`: 全序比较 (`<`, `<=`, `>`, `>=`)
- `semiregular`: 可拷贝、可移动、可默认构造
- `regular`: semiregular + equality
- `unique`: 仅可移动，禁止拷贝

**跨类型操作**：
- `equality_with<Ts...>`: 与其他类型 Ts 进行相等性比较
- `ordered_with<Ts...>`: 与其他类型进行顺序比较

**算术操作**：
- `arithmetic`: 完整算术运算 (`+`, `-`, `*`, `/`, `%`, 一元 `-`)
- `difference`: 差值类型语义（可加减差值、相互相减得差值、可与标量相乘除）
- `affine_point<D>`: 仿射点语义（点-点=差值，点±差值=点）
- `bitarithmetic`: 位运算 (`&`, `|`, `^`, `~`, `<<`, `>>`)
- `incrementable`/`decrementable`/`bicrementable`: 自增/自减

**容器和迭代**：
- `indexed<I>`: 下标访问 (`operator[]`, `at()`)
- `iterator`: 迭代器特性（根据底层迭代器类别自动适配）
- `range`: 范围语义（提供 begin/end/cbegin/cend）

**其他功能**：
- `pointer`: 指针语义（与 nullptr 比较、解引用、箭头访问）
- `boolean`: 显式 bool 转换
- `ostreamable`/`istreamable`/`iostreamable`: 流操作
- `convertible_to<Ts...>`/`implicitly_convertible_to<Ts...>`: 类型转换
- `hashable`: 支持 std::hash
- `formattable`: 支持 std::format/fmt::format

## 类型特征和工具

**类型检测**：
- `is_strong_type<T>`: 判断是否为强类型
- `underlying_type<T>`: 提取底层类型
- `impl::access()`: 统一访问接口（强类型提取值，普通类型直接返回）

**标准库集成**：
- `std::hash` 特化（需 hashable 修饰符）
- `std::is_arithmetic` 特化（检测 arithmetic 修饰符）
- `std::formatter`/`fmt::formatter` 特化（需 formattable 修饰符）

## 使用示例语义

```cpp
// 创建类型安全的 ID
using UserId = strong::type<int, struct user_tag, strong::regular, strong::ordered>;
using ProductId = strong::type<int, struct product_tag, strong::regular, strong::ordered>;

// UserId 和 ProductId 不能混用，即使底层都是 int
UserId uid{42};
ProductId pid{42};
// uid == pid;  // 编译错误
```

---

**ROCm 相关**: 无

**Backward 兼容性**: 无
