# OptionalArrayRef<T> 的主要功能

OptionalArrayRef<T> 是一个包装器类，提供了 std::optional<ArrayRef<T>> 的增强版本，解决了隐式转换构造函数可能导致的悬挂指针问题。

## 核心问题
std::optional<ArrayRef<T>> 的隐式转换构造函数在某些场景下会导致 ArrayRef<T> 存储指向临时对象的悬挂指针。OptionalArrayRef<T> 通过修复构造函数实现来避免这个问题。

## 主要接口

**构造函数:**
- 默认构造、空值构造
- 复制/移动构造
- 从 std::optional<ArrayRef<T>> 构造
- 从单个元素 `const T&` 构造
- 从可转换为 ArrayRef<T> 的类型构造（带 SFINAE 控制）
- 从初始化列表构造
- in_place 构造

**赋值操作:**
- 支持 nullopt、复制/移动赋值
- 支持从 std::optional<ArrayRef<T>> 赋值
- 支持从兼容类型赋值

**观察者方法:**
- operator->() / operator*() - 访问底层 ArrayRef
- operator bool() / has_value() - 检查是否有值
- value() - 获取值或抛异常
- value_or() - 获取值或默认值

**修改方法:**
- swap() - 交换
- reset() - 清空
- emplace() - 原地构造

**便利类型:**
- OptionalIntArrayRef = OptionalArrayRef<int64_t>

## 关键特性

- 完整的 ref 限定符支持（&、&&、const&、const&&）
- noexcept 规范
- constexpr 兼容
- SFINAE 控制防止隐式不安全转换
- 重载相等操作符用于 OptionalIntArrayRef 与 IntArrayRef 的比较
