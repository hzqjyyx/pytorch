这个文件实现了一个通用的**函数重载组合器**（overload combinator），允许将多个可调用对象（如lambda函数）合并成一个单一的可调用对象。

## 核心机制

**模板递归继承结构**：
- `overloaded_t` 是一个可变参数模板，通过多重继承逐步继承每个传入的callable
- 特化版本 `overloaded_t<T0>` 作为递归基础，直接继承单个callable
- 递归版本 `overloaded_t<T0, Ts...>` 同时继承第一个callable和剩余callable的组合

**operator() 注入**：
- 使用 `using` 声明将所有父类的 `operator()` 导入到当前作用域
- 这样编译器在做重载解析时能看到所有被继承的 `operator()` 版本
- C++的重载解析会自动选择最匹配的那个

**转发函数**：
- `overloaded()` 函数是便利工厂函数，接收可变个callable并返回组合后的对象
- 使用完美转发 `std::move` 避免不必要的复制

## 使用场景

典型用途是在 `std::visit` 中处理 `std::variant`，通过多个lambda覆盖所有可能的类型分支。

---

- 通过多重继承聚合多个callable对象
- 递归模板结构逐层组织callable
- `using` 声明合并所有 `operator()`
- 支持编译期多态和重载解析
- 避免手写visitor类或大型switch语句
