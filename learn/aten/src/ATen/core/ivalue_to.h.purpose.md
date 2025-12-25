这个文件定义了一个模板元编程工具，用于确定 `IValue::to()` 方法的返回类型。

**核心机制：**

`ivalue_to_const_ref_overload_return` 是一个模板结构体，它根据模板参数类型来决定返回值应该是引用还是拷贝：

- **默认情况**：返回类型为 `T`（值类型，即拷贝）
- **特化情况**：对于特定类型返回 `const` 引用以避免不必要的拷贝

**功能列表：**

- 为 `at::Tensor` 类型提供 `const at::Tensor&` 返回类型
- 为 `std::string` 类型提供 `const std::string&` 返回类型  
- 为 `IValue` 类型提供 `const IValue&` 返回类型
- 避免复制大对象（张量、字符串）时的性能开销
- 为 List 等其他容器提供类型推导支持
