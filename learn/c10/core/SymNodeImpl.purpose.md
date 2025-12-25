这两个文件定义了PyTorch符号计算系统中的基础抽象类 `SymNodeImpl`，用于表示动态形状和值的符号表示。

## 核心功能

**SymNodeImpl.h** 定义了抽象基类，包含以下主要方法族：

1. **类型检查** - `is_int()`, `is_bool()`, `is_float()` 等，判断符号节点代表的值类型
2. **算术运算** - `add()`, `sub()`, `mul()`, `truediv()`, `pow()`, `floordiv()`, `mod()` 等
3. **比较运算** - `eq()`, `ne()`, `gt()`, `lt()`, `le()`, `ge()`
4. **逻辑运算** - `sym_or()`, `sym_and()`, `sym_not()`, `sym_ite()`
5. **约束检查** - `is_contiguous()`, `is_channels_last_contiguous_2d/3d()`, `is_non_overlapping_and_dense()` 等，用于验证张量内存布局
6. **Guard操作** - `guard_int()`, `guard_bool()`, `guard_float()` 等，在编译时对符号值进行具体化/断言
7. **类型转换** - `sym_float()`, `wrap_int()`, `wrap_float()`, `wrap_bool()`
8. **值提取** - `int_()`, `bool_()`, `constant_int()`, `constant_bool()`, `maybe_as_int()`, `nested_int()`
9. **元数据** - `has_hint()`, `str()`, `_graph_repr()`, `is_constant()`, `is_symbolic()`

**SymNodeImpl.cpp** 仅包含空的命名空间声明，实际实现由子类完成。

---

- **抽象基类**：定义符号值/形状的统一接口
- **虚函数设计**：大部分方法为虚函数，默认抛出"NYI"（Not Yet Implemented）异常
- **动态形状支持**：支持在编译时处理动态的张量维度
- **多态性**：使用 `dyn_cast<T>()` 进行安全的类型转换，支持 `intrusive_ptr` 智能指针管理
- **内存布局验证**：提供张量连续性和通道排列的符号检查
- **Guard机制**：在编译时对符号值进行约束和具体化
- **评论指引**：头文件注释指出修改此接口需同步更新相关文件（jit/python/init.cpp、python_symnode.h、ConstantSymNodeImpl.h）
