这个文件定义了两个随机访问迭代器类，用于遍历步长数组：

**ConstStridedRandomAccessor**
- 常量随机访问迭代器，提供对步长数组的只读访问
- 支持两个模板参数：元素类型 `T` 和索引类型 `index_t`
- 可选的 `PtrTraits` 模板参数用于控制指针的 `__restrict__` 修饰符（平台相关）

**StridedRandomAccessor**
- 非常量版本，继承自 `ConstStridedRandomAccessor`
- 提供可写访问能力

**主要功能列表：**

- 定义了标准随机访问迭代器所有操作符（`*`, `->`, `[]`, `++`, `--`, `+=`, `-=`, `+`, `-`, `==`, `!=`, `<`, `<=`, `>`, `>=`）
- 支持 C10_HOST_DEVICE 宏，可在 CPU 和 GPU 上运行
- 通过 `stride` 参数实现非连续内存访问（如多维数组行/列访问）
- 索引访问 `operator[]` 计算为 `ptr[idx * stride]`
- 指针算术操作也考虑步长因子
- 两个指针的差值除以步长得到元素距离
- 使用指针特性模板区分平台间的 `__restrict__` 语法差异（Windows vs 其他平台）
