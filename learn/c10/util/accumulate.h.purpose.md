这个文件提供了一组模板函数，用于对容器或迭代器范围内的整数进行累积操作：

- **sum_integers**: 两个重载版本，分别接收容器或迭代器范围，返回所有元素之和（int64_t）
- **multiply_integers**: 两个重载版本，分别接收容器或迭代器范围，返回所有元素之积（int64_t）
- **numelements_from_dim**: 计算从指定维度k开始到末尾的所有维度的乘积，k超出范围返回1
- **numelements_to_dim**: 计算从起始到指定维度k（不含）的所有维度的乘积
- **numelements_between_dim**: 计算k到l之间（含k，不含l）所有维度的乘积，k和l顺序可任意

所有函数都使用 `int64_t` 作为累积类型以避免溢出，通过 SFINAE（`std::enable_if_t`）确保只对整数类型容器生效。
