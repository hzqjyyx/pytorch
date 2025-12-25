- **Include Guard Check**: 文件开始检查 `C10_INTERNAL_INCLUDE_COMPLEX_REMAINING_H` 宏，确保只通过 `c10/util/complex.h` 间接包含，不允许直接包含

- **is_complex 模板**: 提供类型检查，判断一个类型是否为复数类型
  - 默认为 `false_type`
  - 对 `std::complex<T>` 特化为 `true_type`
  - 对 `c10::complex<T>` 特化为 `true_type`

- **scalar_value_type 模板**: 提取复数的标量类型
  - 对普通类型 `T` 返回 `T` 本身
  - 对 `std::complex<T>` 返回 `T`
  - 对 `c10::complex<T>` 返回 `T`

- **std::numeric_limits 特化**: 为 `c10::complex<T>` 提供数值限制，继承自其标量类型的限制

- **std::isnan 重载**: 为 `c10::complex<T>` 实现 NaN 检查，判断实部或虚部是否为 NaN
