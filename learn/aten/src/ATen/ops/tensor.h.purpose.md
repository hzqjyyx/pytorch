这个文件定义了 PyTorch 中创建张量的便捷函数接口。

**主要功能：**

- **宏定义 `TENSOR`**：为不同的标量类型生成多个 `tensor()` 函数重载
  - 接受 `ArrayRef<T>` 数组引用
  - 接受 `std::initializer_list<T>` 初始化列表
  - 接受单个标量值 `T`
  - 支持带 `TensorOptions` 参数和不带参数两种形式

- **类型覆盖**：
  - `AT_FORALL_SCALAR_TYPES_AND3(Bool, Half, BFloat16, TENSOR)` - 所有标量类型 + Bool、Half、BFloat16
  - `AT_FORALL_COMPLEX_TYPES(TENSOR)` - 所有复数类型

- **便利性**：允许用户用多种方式创建张量，如：
  ```cpp
  at::tensor({1, 2, 3})
  at::tensor(5.0)
  at::tensor({1.0, 2.0}, at::dtype(torch::kFloat32))
  ```

- **实现位置**：实际函数体在 `ATen/Utils.cpp` 中定义
