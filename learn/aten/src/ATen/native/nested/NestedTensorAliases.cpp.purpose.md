这个文件是 PyTorch 中 ATen 库的一个简单别名/包装器模块。

**主要功能：**

- 为嵌套张量（Nested Tensor）的 `to_padded_tensor` 方法提供一个 C++ 原生函数包装
- `nested_to_padded_tensor` 函数接收：
  - 一个嵌套张量对象
  - 填充值（double 类型）
  - 可选的输出尺寸参数
- 直接委托给张量对象的 `to_padded_tensor` 成员方法执行实际转换逻辑
- 处于 `at::native` 命名空间，属于 ATen 的原生算子层

**简言之：** 这是一个薄包装层，用于暴露嵌套张量的填充转换功能给 C++ 层级调用。
