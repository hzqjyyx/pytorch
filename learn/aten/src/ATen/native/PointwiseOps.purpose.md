**PointwiseOps.h** 定义了三个函数指针类型和五个分发声明：

- `pointwise_fn`: 接收 TensorIterator 和 Scalar，用于基础逐元素操作
- `structured_pointwise_fn`: 接收 TensorIteratorBase 和 Scalar，用于结构化逐元素操作
- `pointwise_fn_double`: 接收 TensorIterator、Scalar 和 double，用于需要额外浮点参数的操作

分发声明包括 `addcmul_stub` 和 `addcdiv_stub`（对应结构化版本）以及 `smooth_l1_backward_stub` 和 `huber_backward_stub`（对应双参数版本）。

**PointwiseOps.cpp** 实现了两个核心操作：

**addcmul 操作**:
- 元数据函数配置 TensorIterator，启用 CPU 标量、输入提升到公共类型、输出类型转换和安全转换检查
- 实现函数调用 `addcmul_stub` 执行实际的逐元素乘加操作

**addcdiv 操作**:
- 元数据函数在整数类型输入时抛出错误（已弃用整数除法）
- 调用 `build_ternary_op` 配置张量迭代器
- 实现函数调用 `addcdiv_stub` 执行实际的逐元素加除操作

**功能总结**:

- 提供 `addcmul` 和 `addcdiv` 两个三元逐元素操作的接口
- 分离元数据处理（形状、数据类型推理）和实现（具体计算）
- 使用分发系统支持不同设备后端（CPU/CUDA）
- 通过 TensorIterator 统一处理广播和数据类型转换
