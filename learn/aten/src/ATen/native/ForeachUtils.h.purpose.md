这个文件提供了一套完整的工具函数，用于支持 PyTorch 的 `foreach` 系列批量操作 API。主要功能包括：

## 核心验证功能

**1. API 限制检查 (`check_foreach_api_restrictions`)**
- 确保张量列表非空
- 验证多个张量列表之间元素数量一致
- 验证张量列表与标量列表长度匹配
- 提供多个重载版本，支持 1-3 个张量列表 + 可选标量列表的组合

**2. Fast Path 路径检查**

用于判断是否可以走优化的快速路径，通过三个辅助函数实现：

- `_check_tensors_share_device_and_dtype`: 检查所有张量是否在同一设备、同一数据类型、strided layout、非重叠且稠密
- `_check_tensors_share_sizes_and_strides`: 检查对应位置的张量是否有相同的尺寸和步幅（忽略大小为 1 的维度）
- `_check_tensors_do_type_promotion_with_scalars`: 检查张量与标量的类型提升是否符合预期

**3. 类型判断辅助函数**
- `has_integral_tensor`: 检查是否有整数类型张量（可选包含布尔）
- `has_bool_tensor`: 检查是否有布尔类型张量

## 高级功能

**4. 张量分组机制 (`_group_tensors_by_first_tensors_device_and_dtype`)**

这是为优化器场景设计的复杂分组函数：
- 接收嵌套的可选张量列表 `nested_optional_tensorvec_t`
- 根据第一个张量列表中每个张量的设备和数据类型进行分组
- 特殊处理优化器中的 `state_step` 张量（允许在 CPU 上且为 float32/64）
- 返回 `FlatMap` 结构，将相同设备和数据类型的张量分组到一起
- 可选记录原始索引信息

**5. 标量转换**
- `convert_tensor_to_scalar_list`: 将 CPU 上的一维连续张量转换为标量列表，支持所有数据类型

## 设计要点

**Fast Path 的条件**：
- 所有张量必须同设备、同数据类型
- 必须是 strided layout
- 必须非重叠且稠密
- 结果张量与输入数据类型一致
- 对于会将整数提升为浮点的操作（如除法），整数输入会被拒绝走 fast path

**类型别名定义**：
- `DeviceDtypeKey`: 设备和数据类型的配对
- `nested_optional_tensorvec_t`: 嵌套的可选张量向量
- `FlatMap`: 用于分组的哈希映射，使用自定义 `ParamsHash`

---

**ROCm/Backward 相关**：无（此文件仅包含通用工具函数）
