这个文件定义了两个内联工具函数，用于在排序相关操作中管理输出张量的内存分配和调整：

**`_reduction_with_indices_allocate_or_resize_output`**
- 为 `kthvalue` 和 `mode` 操作分配或调整输出张量
- 接收 `values` 和 `indices` 两个输出张量引用
- 将指定维度缩减为大小 1
- 支持 `keepdim` 参数控制是否保持维度
- 如果张量已定义则调整大小，否则创建新张量
- 验证 `values` 类型与输入一致，`indices` 为 Long 类型

**`_allocate_or_resize_output_with_indices`**
- 为 `topk` 操作分配或调整输出张量
- 接收 `values` 和 `indices` 两个输出张量引用
- 将指定维度调整为 `k` (topk 的 k 值)
- 如果张量已定义则调整大小，否则创建新张量
- 同样验证 `values` 类型与输入一致，`indices` 为 Long 类型

**关键功能点**：
- 避免重复分配：如果输出张量已定义则复用，否则创建
- 维度处理：支持负索引的维度包装
- 类型安全：检查输出张量类型和设备
- 内存效率：保留传入的非连续张量的原始布局
