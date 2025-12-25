这个文件提供了三个检验函数，用于验证 scatter 和 gather 操作的输入参数：

**scatter_gather_dtype_check** (第14-34行)
- 检查 index 张量的数据类型是否为 int64
- 如果提供了 src 张量，验证 self 和 src 的数据类型是否相同

**gather_shape_check** (第41-60行)
- 验证 index 和 self 张量的维度数相同
- 检查除了指定维度 dim 外，index 在其他维度的大小不超过 self 对应维度的大小

**scatter_shape_check** (第67-124行)
- 验证 index、self 和 src（如果存在）的维度数相同
- 检查除了指定维度 dim 外，index 在其他维度的大小不超过 self 对应维度的大小
- 如果提供了 src，还要检查 index 在所有维度的大小都不超过 src 对应维度的大小

**总结：**
- 验证 scatter/gather 操作的索引张量类型和形状
- 确保数据类型兼容性
- 确保索引张量的大小与输入/输出张量匹配
