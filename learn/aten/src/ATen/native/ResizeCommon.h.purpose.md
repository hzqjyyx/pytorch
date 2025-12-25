- **storage_size_for(size, stride)**: 模板函数，计算给定大小和步长对应的存储大小。如果任何维度大小为0，则存储大小为0；否则累加计算所需的总元素数。

- **resize_named_tensor_(self, size, optional_memory_format)**: 处理命名张量的resize操作。验证新大小与现有大小相同（命名张量不支持实际resize），并确保不指定内存格式。

- **fill_resize_deterministic_(tensor, old_storage_nbytes)**: 在张量存储resize后，用确定性值（NaN或MAX_INT）填充新增的存储区域，用于保证输出的确定性。比较旧存储字节数与新存储字节数，如果有新增空间则创建视图并填充。
