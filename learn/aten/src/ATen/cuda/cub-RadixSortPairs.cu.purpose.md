**文件功能分析**

这是 PyTorch ATen 库中的 CUDA 基数排序实现文件。核心功能：

- **基数排序（Radix Sort）实现**：提供 `radix_sort_pairs_impl` 模板函数，用于对键值对进行排序
- **键值对同步排序**：能同时对 keys 和 values 进行排序，保持它们的对应关系
- **双向排序**：支持升序（SortPairs）和降序（SortPairsDescending）两种模式
- **位范围控制**：允许通过 `begin_bit` 和 `end_bit` 参数控制排序的比特范围
- **CUB 库封装**：将 NVIDIA CUB 库的 `DeviceRadixSort` 进行包装，便于 PyTorch 集成
- **内存管理**：如果输出缓冲为空，自动分配 CUDA 内存
- **类型支持**：提供多种实例化版本，支持 int32_t、int64_t、uint16_t、uint32_t、uint64_t、BFloat16、Bool、Half 等数据类型
- **值大小灵活性**：通过 `value_size` 模板参数支持不同大小的 value 数据（1、2、4、8 字节）
- **流异步执行**：使用当前 CUDA 流进行排序操作，支持异步执行
