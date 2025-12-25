- **文件目的**：为 PyTorch 的 CUDA 操作提供基数排序（Radix Sort）的包装实现

- **核心函数** `radix_sort_keys`（8-42行）：
  - 接收输入/输出键数组、数组大小、排序方向（升序/降序）、位范围参数
  - 验证排序数量不超过 `INT_MAX`
  - 通过类型转换适配 CUDA 类型系统
  - 调用 CUB 库的 `DeviceRadixSort::SortKeys` 或 `DeviceRadixSort::SortKeysDescending`
  - 使用当前 CUDA 流执行排序操作

- **模板实例化**（44-56行）：
  - 为多种标量类型生成模板实例：Bool、BFloat16、Half、UInt16、UInt32、UInt64 及其他标准类型
  - 确保排序函数支持 PyTorch 的所有数值类型

- **依赖关系**：
  - 基于 CUB（CUDA Unbound）库的高性能基数排序实现
  - 与 CUDA 内存和流管理集成
