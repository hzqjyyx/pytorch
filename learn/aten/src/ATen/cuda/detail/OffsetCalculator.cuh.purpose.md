**OffsetCalculator.cuh 文件功能分析**

这是一个 CUDA 优化工具，用于计算多张张量在内存中的偏移量。核心设计用于支持 elementwise 操作中的不同步幅 (strides) 张量。

**主要组件：**

- **OffsetCalculator 结构体** (第 21-73 行)
  - 将线性索引转换为多张张量的内存偏移量
  - 支持最多 25 维张量
  - 可选择字节偏移或元素偏移
  - 支持有符号/无符号步幅（用于 flip 等操作）
  - `get()` 方法：输入线性索引，返回每个张量对应的内存偏移

- **TrivialOffsetCalculator 结构体** (第 75-92 行)
  - 简化版本，仅用于所有张量步幅相同的情况
  - 直接返回线性索引作为所有张量的偏移

- **辅助工厂函数** (第 94-118 行)
  - `make_offset_calculator()` - 从 TensorIteratorBase 创建字节偏移计算器
  - `make_element_offset_calculator()` - 从 TensorIteratorBase 创建元素偏移计算器

**关键特性：**

- 使用 IntDivider 优化除法运算（加速线性索引到多维坐标的转换）
- 列主序 (column-major order) 遍历张量
- 编译时 `#pragma unroll` 展开循环以提高性能
- HOST_DEVICE 标记支持 CPU 和 GPU 执行
- 处理零参数张量的特殊情况（创建占位符数组）

**使用场景：**

- 多输入 elementwise 操作（add、mul 等）
- 张量广播计算
- CUDA kernel 中的高效内存访问模式
