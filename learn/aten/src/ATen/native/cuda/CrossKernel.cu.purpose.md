这个文件实现了 PyTorch 中**向量叉积（Cross Product）** 的 CUDA 核心计算。

**主要功能分解：**

- **`cross_kernel`**（第 10-34 行）：GPU 核心函数，对每个元素并行计算两个 3D 向量的叉积
  - 通过 `CUDA_KERNEL_LOOP` 遍历所有元素
  - 从三个输入张量（输出、x1、x2）中获取行数据
  - 执行标准叉积公式：`(a×b) = [a₁b₂-a₂b₁, a₂b₀-a₀b₂, a₀b₁-a₁b₀]`
  - 支持不同的步长（stride）处理非连续内存布局

- **`launch_cross_kernel`**（第 36-62 行）：GPU 内核启动包装器
  - 计算网格和线程块配置
  - 支持多种数据类型（float、double、complex、half、bfloat16）
  - 处理步长溢出：当步长乘以 2 超过 int32 范围时使用 int64；否则转换为 int32 优化性能

- **`cross_impl`**（第 64-88 行）：主接口函数
  - 提取三个张量在指定维度的步长
  - 使用 `TensorIteratorConfig` 构建高效的张量遍历器
  - 处理空张量情况
  - 支持 32 位和 64 位索引两种寻址模式

- **`REGISTER_DISPATCH`**（第 90 行）：将实现注册到调度系统，使高层 API 能调用此实现
