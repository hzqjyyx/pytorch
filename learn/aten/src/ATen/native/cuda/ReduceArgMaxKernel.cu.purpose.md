## ReduceArgMaxKernel.cu 功能分析

**核心功能：** 实现 CUDA 上的 argmax 操作，找到张量中最大值的索引位置。

**主要组件：**

- **argmax_kernel_cuda_impl()** (第20-27行)
  - 通用的 argmax 实现模板
  - 使用 `gpu_reduce_kernel` 执行 GPU 规约操作
  - `ArgMaxOps` 定义比较逻辑
  - 初始值为数据类型的下界和索引 0

- **argmax_kernel_cuda()** (第29-42行)
  - 公共入口函数，处理不同数据类型
  - 特殊处理 float16 和 bfloat16：转换为 float 后在 float 精度下计算，避免实现额外的 is_nan 和 warp 操作
  - 其他数据类型使用 `AT_DISPATCH_ALL_TYPES` 宏分发

- **REGISTER_DISPATCH** (第44行)
  - 将 `argmax_kernel_cuda` 注册到 `argmax_stub` 调度器

**关键点：**

- 头文件包含了规约框架 (`Reduce.cuh`, `ReduceOps.h`) 和张量操作基础设施
- 使用 TensorIterator 处理多维张量的索引和遍历
- 输入为标量张量，输出为 int64_t 索引

**工作流：**

1. 接收 TensorIterator (定义输入输出张量和迭代方式)
2. 根据输入数据类型选择合适的实现
3. 调用 GPU 规约内核执行计算
4. 返回最大值对应的索引

---

- 实现 CUDA argmax 规约操作
- 支持多种数据类型（float、double、int、half、bfloat16 等）
- float16/bfloat16 特殊处理：上升转 float 避免精度问题
- 使用 TensorIterator 统一张量访问接口
- 通过 GPU 并行规约内核高效计算
