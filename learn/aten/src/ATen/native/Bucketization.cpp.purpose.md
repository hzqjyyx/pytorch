**主要功能概述**

这个文件实现了 PyTorch 中的 `searchsorted` 和 `bucketize` 两个核心张量操作，运行在 CPU 上。

**核心算法**

- **`cus_lower_bound`** (49-64行): 自定义的下界二分搜索，处理 NaN/Inf 等特殊值，支持通过 sorter 参数进行间接索引
- **`cus_upper_bound`** (70-85行): 自定义的上界二分搜索，逻辑类似
- **`searchsorted_cpu_contiguous`** (88-115行): 核心搜索实现，使用 `at::parallel_for` 进行并行化处理，支持多维张量和标量输入

**调度与类型处理**

- **`dispatch`** (117-140行): 根据输出类型标志 (`out_int32`) 分发到不同的类型实现，支持所有数值类型及 Half/BFloat16

**公开接口 (searchsorted)**

- `searchsorted_out_cpu(Tensor&, Tensor&, ...)` - 张量输入版本，带输出参数
- `searchsorted_out_cpu(Tensor&, Scalar&, ...)` - 标量输入版本，转换为张量后调用上一版本
- `searchsorted_cpu(...)` - 无输出参数的张量版本，创建新张量返回
- `searchsorted_cpu(...)` - 无输出参数的标量版本

**公开接口 (bucketize)**

- `bucketize_out_cpu(...)` - 带输出参数版本
- `bucketize_cpu(...)` - 张量版本
- `bucketize_cpu(...)` - 标量版本

**关键特性**

- 支持排序者索引 (`sorter`) 用于未排序的边界张量
- 支持左界/右界搜索 (`right` 参数)
- 支持 int32/int64 两种输出类型
- 支持非连续张量输出（先写入连续副本再复制回去）
- 并行化粒度设置为 200 个元素 (`SEARCHSORTED_GRAIN_SIZE`)
