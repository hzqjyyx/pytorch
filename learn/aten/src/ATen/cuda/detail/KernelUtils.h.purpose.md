- **CUDA 网格步幅循环宏** (`CUDA_KERNEL_LOOP_TYPE` 和 `CUDA_KERNEL_LOOP`)：用于在 CUDA 核函数中实现网格步幅循环，允许线程处理多个元素。使用 `int64_t` 防止循环增量溢出。

- **线程配置常量** (`CUDA_NUM_THREADS`)：定义每个块使用 1024 个线程，需要 CUDA 计算能力 2.0 或更高。

- **块数计算函数** (`GET_BLOCKS`)：根据总元素数 N 和每块最大线程数计算所需的 CUDA 块数。使用向上取整除法避免整数溢出，并验证块数不超过 int 最大值。

- **命名空间**：所有代码位于 `at::cuda::detail` 命名空间中。
