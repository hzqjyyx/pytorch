这个文件是 PyTorch 内存高效注意力机制 (Memory-Efficient Attention) 的 CUDA 调试工具集合。

**主要功能：**

- **NaN/Inf 检测** (`NANCHECK`): 宏定义，用于在调试时检查张量片段中是否存在 NaN 或无穷大值

- **有限线程打印** (`PRINT_B0_T0`, `PRINT_T0`): 仅在第一个线程块的第一个线程上执行 printf，避免并行重复输出

- **全局线程追踪** (`PRINT_TX_LX`): 遍历所有线程网格配置，打印每个特定线程的调试信息

- **类型名提取** (`__get_type_name`): 编译期模板元编程，使用 `__PRETTY_FUNCTION__` 获取泛型类型的可读名称

- **数组打印** (`PRINT_ACCUM8_T0_L0*`, `PRINT_ARRAY_T0_L0*`): 以 8 元素为单位打印累积器或数组内容

- **张量矩阵打印** (`PRINT_TENSOR4x4_T0_L0*`): 打印 4×4 矩阵切片，用于验证计算结果

- **问题规模打印** (`PRINT_PROBLEM_SIZE`): 输出矩阵运算的维度信息 (m, n, k)

- **Warp 级累积器打印** (`print_warp_accum`): 函数模板，按行列遍历 warp 级并行数据，逐元素打印计算结果
