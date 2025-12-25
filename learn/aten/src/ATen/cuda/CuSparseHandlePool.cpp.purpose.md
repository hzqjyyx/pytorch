## CuSparseHandlePool.cpp 功能分析

这个文件实现了 cuSPARSE 句柄池管理机制，用于高效地创建和重用 CUDA 稀疏矩阵操作的句柄。

**核心功能：**

- **句柄创建** (`createCusparseHandle`):  调用 `cusparseCreate()` 初始化 cuSPARSE 句柄

- **句柄销毁** (`destroyCusparseHandle`):  销毁句柄，但在 `NO_CUDNN_DESTROY_HANDLE` 宏定义时跳过销毁操作（规避 atexit 时的 CUDA 上下文销毁顺序问题）

- **句柄池类型定义**:  基于 `DeviceThreadHandlePool` 模板创建 `CuSparsePoolType`，管理 cusparseHandle_t 对象的生命周期

- **获取当前设备句柄** (`getCurrentCUDASparseHandle`):  
  - 获取当前 CUDA 设备索引
  - 使用线程局部存储 (thread_local) 维护每个线程的句柄池窗口
  - 懒初始化池对象避免 Windows 上的初始化问题
  - 从池中预留句柄并设置当前 CUDA 流
  - 线程终止时自动释放句柄回到池

**设计亮点：**

- 线程安全的句柄复用机制
- 避免频繁创建/销毁句柄的性能开销
- 处理 CUDA 上下文生命周期的边界问题
