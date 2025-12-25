## 主要功能分析

**cuBLAS Handle 池管理**
- 维护 cuBLAS 和 cuBLASLt 句柄的线程局部池，避免频繁创建销毁
- 每个线程获取独立的 PoolWindow，线程退出时自动释放句柄

**工作空间管理**
- 为 cuBLAS 操作分配和缓存 GPU 工作空间内存
- 通过环境变量 `CUBLAS_WORKSPACE_CONFIG` 配置大小，支持 Hopper 架构特殊配置（32MiB）
- 使用 handle + stream 组合作为键缓存工作空间，复用同一流的内存

**流和数学模式设置**
- `getCurrentCUDABlasHandle()` 获取当前线程的 cuBLAS 句柄，自动设置当前 CUDA 流
- 根据 `allow_tf32` 标志启用 TF32 加速（AMPERE+ 架构）
- 处理 CUDA context 不存在的边界情况

**核心函数**：
- `getCurrentCUDABlasHandle()` - 获取线程局部的 cuBLAS 句柄
- `getCurrentCUDABlasLtHandle()` - 获取 cuBLASLt 句柄（CUDA 上别名为 cuBLAS）
- `parseChosenWorkspaceSize()` - 解析配置参数计算工作空间大小
- `getNewWorkspace()` - 通过 CUDA 缓存分配器申请新工作空间

**关键设计**：
- 使用"泄漏单例"模式创建全局句柄池，避免静态析构时的 CUDA context 销毁问题
- 线程局部 PoolWindow 实现自动生命周期管理
- 工作空间缓存减少重复分配开销
