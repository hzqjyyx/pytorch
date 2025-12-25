- **文件用途**：管理 cuSolver 库的 GPU 句柄池，实现线程安全的句柄复用机制

- **核心组件**：
  - `createCusolverDnHandle()`：创建新的 cuSolver 句柄
  - `destroyCusolverDnHandle()`：销毁句柄，处理 CUDA 上下文销毁顺序问题（`NO_CUDNN_DESTROY_HANDLE` 宏可跳过实际销毁）
  - `CuSolverDnPoolType`：基于 `DeviceThreadHandlePool` 的句柄池模板类

- **关键函数 `getCurrentCUDASolverDnHandle()`**：
  - 获取当前线程的 cuSolver 句柄
  - 使用静态全局句柄池 + 线程本地窗口实现线程安全
  - 线程终止时自动释放预留的句柄
  - 为获取的句柄设置当前 CUDA 流

- **设计优势**：
  - 避免频繁创建/销毁句柄的开销
  - 线程本地存储避免竞争条件
  - 应对 Windows 初始化问题（延迟初始化）
  - 处理程序退出时 CUDA 上下文可能已销毁的边界情况
