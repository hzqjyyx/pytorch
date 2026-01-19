## ParallelThreadPoolNative.cpp 文件分析

这个文件实现了 PyTorch 的原生线程池管理机制，用于处理跨操作（inter-op）的并行任务。

**核心组件：**

- **全局状态管理**：`num_interop_threads` 原子变量追踪线程数设置状态，使用状态机（NOT_SET → 正数 → CONSUMED）防止重复设置

- **线程池单例**：`get_pool()` 函数创建并返回全局 `PTThreadPool` 实例，采用懒初始化模式

- **线程数配置**：
  - `set_num_interop_threads()` - 在线程池启动前设置线程数（只能调用一次）
  - `get_num_interop_threads()` - 获取当前线程数，支持默认值回退

- **任务启动**：
  - `launch_no_thread_state()` - 直接向线程池提交任务
  - `launch()` - 包装版本，自动捕获并传递线程本地状态（ThreadLocalState）

- **工厂注册**：`create_c10_threadpool()` 通过 ThreadPoolRegistry 注册，支持动态线程池创建

**关键特性：**

- 线程安全的原子操作确保配置一致性
- 线程本地状态自动传播到工作线程
- 支持编译时选项切换（单线程池 vs 多线程池）
- 条件编译保护（仅在 OpenMP 或原生线程池启用时编译）
