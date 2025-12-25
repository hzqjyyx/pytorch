## ParallelNative.cpp 和 ParallelNative.h 功能分析

这两个文件实现了 PyTorch 的原生并行后端，用于在多线程环境下执行并行任务。

### 核心功能

**线程池管理**
- 通过 `_get_intraop_pool()` 创建和管理一个全局的 C10 线程池
- 线程数由用户设置或使用系统默认值
- 采用状态机模式：NOT_SET → 用户设置值 → CONSUMED（池已初始化）

**并行任务执行**
- `invoke_parallel()` 函数分割工作范围为多个任务块，分配给线程池执行
- 根据 grain_size（最小块大小）和线程数自动计算任务数量和块大小
- 采用 RAII 的 `ParallelRegionGuard` 跟踪线程执行上下文

**线程上下文管理**
- 维护 thread-local 变量：
  - `in_parallel_region_`：标记当前线程是否在并行区域
  - `thread_num_`：当前线程在并行任务中的 ID
- 提供 `get_thread_num()` 和 `in_parallel_region()` 查询接口

**异常处理**
- 使用原子标志和异常指针捕获任务中的异常
- 等待所有任务完成后重新抛出首个捕获的异常

**线程数控制**
- `set_num_threads()`：在池初始化前设置线程数，初始化后只能警告
- `get_num_threads()`：返回当前有效线程数
- `init_num_threads()`：初始化时禁用 OpenMP 和 MKL 的多线程，避免竞争

**任务启动接口**
- `intraop_launch()`：异步执行单个函数，如果已在并行区域则直接执行
- `intraop_launch_future()`：返回 Future 对象的异步任务接口

### 关键设计
- C10_MOBILE 条件编译用于区分桌面版（C10 线程池）和移动版（Caffe2 PThreadPool）
- 主线程直接执行第一个任务（task_id=0），其余任务由线程池执行
- 使用条件变量同步等待所有任务完成

### 功能总结

- 提供跨平台的任务并行化框架
- 自动负载均衡和线程管理
- 线程安全的上下文查询接口
- 支持嵌套并行和异常处理
