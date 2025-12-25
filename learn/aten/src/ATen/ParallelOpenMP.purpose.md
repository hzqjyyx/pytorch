# ParallelOpenMP 文件功能分析

## ParallelOpenMP.h
定义了 OpenMP 并行计算的核心模板函数 `invoke_parallel`：

- **功能**：在指定范围 `[begin, end)` 内并行执行函数 `f`
- **粒度控制**：通过 `grain_size` 参数动态调整任务数量，避免创建过多线程
- **线程分配**：将范围分割成多个 chunk，每个线程处理一个 chunk
- **异常处理**：使用原子标志捕获异常，确保只有第一个异常被重新抛出
- **线程ID追踪**：通过 `ThreadIdGuard` 在每个线程设置其 ID

## ParallelOpenMP.cpp
实现了线程池和并行策略的管理函数：

### 核心全局状态
- `num_threads`：原子变量，存储用户设置的线程数
- `this_thread_id`：线程局部变量，记录当前线程 ID

### 主要函数

**`init_num_threads()`**
- 初始化 OpenMP 线程数
- 如果用户设置过线程数，使用该值；否则根据 MKL 配置或默认值初始化

**`set_num_threads(int nthreads)`**
- 验证输入（必须为正数）
- 更新全局 `num_threads` 变量
- 调用 `omp_set_num_threads()` 配置 OpenMP
- 同步更新 MKL 和 pthreadpool 的线程数
- 清空 MKLDNN 计算缓存（避免线程数变化导致的状态不一致）

**`get_num_threads()`**
- 返回当前 OpenMP 最大线程数
- 调用 `lazy_init_num_threads()` 确保初始化完成

**`get_thread_num()`**
- 返回当前线程的 ID

**`in_parallel_region()`**
- 判断当前是否在 OpenMP 并行区域内

**`intraop_launch()` 和 `intraop_launch_future()`**
- 在 OpenMP 模式下直接内联执行函数（不额外创建线程）

---

## 快速总结

• **线程管理**：维护全局线程配置，支持用户自定义线程数

• **MKL 同步**：确保 OpenMP 和 MKL 线程数一致，防止性能衰退

• **并行执行**：提供范围并行化模板，支持异常传播和线程 ID 追踪

• **跨库协调**：统一管理 OpenMP、MKL、pthreadpool 的线程配置

• **缓存管理**：线程数变化时清空 MKLDNN 缓存以保证正确性
