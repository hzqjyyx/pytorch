这个文件主要负责 ATen 的并行计算配置和信息收集。

## 核心功能

### 1. 环境变量读取
提供了两个辅助函数来读取环境变量：
- `get_env_var()` - 读取字符串类型的环境变量
- `get_env_num_threads()` - 读取并验证线程数相关的环境变量（非移动端）

### 2. 并行信息收集 (`get_parallel_info()`)
生成一个详细的并行配置报告，包括：
- 当前线程数设置（`at::get_num_threads()`, `at::get_num_interop_threads()`）
- OpenMP 版本和最大线程数
- MKL 版本和最大线程数（x86_64 平台）
- MKL-DNN 版本
- 硬件并发数（`std::thread::hardware_concurrency()`）
- 环境变量状态（`OMP_NUM_THREADS`, `MKL_NUM_THREADS`）
- 当前使用的并行后端（OpenMP 或 native thread pool）

### 3. 默认线程数计算 (`intraop_default_num_threads()`)
确定 intraop 操作的默认线程数，优先级顺序：
1. 读取 `OMP_NUM_THREADS` 环境变量
2. 读取 `MKL_NUM_THREADS` 环境变量
3. 如果都未设置：
   - **Apple Silicon 特殊处理**：通过 `sysctlbyname("hw.perflevel0.physicalcpu")` 获取性能核心数，限制并行算法只使用性能核心
   - FBCODE ARM64 平台：默认 1 线程
   - 其他平台：使用 `TaskThreadPoolBase::defaultNumThreads()`

移动端（C10_MOBILE）会抛出错误，因为移动端的线程池大小应该由 cpuinfo 决定。

---

**其他相关内容：**
- 支持 MKL、OpenMP、MKL-DNN 等并行库的集成
- 包含移动端和实验性单线程池的条件编译支持
