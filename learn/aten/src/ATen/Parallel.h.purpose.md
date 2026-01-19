## ATen/Parallel.h 主要功能

这个头文件提供了 PyTorch 的并行计算基础设施，用于在 ATen（A Tensor Library）中实现多线程并行操作。

### 核心组件

**1. 线程管理 API**
- `set_num_threads(int)` / `get_num_threads()` - 设置和获取并行区域使用的线程数
- `get_thread_num()` - 获取当前线程编号（从0开始）
- `in_parallel_region()` - 检查代码是否在并行区域中运行
- `init_num_threads()` - 新线程初始化时调用

**2. 并行循环 - `parallel_for`** (aten/src/ATen/Parallel.h:76-80)
```cpp
template <class F>
void parallel_for(int64_t begin, int64_t end, int64_t grain_size, const F& f)
```
- 将 `[begin, end)` 范围分块并行执行用户函数
- `grain_size` 控制每个块的元素数量，影响并行度
- 函数签名：`void f(int64_t begin, int64_t end)`
- **重要限制**：不会复制线程局部状态，函数体内不能使用 Tensor 操作，只能使用数据指针

**3. 并行归约 - `parallel_reduce`** (aten/src/ATen/Parallel.h:121-127)
```cpp
template <class scalar_t, class F, class SF>
scalar_t parallel_reduce(int64_t begin, int64_t end, int64_t grain_size, 
                         scalar_t ident, const F& f, const SF& sf)
```
- 并行计算范围内的归约操作（如求和、求最大值等）
- `ident` - 归约操作的单位元（如加法的0，乘法的1）
- `f` - 对子范围进行归约的函数
- `sf` - 合并两个部分结果的函数
- 工作原理：将数据分块 → 各块并行归约 → 合并部分结果
- 同样不能在函数体内使用 Tensor 操作

**4. 线程间并行（Inter-op Parallelism）**
- `set_num_interop_threads(int)` / `get_num_interop_threads()` - 管理操作间并行的线程数
- `launch(std::function<void()>)` - 启动线程间并行任务

**5. 线程内并行（Intra-op Parallelism）**
- `intraop_launch(const std::function<void()>&)` - 启动操作内并行任务
- `intraop_default_num_threads()` - 获取默认的操作内线程数

**6. 内部工具**
- `lazy_init_num_threads()` - 首次并行调用时延迟初始化线程数
- `ThreadIdGuard` - RAII 风格的线程 ID 管理类，自动恢复旧的线程 ID
- `divup(x, y)` - 向上整除辅助函数：`(x + y - 1) / y`

### 后端实现

根据编译配置选择不同的并行后端：
- `AT_PARALLEL_OPENMP` → 使用 OpenMP 实现
- `AT_PARALLEL_NATIVE` → 使用原生线程池实现
- 具体实现在 `Parallel-inl.h` 中

### 关键设计特点

1. **模板化设计**：`parallel_for` 和 `parallel_reduce` 都是模板函数，支持任意函数对象
2. **粒度控制**：通过 `grain_size` 参数平衡并行开销和并行度
3. **线程安全限制**：明确警告不能在并行函数体内使用 Tensor 操作，只能操作原始数据指针
4. **两级并行**：区分 inter-op（操作间）和 intra-op（操作内）并行

---

**ROCm 相关**：无

**Backward 相关**：无
