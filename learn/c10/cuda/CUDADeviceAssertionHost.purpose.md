# CUDA Device-Side Assertion (DSA) 系统

这两个文件实现了 PyTorch 的 CUDA 设备端断言系统，允许在 GPU kernel 中进行断言检查，并在 CPU 端收集和报告失败信息。

## 核心机制

**统一内存(UVM)通信**
- 使用 CUDA Managed Memory 在 CPU 和 GPU 之间共享断言数据
- `DeviceAssertionsData` 结构体存储在 managed memory 中，双方都可访问
- CPU 端通过 `cudaMallocManaged` 分配，使用 `cudaMemAdvise` 优化访问模式

**循环缓冲区设计**
- `CUDAKernelLaunchRegistry` 维护一个大小为 1024 的循环缓冲区
- 记录每次 kernel 启动的元信息（文件名、函数名、行号、设备、流等）
- 使用 `generation_number` 唯一标识每次启动，防止循环覆盖导致的信息错误

## 主要组件

### 数据结构

**DeviceAssertionData** (c10/cuda/CUDADeviceAssertionHost.h:25-45)
```
- assertion_msg: 断言消息字符串
- filename: 断言所在文件
- function_name: 断言所在函数
- line_number: 行号
- caller: 关联的 kernel 启动 generation number
- block_id[3]: 失败的 thread block 坐标
- thread_id[3]: 失败的线程坐标
```

**DeviceAssertionsData** (c10/cuda/CUDADeviceAssertionHost.h:49-56)
```
- assertion_count: 总断言失败次数
- assertions[10]: 最多存储 10 个断言详情
```

**CUDAKernelLaunchInfo** (c10/cuda/CUDADeviceAssertionHost.h:61-79)
```
- launch_filename/function/linenum: CPU 端启动位置
- launch_stacktrace: 可选的调用栈
- kernel_name: 被启动的 kernel 名称
- device/stream: 运行环境
- generation_number: 唯一标识符
```

### CUDAKernelLaunchRegistry 单例

**初始化** (c10/cuda/CUDADeviceAssertionHost.cpp:194-204)
- 检查所有设备是否支持 managed memory (Pascal 架构及以上，compute capability >= 6)
- 从环境变量读取配置：
  - `PYTORCH_USE_CUDA_DSA`: 启用 DSA
  - `PYTORCH_CUDA_DSA_STACKTRACING`: 启用调用栈追踪
- 为每个设备预分配 UVM assertions 指针槽位

**Kernel 启动注册** (c10/cuda/CUDADeviceAssertionHost.cpp:215-246)
```cpp
uint32_t insert(launch_filename, launch_function, launch_linenum, 
                kernel_name, stream_id)
```
- 记录启动信息到循环缓冲区
- 可选地捕获调用栈 (`c10::get_backtrace()`)
- 返回 generation number 传递给 kernel

**UVM 内存管理** (c10/cuda/CUDADeviceAssertionHost.cpp:267-329)
```cpp
DeviceAssertionsData* get_uvm_assertions_ptr_for_current_device()
```
- 延迟分配：首次在设备上启动 kernel 时才分配 managed memory
- 双重检查锁定模式防止竞态
- 配置内存访问策略：
  - `cudaMemAdviseSetPreferredLocation(cudaCpuDeviceId)`: 优先存放在 CPU
  - `cudaMemAdviseSetAccessedBy(cudaCpuDeviceId)`: CPU 直接映射，避免 page fault

## 故障检测与报告

**检测** (c10/cuda/CUDADeviceAssertionHost.cpp:336-343)
```cpp
bool has_failed() const
```
遍历所有设备的 `uvm_assertions`，检查 `assertion_count > 0`

**信息收集** (c10/cuda/CUDADeviceAssertionHost.cpp:99-192)
```cpp
std::string c10_retrieve_device_side_assertion_info()
```
1. 睡眠 1 秒等待 GPU 完成错误信息写入（避免同步复杂度）
2. 获取 assertions 和 kernel launches 的快照
3. 遍历每个设备，对于每个失败的断言输出：
   - GPU 断言消息和位置（文件:行号）
   - 失败的 thread/block ID
   - 启动该 kernel 的 CPU 代码位置
   - Kernel 名称、设备、流信息
   - 可选的启动调用栈

## 技术要点

**自定义 CUDA 函数避免循环依赖**
- `dsa_get_device_id()` / `dsa_get_device_count()` / `dsa_get_device_compute_capability()`
- 使用 `C10_CUDA_CHECK_WO_DSA` 宏（不触发 DSA）避免初始化死锁

**线程安全**
- `read_write_mutex`: 保护循环缓冲区的读写
- `gpu_alloc_mutex`: 保护 managed memory 分配
- `snapshot()` 加锁复制数据避免竞态

**Generation Number 匹配**
- Kernel 启动时的 generation number 存入 assertions 的 `caller` 字段
- 报告时通过 `generation_number % max_kernel_launches` 匹配启动信息
- 如果循环队列已覆盖，会提示增加 `max_size`

## 使用模式

代码通过宏定义 kernel 参数：
```cpp
#define TORCH_DSA_KERNEL_ARGS \
  DeviceAssertionsData *const assertions_data, \
  uint32_t assertion_caller_id
```

Kernel 内部可使用这些参数记录断言失败（具体实现在设备端代码中）。

---

**ROCm/Backward 相关**:
- 代码中未见 ROCm 特定逻辑，仅依赖 `TORCH_USE_CUDA_DSA` 编译开关
- 无明显向后兼容处理，仅通过运行时环境变量控制启用/禁用
