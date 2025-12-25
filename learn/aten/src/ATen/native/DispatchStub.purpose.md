# DispatchStub 核心功能

这是 PyTorch 的**指令集自适应分发系统**，根据 CPU 硬件能力在运行时选择最优的 SIMD 实现。

## 运行机制

**编译时**：同一个 kernel 函数会被编译多次，每次使用不同的编译器优化标志（如 `-mavx2`, `-mavx512`）

**运行时**：通过 `cpuinfo` 库检测 CPU 支持的指令集，自动选择最快的实现版本

## 核心组件

### 1. CPU 能力检测 (`compute_cpu_capability()`)

aten/src/ATen/native/DispatchStub.cpp:28-117

检测优先级（从高到低）：
- 环境变量 `ATEN_CPU_CAPABILITY` 可强制指定
- x86: AVX512 > AVX2 > DEFAULT
- ARM: SVE256 > DEFAULT
- PowerPC: VSX
- s390x: ZVECTOR

检测要求：
- AVX512: 需要 VL/BW/DQ/FMA3 指令集组合
- AVX2: 需要配合 FMA3
- SVE256: 需要硬件支持 256-bit 向量长度

### 2. 函数指针分发表 (`DispatchStubImpl`)

aten/src/ATen/native/DispatchStub.h:95-211

存储各设备/架构的函数指针：
```cpp
std::atomic<void*> cpu_dispatch_ptr;  // CPU 用原子变量（lazy init）
void* cuda_dispatch_ptr;               // GPU 设备用普通指针
void* hip_dispatch_ptr;
void* mps_dispatch_ptr;                // Apple Silicon
void* mtia_dispatch_ptr;               // Meta 训练加速器
void* xpu_dispatch_ptr;                // Intel GPU
void* privateuse1_dispatch_ptr;        // 自定义设备扩展
```

### 3. 分发逻辑

**CPU 分发** (`try_choose_cpu_impl()` aten/src/ATen/native/DispatchStub.cpp:275-335):
- 按能力等级降序选择实现
- AVX512 缺失时自动降级到 AVX2
- SVE256 缺失时降级到 DEFAULT
- 使用 `memory_order_relaxed` 避免多线程竞态（结果相同）

**设备分发** (`try_get_call_ptr()` aten/src/ATen/native/DispatchStub.cpp:124-213):
```
CPU → 延迟初始化（首次调用时选择架构）
GPU → 直接返回预设指针（CUDA/HIP/MPS/XPU）
```

支持的设备类型硬编码在 143-151 行的白名单中。

### 4. 错误处理

aten/src/ATen/native/DispatchStub.h:75-82

使用 `std::variant<void*, ErrorType>` 返回：
- `void*`: 成功获取函数指针
- `ErrorType::MissingDeviceKernel`: kernel 未实现
- `ErrorType::DeviceNotSupported`: 设备类型不支持

## 使用模式

**声明** (在头文件):
```cpp
using my_fn = void(*)(const Tensor&);
DECLARE_DISPATCH(my_fn, my_kernel_stub);
```

**定义** (在 .cpp):
```cpp
DEFINE_DISPATCH(my_kernel_stub);
```

**注册实现** (在 cpu/xxx.cpp):
```cpp
namespace { void kernel_impl(const Tensor& x) {...} }
REGISTER_DISPATCH(my_kernel_stub, &kernel_impl);
```

**调用**:
```cpp
my_kernel_stub(kCPU, tensor);  // 自动选择最优实现
```

## 关键设计细节

### CPU_CAPABILITY 宏系统

aten/src/ATen/native/DispatchStub.h:464-475

- 同一个 `.cpp` 源文件会被编译成多个目标文件
- `CPU_CAPABILITY` 在编译时定义为 `DEFAULT/AVX2/AVX512` 等
- `REGISTER_DISPATCH` 宏会根据当前编译的 `CPU_CAPABILITY` 注册到对应槽位

### AVX512 特殊处理

aten/src/ATen/native/DispatchStub.cpp:301-306, 362-368

某些 kernel 在 Windows 上测试不稳定，可能不提供 AVX512 实现（传入 `nullptr`）。此时自动降级使用 AVX2 版本。

### 线程安全

CPU 指针使用 `std::atomic` + `memory_order_relaxed`：
- 允许多线程同时初始化（计算结果相同）
- 避免锁开销
- 不保证可见性顺序（但对分发逻辑无影响）

---

**ROCm 相关**：
- `RegisterHIPDispatch` 目前重定向到 `cuda_dispatch_ptr`（350-352 行）
- 未来计划切换到独立的 `hip_dispatch_ptr`

**Backward 相关**：
- 该文件不直接处理反向传播
- 仅提供基础设施，具体 backward kernel 由各算子独立注册
