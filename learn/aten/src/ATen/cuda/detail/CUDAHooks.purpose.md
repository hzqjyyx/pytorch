# CUDAHooks 核心功能分析

## 架构设计

CUDAHooks 是 `CUDAHooksInterface` 的具体实现，采用 Hooks 模式将 CUDA 功能注入到 ATen 核心库中。通过这种设计，ATen 可以在没有 CUDA 的情况下编译，只在运行时按需加载 CUDA 功能。

**注册机制** (aten/src/ATen/cuda/detail/CUDAHooks.cpp:473):
```cpp
REGISTER_CUDA_HOOKS(CUDAHooks)
```
通过宏将 CUDAHooks 注册到全局 registry，使其成为 CUDA 功能的入口点。

## 初始化流程

**init() 方法** (aten/src/ATen/cuda/detail/CUDAHooks.cpp:86-103):

1. **使用日志记录**: 标记 CUDA 已被使用
2. **环境变量设置**: 设置 `CUDA_MODULE_LOADING=LAZY` 以延迟加载 CUDA 模块
3. **设备初始化**: 
   - 获取设备数量（确保非零）
   - 初始化 CUDA 内存分配器
   - 初始化 P2P 访问缓存
4. **MAGMA 初始化**: 如果启用 MAGMA（线性代数库），调用其初始化函数

## 内存管理

**Pinned Memory 检测** (aten/src/ATen/cuda/detail/CUDAHooks.cpp:117-148):

`isPinnedPtr()` 检查指针是否指向 pinned (page-locked) 内存：
- 优化：如果已有设备存在 primary context，切换到该设备以避免创建新 context
- 使用 `cudaPointerGetAttributes()` 查询内存类型
- 错误处理：`cudaErrorInvalidValue` 表示非 CUDA 指针，返回 false

**分配器访问** (aten/src/ATen/cuda/detail/CUDAHooks.cpp:248-254):
- `getPinnedMemoryAllocator()`: 返回 pinned memory 分配器
- `getCUDADeviceAllocator()`: 返回设备内存分配器

## Context 管理

**Primary Context 检测** (aten/src/ATen/cuda/detail/CUDAHooks.cpp:60-69):

```cpp
bool _hasPrimaryContext(DeviceIndex device_index) {
  unsigned int ctx_flags = 0;
  int ctx_is_active = 0;
  AT_CUDA_DRIVER_CHECK(nvrtc().cuDevicePrimaryCtxGetState(
      device_index, &ctx_flags, &ctx_is_active));
  return ctx_is_active == 1;
}
```

通过 CUDA Driver API 检查设备是否已创建 primary context。关键细节：
- `ctx_is_active` 必须初始化为 0，否则可能得到垃圾值
- 使用 NVRTC wrapper 调用底层 driver API

**回调注册** (aten/src/ATen/cuda/detail/CUDAHooks.cpp:72-79):

使用静态初始化器将 `_hasPrimaryContext` 函数指针注册到 c10::cuda，实现跨模块通信：
```cpp
struct _Initializer {
  _Initializer() {
    c10::cuda::_internal::setHasPrimaryContext(_hasPrimaryContext);
  }
  ~_Initializer() {
    c10::cuda::_internal::setHasPrimaryContext(nullptr);
  }
} initializer;
```

## NVRTC (Runtime Compilation)

**动态加载机制** (aten/src/ATen/cuda/detail/CUDAHooks.cpp:194-222):

根据编译配置采用不同策略：
- **USE_DIRECT_NVRTC**: 直接加载 NVRTC
- **标准 CUDA**: 使用 lazy loading wrapper (`lazyNVRTC`)
- **其他平台**: 动态加载 `libcaffe2_nvrtc` 库并调用其 `load_nvrtc()` 函数

通过静态变量确保只加载一次：
```cpp
const at::cuda::NVRTC& nvrtc() {
  static auto handle = load_nvrtc();
  return *handle.second;
}
```

## 设备信息查询

**设备操作** (aten/src/ATen/cuda/detail/CUDAHooks.cpp:228-235, 442-448):

- `current_device()`: 获取当前设备，失败返回 -1 而非抛异常
- `deviceCount()`: 返回可用 GPU 数量
- `getCurrentDevice()`: 当前活动设备索引

**能力查询** (aten/src/ATen/cuda/detail/CUDAHooks.cpp:264-300):

- `supportsDepthwiseConvolutionWithCuDNN()`: 检查是否为 Volta 架构（compute capability >= 7.0）
- `supportsBFloat16ConvolutionWithCuDNNv8()`: 检查是否为 Ampere 架构（>= 8.0）

## 库版本管理

**版本信息** (aten/src/ATen/cuda/detail/CUDAHooks.cpp:302-326):

- `versionCuDNN()`: 返回编译时 CUDNN_VERSION
- `versionCUDART()`: 返回编译时 CUDART_VERSION
- `hasCUDART()`: 检查是否可用 CUDA Runtime

**配置展示** (aten/src/ATen/cuda/detail/CUDAHooks.cpp:328-408):

`showConfig()` 生成详细配置字符串：
- Runtime 版本（查询 `cudaRuntimeGetVersion()`）
- 编译时版本（如果不同）
- NVCC 架构标志
- CuDNN 版本及其 CUDA 依赖
- MAGMA 版本

版本格式化处理 CUDA 版本号（例如 11020 → 11.2）。

## 随机数生成

**Generator 管理** (aten/src/ATen/cuda/detail/CUDAHooks.cpp:105-111):

- `getDefaultGenerator()`: 返回指定设备的默认 CUDA 随机数生成器（单例）
- `getNewGenerator()`: 创建新的独立生成器实例

## cuFFT 缓存管理

**Plan Cache 操作** (aten/src/ATen/cuda/detail/CUDAHooks.cpp:419-433):

提供 cuFFT plan cache 的完整管理接口：
- `cuFFTGetPlanCacheMaxSize()`: 获取最大缓存大小
- `cuFFTSetPlanCacheMaxSize()`: 设置最大缓存大小
- `cuFFTGetPlanCacheSize()`: 获取当前缓存大小
- `cuFFTClearPlanCache()`: 清空缓存

所有操作都转发到 `at::native::detail` 的实现。

## 设备同步

**deviceSynchronize()** (aten/src/ATen/cuda/detail/CUDAHooks.cpp:464-467):

```cpp
void deviceSynchronize(DeviceIndex device_index) const {
  at::DeviceGuard device_guard(Device(DeviceType::CUDA, device_index));
  c10::cuda::device_synchronize();
}
```

使用 DeviceGuard 确保在正确设备上同步，自动恢复原设备。

---

## ROCm 相关
- `hasROCM()`: 检查是否为 ROCm 构建
- `compiledWithMIOpen()`: MIOpen (ROCm 的 cuDNN 等价物) 编译状态
- `isGPUArch()`: 检查 AMD GPU 架构名称匹配
- 配置信息显示 HIP/MIOpen 版本而非 CUDA/cuDNN
