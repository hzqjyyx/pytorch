# LazyNVRTC 主要功能分析

这两个文件实现了 NVIDIA CUDA Runtime Compilation (NVRTC) 和 CUDA Driver API 的**延迟加载机制**。

## 核心设计思想

通过动态加载共享库，而非编译时链接，使 PyTorch 能够在运行时按需加载 NVRTC 和 CUDA 驱动库。这样即使系统没有安装 CUDA，程序也能编译和启动。

## 关键组件

### 1. 动态库管理 (aten/src/ATen/cuda/detail/LazyNVRTC.cpp:12-83)

**getCUDALibrary()** - 加载 CUDA 驱动库
- Windows: `nvcuda.dll`
- Linux: `libcuda.so.1`

**getNVRTCLibrary()** - 加载 NVRTC 编译器库
- 根据 CUDA 版本动态确定库名称
- **版本命名规则**:
  - CUDA < 11.3: `libnvrtc.so.{major}.{minor}` (Linux) / `nvrtc64_{major}{minor}_0.dll` (Windows)
  - CUDA 11.3-11.x: `libnvrtc.so.11.2` / `nvrtc64_112_0.dll`
  - CUDA >= 12.0: `libnvrtc.so.{major}` / `nvrtc64_{major}0_0.dll`
- Linux 支持备用库名 `libnvrtc-{hash}.so.{version}`

### 2. 函数存根 (Stub) 机制 (aten/src/ATen/cuda/detail/LazyNVRTC.cpp:85-287)

**工作流程**:
1. 首次调用函数时，存根被触发
2. 通过 `DynamicLibrary.sym()` 查找动态库中的真实函数符号
3. 将函数指针存储到 `lazyNVRTC` 结构体中
4. 调用真实函数并返回结果
5. 后续调用直接使用缓存的函数指针

**宏定义模板**:
- `_STUB_N`: 生成 N 个参数的存根函数
- `NVRTC_STUBN`: NVRTC 函数存根（返回 `nvrtcResult`）
- `CUDA_STUBN`: CUDA Driver API 存根（返回 `CUresult`）

### 3. 包装的函数

**NVRTC 编译器函数**:
- `nvrtcCreateProgram`: 创建 CUDA C++ 程序对象
- `nvrtcCompileProgram`: 编译为 PTX/CUBIN
- `nvrtcGetPTX/GetCUBIN`: 获取编译产物
- `nvrtcGetLoweredName`: 获取 mangled 名称
- `nvrtcGetErrorString`: 错误信息

**CUDA Driver API**:
- 模块管理: `cuModuleLoadData/Ex`, `cuModuleUnload`, `cuModuleGetFunction`
- 内核启动: `cuLaunchKernel`, `cuLaunchCooperativeKernel`
- JIT 链接: `cuLinkCreate`, `cuLinkAddData`, `cuLinkComplete`
- 上下文: `cuCtxGetCurrent`, `cuCtxSetCurrent`
- 占用率: `cuOccupancyMaxActiveBlocksPerMultiprocessor`
- 函数属性: `cuFuncSetAttribute`, `cuFuncGetAttribute`
- CUDA 12.0+: `cuTensorMapEncodeTiled` (用于 TMA - Tensor Memory Accelerator)

### 4. lazyNVRTC 全局对象 (aten/src/ATen/cuda/detail/LazyNVRTC.cpp:291-295)

- 存储所有函数指针的单例结构体
- 初始化时指向存根函数
- 首次调用后替换为真实函数指针

## 技术细节

**错误处理**: 如果无法找到符号，抛出 `std::runtime_error`

**条件编译**:
- `CUDA_VERSION >= 11010`: 启用 CUBIN 支持
- `CUDA_VERSION >= 12000`: 启用 Tensor Map 支持
- `NVRTC_SHORTHASH`: Linux 备用库名支持

**特殊函数**: 部分函数（如 `cuLaunchKernel`, `nvrtcCreateProgram`）参数过多，手动实现而非使用宏

---

**ROCm 相关**: 无（此文件仅处理 NVIDIA CUDA）

**Backward 相关**: 无（纯底层 API 封装，不涉及自动微分）
