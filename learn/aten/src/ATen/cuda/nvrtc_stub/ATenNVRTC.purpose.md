## 主要功能

这两个文件实现了 PyTorch 对 NVRTC（NVIDIA Runtime Compilation）和 CUDA 驱动 API 的**动态加载机制**。

### 设计目的

- ATen 不直接链接 libnvrtc 和 libcuda，因为这些库需要 GPU 驱动安装
- 通过动态加载，即使驱动未安装，CPU-only 构建也能正常工作
- 使用 lazily loading 的方式，只在真正需要时才加载这些库

### 技术实现

**ATenNVRTC.h：** 定义了一个 `NVRTC` 结构体，包含所有需要使用的 NVRTC 和驱动 API 的**函数指针**

关键宏定义：
- `AT_FORALL_NVRTC_BASE` — 基础的 nvrtc 编译和 cu 驱动函数（如 `cuModuleLoadData`、`cuLaunchKernel`）
- `AT_FORALL_NVRTC_EXTENDED` — CUDA 12.0+ 扩展函数（`cuTensorMapEncodeTiled`）
- `AT_FORALL_NVRTC` — CUDA 11.0.1+ 额外函数（`nvrtcGetCUBIN`）

**ATenNVRTC.cpp：** 实现 `load_nvrtc()` 函数，**动态绑定**所有函数指针到真实的库函数

### 关键设计约束

- **禁止直接调用** nvrtc* 或 cu* 函数
- 必须通过 `detail::getCUDAHooks().nvrtc()` 或 `globalContext().getNVRTC()` 访问
- 新增 API 需同时修改两个文件

### 功能清单

- **编译相关**：版本查询、程序创建/销毁、PTX/CUBIN 生成、编译、错误信息
- **模块加载**：加载编译后的 GPU 代码到显存
- **核函数执行**：启动普通/合作核函数、设置属性、获取占用率
- **上下文管理**：CUDA 上下文切换和主上下文管理
- **链接阶段**：JIT 链接（cuLink*）
