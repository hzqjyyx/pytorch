# ATen Context 核心功能分析

## 全局上下文管理

`Context` 类是 ATen 库的全局配置管理中心，通过单例模式提供：

```cpp
Context& globalContext() {
  static Context globalContext_;
  return globalContext_;
}
```

所有配置都通过 `at::globalContext()` 访问。

## 设备抽象与延迟初始化

### 设备类型抽象

通过 `AcceleratorHooksInterface` 统一管理不同硬件后端（CUDA/XPU/MPS/HIP/HPU/MTIA/PrivateUse1）：

```cpp
const AcceleratorHooksInterface& getAcceleratorHooksInterface(
    std::optional<c10::DeviceType> opt_device_type)
```

根据设备类型返回对应的 Hooks 实现，提供：
- 默认随机数生成器
- 设备数量查询
- 内存管理（pinned memory）
- 设备初始化

### 延迟初始化机制

```cpp
void lazyInitDevice(c10::DeviceType device_type) {
  if (device_type != at::kCPU) {
    c10::call_once(init_[static_cast<int8_t>(device_type)], [&] {
      getAcceleratorHooksInterface(device_type).init();
    });
  }
}
```

使用 `c10::once_flag` 数组确保每种设备类型只初始化一次。

## CuDNN 配置管理

控制 CuDNN 库的行为：

- `userEnabledCuDNN()` / `setUserEnabledCuDNN()`: 启用/禁用 CuDNN
- `benchmarkCuDNN()` / `setBenchmarkCuDNN()`: 自动选择最快算法
- `benchmarkLimitCuDNN()`: 限制 benchmark 候选数量
- `deterministicCuDNN()`: 强制确定性算法
- `allowTF32CuDNN()`: TensorFloat-32 加速

## 确定性算法控制

### 全局确定性设置

```cpp
void setDeterministicAlgorithms(bool b, bool warn_only = false) {
  _deterministic_algorithms = b;
  _deterministic_algorithms_warn_only = warn_only;
}
```

### 确定性检查机制

**通用检查** (`alertNotDeterministic`):
- 确定性模式下，若操作无确定性实现，抛出错误或警告
- 错误信息引导用户使用 `warn_only=True` 或提交 issue

**CuBLAS 特定检查** (`alertCuBLASConfigNotDeterministic`):
- CUDA >= 10.2 时检查 `CUBLAS_WORKSPACE_CONFIG` 环境变量
- 必须设置为 `:4096:8` 或 `:16:8` 才能保证确定性

```cpp
bool checkCuBLASConfigDeterministic() {
  const auto workspace_config = c10::utils::get_env(cublas_config_var_name);
  return (workspace_config == cublas_deterministic_configs[0] || 
          workspace_config == cublas_deterministic_configs[1]);
}
```

## 精度控制

### Float32 矩阵乘法精度

三级精度控制（`Float32MatmulPrecision` 枚举）：
- `HIGHEST`: 完全 FP32（禁用 TF32）
- `HIGH`: TF32 加速
- `MEDIUM`: 更激进的优化

```cpp
bool allowTF32CuBLAS() const {
  return float32_matmul_precision != at::Float32MatmulPrecision::HIGHEST;
}
```

默认值受环境变量 `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` 影响。

### 低精度累加控制

- `allowFP16ReductionCuBLAS()`: FP16 归约
- `allowBF16ReductionCuBLAS()`: BF16 归约  
- `allowFP16AccumulationCuBLAS()`: FP16 累加
- `allowFP16ReductionCPU()`: CPU FP16 归约（仅 ARM 架构 + FP16 指令支持）

CPU FP16 运行时检查：
```cpp
void setAllowFP16ReductionCPU(bool b) {
  if (b && !allow_fp16_reduction_cpu) {
    #if defined(__aarch64__) && !defined(C10_MOBILE)
    if (!cpuinfo_initialize() || !cpuinfo_has_arm_fp16_arith())
    #else
    if (true)
    #endif
      throw std::runtime_error("Float16 arithmetic is not supported by the CPU!");
  }
  allow_fp16_reduction_cpu = b;
}
```

## Scaled Dot-Product Attention (SDP) 后端管理

### 多后端支持

四种实现后端：
1. **Flash Attention**: 内存高效的快速实现
2. **Memory Efficient**: 另一种内存优化实现
3. **Math**: 原始 PyTorch 数学实现
4. **CuDNN Attention**: CuDNN 库实现

### 优先级控制

```cpp
std::array<at::SDPBackend, at::num_sdp_backends> sdp_priority_order = {
  at::SDPBackend::flash_attention,
  at::SDPBackend::efficient_attention,
  at::SDPBackend::math,
  at::SDPBackend::cudnn_attention
};
```

通过 `setSDPPriorityOrder()` 自定义选择顺序。

### 单独启用/禁用

每个后端都有独立开关：
- `setSDPUseFlash()` / `userEnabledFlashSDP()`
- `setSDPUseMemEfficient()` / `userEnabledMemEfficientSDP()`
- `setSDPUseMath()` / `userEnabledMathSDP()`
- `setSDPUseCuDNN()` / `userEnabledCuDNNSDP()`

## 线性代数后端选择

### BLAS 后端

```cpp
at::BlasBackend blasPreferredBackend()
```

智能默认选择逻辑：
- 默认使用 `Cublas`
- AMD Instinct GPU (gfx90a/gfx942/gfx950) 优先使用 `Cublaslt` (hipBLASLt)
- 运行时检查架构兼容性，不支持时自动降级

环境变量：`TORCH_BLAS_PREFER_CUBLASLT` / `TORCH_BLAS_PREFER_HIPBLASLT`

### Linalg 后端

选择 `Cusolver` (cuSOLVER) 或 `Magma`：
- 设置时检查编译支持
- 环境变量：`TORCH_LINALG_PREFER_CUSOLVER` / `TORCH_LINALG_PREFER_HIPSOLVER`

## 量化引擎管理

### 自动选择逻辑

```cpp
at::QEngine qEngine() const {
  static auto _quantized_engine = []() {
    at::QEngine qengine = at::kNoQEngine;
    
    #if defined(C10_MOBILE) && defined(USE_PYTORCH_QNNPACK)
    qengine = at::kQNNPACK;
    #endif
    
    #if AT_MKLDNN_ENABLED()
    qengine = at::kONEDNN;
    #endif
    
    #ifdef USE_FBGEMM
    if (fbgemm::fbgemmSupportedCPU()) {
      qengine = at::kX86;  // 结合 FBGEMM 和 OneDNN
    }
    #endif
    
    return qengine;
  }();
  return quantized_engine.value_or(_quantized_engine);
}
```

优先级：X86 (FBGEMM) > OneDNN > QNNPACK > NoQEngine

### 支持的引擎列表

通过 `supportedQEngines()` 返回编译时可用的引擎列表。

## 其他配置功能

### NNPACK 控制
- `userEnabledNNPACK()` / `setUserEnabledNNPACK()`

### MKL-DNN 控制
- `userEnabledMkldnn()` / `setUserEnabledMkldnn()`
- `deterministicMkldnn()` / `setDeterministicMkldnn()`

### 稀疏张量不变量检查
- `checkSparseTensorInvariants()` / `setCheckSparseTensorInvariants()`

### 预打包权重释放
- `releaseWeightsWhenPrepacking()`: 移动端默认启用

### Vmap Fallback 警告
- `areVmapFallbackWarningsEnabled()` / `setDisplayVmapFallbackWarnings()`

### 移动端 CPU 分配器
- `setDefaultMobileCPUAllocator()` / `unsetDefaultMobileCPUAllocator()`

### 实验性功能
- `_SMCarveout_EXPERIMENTAL()`: 为矩阵乘法预留部分 SM，避免与通信内核冲突

## NoTF32Guard RAII 保护

临时禁用 TF32 的作用域守卫：

```cpp
thread_local bool override_allow_tf32_flag = false;

NoTF32Guard::NoTF32Guard() {
  if (!override_allow_tf32_flag) {
    changed = true;
    override_allow_tf32_flag = true;
  }
}
```

使用场景：某些操作（如 `addmv`）使用 TF32 无性能提升但损失精度。

## 静态能力查询

Context 提供静态方法查询编译时功能：
- `hasCUDA()` / `hasCUDART()` / `versionCUDART()`
- `hasCuDNN()` / `versionCuDNN()` / `hasCuSOLVER()` / `hasCuBLASLt()`
- `hasXPU()` / `hasMPS()` / `hasHPU()` / `hasMTIA()`
- `hasOpenMP()` / `hasMKL()` / `hasMKLDNN()` / `hasLAPACK()`
- `hasKleidiAI()` / `isXNNPACKAvailable()`

---

## 简要补充

**ROCm 相关**：
- `hasROCM()` 检查 ROCm 编译支持
- hipBLAS/hipBLASLt 架构检测与自动降级
- `HIPBLASLT_ALLOW_TF32` 环境变量控制 TF32
- `ROCmFABackend` 选择 Flash Attention 后端（CK vs 默认）

**Backward Pass 相关**：
- `ROCmBackwardPassGuard`: 标记反向传播执行阶段
- `thread_local bool rocm_is_backward_pass`: 线程局部状态
- 用途：算子根据前向/反向选择不同数值或性能特性的实现
