# torch.randn/normal 正态分布随机数生成详解

## 概述

本文档详细介绍了 PyTorch 正态分布随机数生成 API（以 `torch.randn` 和 `torch.normal` 为例）从 Python 用户 API 到底层随机数引擎的完整调用流程。正态分布在深度学习中应用极为广泛，包括权重初始化（Xavier/Kaiming）、噪声注入、数据增强等场景。

**本文重点关注 CUDA 实现**，因为 GPU 加速的随机数生成在深度学习中使用最为广泛，其实现也更加复杂和精巧。

**与 torch.rand 的对比**：
- `torch.rand`: 生成均匀分布 U[0, 1)，使用线性映射变换
- `torch.randn`: 生成标准正态分布 N(0, 1)，使用 **Box-Muller 变换**

## 整体架构

```
用户层 (Python)
    ↓
API绑定层 (Python C Extension)
    ↓
分派层 (C++ Dispatcher)
    ↓
核心实现层 (ATen Native Functions)
    ↓
分布层 (Distribution Templates)
    ↓
随机数生成器引擎层 (RNG Engine)
    ↓
硬件执行层 (CPU: MT19937 + Box-Muller / GPU: Philox + cuRAND)
```

## 1. Python 层入口

### 1.1 用户 API

PyTorch 提供了三种主要的正态分布 API：

```python
# 1. torch.randn - 生成标准正态分布 N(0, 1)
result = torch.randn(3, 4)

# 2. torch.normal - 生成任意正态分布 N(mean, std)
# 2.1 标量参数
result = torch.normal(mean=10.0, std=2.0, size=(3, 4))

# 2.2 张量参数（广播）
means = torch.tensor([1.0, 2.0, 3.0])
stds = torch.tensor([0.1, 0.2, 0.3])
result = torch.normal(means, stds)  # 每个元素不同的参数

# 3. tensor.normal_ - 原地操作
tensor = torch.empty(3, 4)
tensor.normal_(mean=0, std=1)  # 原地填充，避免内存分配

# 使用自定义生成器
generator = torch.Generator()
generator.manual_seed(42)
result = torch.randn(3, 4, generator=generator)

# CUDA 张量
result = torch.randn(3, 4, device='cuda')
```

**API 对比**：

| API | 分布 | 内存分配 | 使用场景 |
|-----|------|----------|----------|
| `torch.randn(size)` | N(0, 1) | 创建新张量 | 最常用，权重初始化 |
| `torch.normal(mean, std, size)` | N(mean, std) | 创建新张量 | 自定义均值/标准差 |
| `tensor.normal_(mean, std)` | N(mean, std) | 原地操作 | 复用缓冲区，高性能 |

### 1.2 API 绑定

这些函数是对 C++ 扩展的直接绑定：
- `torch.randn` → `torch._C.randn`
- `torch.normal` → `torch._C.normal`
- `tensor.normal_` → `torch._C.TensorBase.normal_`

**文件位置**: `torch/_torch_docs.py`

## 2. C++ 分派层

### 2.1 torch.randn 入口函数

**文件**: `aten/src/ATen/native/TensorFactories.cpp:1235`

```cpp
Tensor randn(
    IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::randn(
      size,
      static_cast<std::optional<Generator>>(std::nullopt),
      dtype,
      layout,
      device,
      pin_memory);
}
```

### 2.2 带生成器的版本

**文件**: `aten/src/ATen/native/TensorFactories.cpp:1250`

```cpp
Tensor randn(
    IntArrayRef size,
    std::optional<Generator> generator,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 构建 TensorOptions
  TensorOptions options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);

  // 创建空张量
  auto result = at::empty(size, options);

  // 关键调用：使用 normal_ 填充 N(0, 1) 的标准正态分布
  return result.normal_(0, 1, std::move(generator));
}
```

**主要功能**:
- 构建 `TensorOptions` 对象，封装数据类型、布局、设备等信息
- 创建指定形状的空张量（未初始化）
- 调用 `normal_` 原地操作填充标准正态分布

**与 torch.rand 的对比**:
```cpp
// torch.rand 实现
auto result = at::empty(size, options);
return result.uniform_(0, 1, generator);  // 均匀分布

// torch.randn 实现
auto result = at::empty(size, options);
return result.normal_(0, 1, generator);   // 正态分布
```

### 2.3 输出参数版本

**文件**: `aten/src/ATen/native/TensorFactories.cpp:1270`

```cpp
Tensor& randn_out(
    IntArrayRef size,
    std::optional<Generator> generator,
    Tensor& result) {
  result.resize_(size);
  return result.normal_(0, 1, std::move(generator));
}
```

**优化点**: 不创建新张量，直接使用用户提供的输出张量，避免内存分配

### 2.4 torch.normal 入口函数

**文件**: `aten/src/ATen/native/TensorFactories.cpp:1278`

```cpp
Tensor normal(
    double mean,
    double std,
    IntArrayRef size,
    std::optional<Generator> generator,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  TensorOptions options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);
  auto result = at::empty(size, options);
  return result.normal_(mean, std, std::move(generator));
}
```

**与 randn 的区别**: 允许指定任意 mean 和 std，而不是固定的 N(0, 1)

## 3. 核心分布层

### 3.1 normal_ 实现

**文件**: `aten/src/ATen/native/Distributions.cpp:275`

```cpp
Tensor& normal_(Tensor& self, double mean, double std, std::optional<Generator> gen) {
  return at::native::templates::normal_impl_<NormalStub, Generator>(self, mean, std, std::move(gen));
}
```

这里使用了模板化设计：
- `NormalStub`: 分发器存根，用于分派到 CPU/CUDA 实现
- `Generator`: 随机数生成器类型

### 3.2 normal_impl_ 模板函数

**文件**: `aten/src/ATen/native/DistributionTemplates.h:203`

```cpp
template<template<typename> class normal_kernel, typename RNG>
Tensor& normal_impl_(Tensor& self, double mean, double std, std::optional<Generator> gen) {
  // 检查标准差有效性
  CHECK_NORMAL_STD(std);  // TORCH_CHECK(std >= 0.0)

  CHECK_EMPTY_AND_RETURN(self);

  if (self.is_complex()) {
    // 处理复数张量：实部和虚部独立采样
    auto float_tensor = at::view_as_real(self);
    // 复数正态分布的方差分配：
    // 若 Z ~ CN(μ, σ²)，则 Re(Z), Im(Z) ~ N(μ, σ²/2)
    normal_kernel<RNG>()(float_tensor, mean, std/(std::sqrt(2)), gen);
  } else {
    // 实数张量：直接调用 kernel
    normal_kernel<RNG>()(self, mean, std, gen);
  }
  return self;
}
```

**关键功能**:
- **标准差检查**: 确保 std ≥ 0
- **复数支持**: 复数正态分布的实部和虚部各有 σ²/2 的方差
- **空张量处理**: 空张量直接返回
- **设备分派**: 根据张量设备类型分派到 CPU 或 CUDA 实现

**复数正态分布原理**:
```
复数正态分布 CN(μ, σ²):
  Z = X + iY
  X ~ N(Re(μ), σ²/2)
  Y ~ N(Im(μ), σ²/2)

为什么方差是 σ²/2？
  Var(Z) = Var(X) + Var(Y) = σ²/2 + σ²/2 = σ²
```

## 4. CPU 实现（简介）

CPU 实现使用经典的 Box-Muller 变换，并利用缓存优化。

**文件**: `aten/src/ATen/native/cpu/DistributionKernels.cpp:216`

```cpp
void normal_kernel(const TensorBase &self, double mean, double std, std::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::normal_kernel(self, mean, std, generator);
}
```

**实现细节**:
- 使用互斥锁保护生成器状态
- Box-Muller 变换一次生成两个正态随机数
- **缓存优化**: 缓存第二个样本，下次调用直接使用
- `cpu_serial_kernel` 自动根据张量大小选择串行或并行执行

**CPU Generator 的缓存字段**:
```cpp
struct CPUGeneratorImpl {
  at::mt19937 engine_;
  std::optional<float> next_float_normal_sample_;   // 缓存的 float 正态样本
  std::optional<double> next_double_normal_sample_; // 缓存的 double 正态样本
};
```

**为什么需要缓存？**
```cpp
// Box-Muller 一次生成两个正态随机数
float u1 = uniform();
float u2 = uniform();
float z0 = sqrt(-2 * log(u1)) * cos(2 * PI * u2);
float z1 = sqrt(-2 * log(u1)) * sin(2 * PI * u2);  // 缓存这个

// 下次调用直接使用缓存的 z1，无需重复计算
```

## 5. CUDA 实现（重点）

### 5.1 CUDA normal_kernel 入口

**文件**: `aten/src/ATen/native/cuda/DistributionNormal.cu:8`

```cpp
void normal_kernel(const TensorBase &self, double mean, double std, std::optional<Generator> gen) {
  // 获取或创建 CUDA 生成器
  auto generator = get_generator_or_default<CUDAGeneratorImpl>(gen, cuda::detail::getDefaultCUDAGenerator());

  // 调用模板化的 CUDA 实现
  at::native::templates::cuda::normal_kernel(self, mean, std, generator);
}
```

**关键点**：
- **Generator 的第一次使用**: 在这里将 `std::optional<Generator>` 转换为具体的 `CUDAGeneratorImpl*`
- 如果用户没有提供 generator，使用默认的 CUDA 生成器

### 5.2 CUDA 模板实现

**文件**: `aten/src/ATen/native/cuda/DistributionTemplates.h:460`

```cpp
template<typename RNG>
void normal_kernel(const TensorBase &self, double mean_, double std_, RNG gen) {
  auto iter = TensorIterator::borrowing_nullary_op(self);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                  iter.dtype(), "normal_kernel_cuda", [&] {
    using accscalar_t = at::acc_type<scalar_t, true>;
    auto mean = static_cast<accscalar_t>(mean_);
    auto std = static_cast<accscalar_t>(std_);

    // 定义变换 lambda（在设备端执行）
    auto normal_func = [mean, std] __device__ (accscalar_t rand) {
      return static_cast<scalar_t>(transformation::normal<accscalar_t>(rand, mean, std));
    };

    // 调用 normal_and_transform（CUDA 实现核心）
    normal_and_transform<scalar_t, accscalar_t>(iter, gen, normal_func);
  });
}
```

**CUDA 特殊处理**:
- **混合精度计算**: 使用 `accscalar_t` 进行中间计算，提高精度
  - Half/BFloat16 → float 计算
  - Float → float 计算
  - Double → double 计算
- **设备端 lambda**: `__device__` 变换函数在 GPU 上执行
- **变换函数**: 将标准正态分布 N(0,1) 转换为 N(mean, std)

**transformation::normal 实现**:
```cpp
// 标准化变换：Z ~ N(0,1) → X = σZ + μ ~ N(μ, σ²)
template<typename T>
__device__ inline T normal(T rand, T mean, T std) {
  return rand * std + mean;
}
```

### 5.3 normal_and_transform 实现

**文件**: `aten/src/ATen/native/cuda/DistributionTemplates.h:443`

这是 CUDA 正态分布生成的核心函数，使用 cuRAND 的 Box-Muller 实现。

```cpp
template<typename scalar_t, typename accscalar_t, typename RNG, typename transform_t>
void normal_and_transform(TensorIteratorBase& iter, RNG gen, transform_t transform) {
  if (std::is_same_v<scalar_t, double>) {
    // double 类型：使用 curand_normal2_double，一次生成 2 个 double
    distribution_nullary_kernel<scalar_t, accscalar_t, double2>(iter,
      gen,
      [] __device__ (curandStatePhilox4_32_10_t* state) -> double2 {
        return curand_normal2_double(state);
      },
      transform);
  } else {
    // float/half/bfloat16 类型：使用 curand_normal4，一次生成 4 个 float
    distribution_nullary_kernel<scalar_t, accscalar_t, float4>(iter,
      gen,
      [] __device__ (curandStatePhilox4_32_10_t* state) -> float4 {
        return curand_normal4(state);
      },
      transform);
  }
}
```

**与 uniform_and_transform 的对比**:

| 特性 | uniform_and_transform | normal_and_transform |
|------|----------------------|---------------------|
| **cuRAND 函数** | `curand_uniform4()` | `curand_normal4()` |
| **算法** | 线性映射 | Box-Muller 变换 |
| **输出范围** | (0, 1] | N(0, 1) |
| **计算复杂度** | O(1) 线性 | O(log + trig) 超越函数 |
| **double 版本** | `curand_uniform2_double()` | `curand_normal2_double()` |
| **向量化** | 一次 4 个 float / 2 个 double | 一次 4 个 float / 2 个 double |

**关键设计**:
1. **类型特化**: double 和 float 使用不同的 cuRAND 函数
2. **向量化生成**:
   - `curand_normal4()`: 一次生成 4 个 float 标准正态随机数
   - `curand_normal2_double()`: 一次生成 2 个 double 标准正态随机数
3. **分离关注点**:
   - cuRAND 负责生成 N(0,1) 的标准正态分布
   - `transform` lambda 负责转换为 N(mean, std)

### 5.4 distribution_nullary_kernel（复用）

**文件**: `aten/src/ATen/native/cuda/DistributionTemplates.h:112`

这是 CUDA 随机数生成的最核心函数，正态分布和均匀分布都复用这个 kernel。详细实现参见 [torch.rand 调用流程 - 第 5.4 节](random_call_flow.md#54-distribution_nullary_kernel%EF%BC%88cuda-%E6%A0%B8%E5%BF%83%EF%BC%89)。

**核心职责**:
1. 计算 Grid/Block 配置
2. 从 Generator 获取 Philox 状态（**Generator 的第二次使用**）
3. 启动 CUDA kernel

**与 uniform 的复用**:
```cpp
// uniform 调用
distribution_nullary_kernel<scalar_t, accscalar_t, float4>(iter,
  gen,
  [] __device__ (curandStatePhilox4_32_10_t* state) -> float4 {
    return curand_uniform4(state);  // 均匀分布
  },
  uniform_transform);

// normal 调用
distribution_nullary_kernel<scalar_t, accscalar_t, float4>(iter,
  gen,
  [] __device__ (curandStatePhilox4_32_10_t* state) -> float4 {
    return curand_normal4(state);  // 正态分布
  },
  normal_transform);
```

### 5.5 CUDA Kernel 执行（复用）

**文件**: `aten/src/ATen/native/cuda/DistributionTemplates.h:66`

CUDA kernel 的 Grid-Stride Loop 实现与 uniform 完全相同，只是调用的 cuRAND 函数不同：

```cpp
__global__ void distribution_elementwise_grid_stride_kernel(...) {
  // 1. 初始化 Philox 状态
  auto [seed, offset] = at::cuda::philox::unpack(philox_args);
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);

  // 2. Grid-stride loop
  for(int64_t linear_index = idx; linear_index < rounded_size;
      linear_index += blockDim.x * gridDim.x * unroll_factor) {

    // 3. 生成随机数
    // uniform: auto rand = curand_uniform4(&state);
    // normal:  auto rand = curand_normal4(&state);  // 区别在这里
    auto rand = dist_func(&state);

    // 4. 展开循环并应用变换
    #pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int64_t li = linear_index + blockDim.x * gridDim.x * ii;
      if (li < numel) {
        transform_func(li, static_cast<accscalar_t>((&rand.x)[ii]));
      }
    }
    __syncthreads();
  }
}
```

Grid-Stride Loop 的优势参见 [torch.rand 调用流程 - 第 5.6 节](random_call_flow.md#56-grid-stride-loop-%E7%9A%84%E4%BC%98%E5%8A%BF)。

## 6. Box-Muller 算法原理

Box-Muller 变换是将均匀分布转换为正态分布的经典算法。

### 6.1 基本 Box-Muller 变换

**数学原理**:

给定两个独立的均匀随机数 $U_1, U_2 \sim \text{Uniform}(0, 1)$，可以生成两个独立的标准正态随机数 $Z_0, Z_1 \sim N(0, 1)$：

```
Z₀ = √(-2 ln U₁) · cos(2π U₂)
Z₁ = √(-2 ln U₁) · sin(2π U₂)
```

**代码实现**:
```cpp
// 基本 Box-Muller 变换
float u1 = uniform(0, 1);  // 均匀分布
float u2 = uniform(0, 1);

float r = sqrt(-2.0f * log(u1));
float theta = 2.0f * M_PI * u2;

float z0 = r * cos(theta);  // N(0, 1)
float z1 = r * sin(theta);  // N(0, 1)
```

**为什么有效？** （数学证明见 Appendix A）

### 6.2 Marsaglia 极坐标形式（可选优化）

Marsaglia 提出了一种避免三角函数的优化形式：

```cpp
// Marsaglia 极坐标形式
float x, y, s;
do {
  x = uniform(-1, 1);
  y = uniform(-1, 1);
  s = x*x + y*y;
} while (s >= 1.0f || s == 0.0f);

float scale = sqrt(-2.0f * log(s) / s);
float z0 = x * scale;  // N(0, 1)
float z1 = y * scale;  // N(0, 1)
```

**优势**:
- 避免 cos/sin 计算，使用乘法替代
- 可能更快（取决于硬件）

**劣势**:
- 需要拒绝采样循环（平均拒绝率 ~21%）
- cuRAND 使用标准 Box-Muller

### 6.3 cuRAND 的实现

cuRAND 库提供了高度优化的 Box-Muller 实现：

**curand_normal4 内部实现**:
```cpp
__device__ float4 curand_normal4(curandStatePhilox4_32_10_t* state) {
  // 1. 生成 4 个均匀随机数
  float4 uniform = curand_uniform4(state);  // [u0, u1, u2, u3]

  // 2. 两次 Box-Muller 变换，生成 4 个正态随机数
  // 第一对
  float r0 = sqrtf(-2.0f * logf(uniform.x));
  float theta0 = 2.0f * M_PI * uniform.y;
  float z0 = r0 * cosf(theta0);
  float z1 = r0 * sinf(theta0);

  // 第二对
  float r1 = sqrtf(-2.0f * logf(uniform.z));
  float theta1 = 2.0f * M_PI * uniform.w;
  float z2 = r1 * cosf(theta1);
  float z3 = r1 * sinf(theta1);

  return make_float4(z0, z1, z2, z3);
}
```

**向量化优势**:
- 一次函数调用生成 4 个正态随机数
- 充分利用 GPU 的并行计算能力
- 减少函数调用开销

### 6.4 标准化变换

cuRAND 生成的是标准正态分布 N(0, 1)，需要转换为任意正态分布 N(μ, σ²)：

```cpp
// 标准化公式
X = σZ + μ

// 其中 Z ~ N(0, 1)，则 X ~ N(μ, σ²)
```

**证明**:
```
E[X] = E[σZ + μ] = σE[Z] + μ = σ·0 + μ = μ
Var(X) = Var(σZ + μ) = σ²Var(Z) = σ²·1 = σ²
```

**PyTorch 实现**:
```cpp
auto normal_func = [mean, std] __device__ (accscalar_t rand) {
  return static_cast<scalar_t>(rand * std + mean);
};
```

## 7. 完整调用流程图（CUDA）

```
torch.randn(3, 4, device='cuda')
    ↓
torch._C.randn                                   # Python C扩展绑定
    ↓
at::randn(size, device='cuda', ...)              # C++ API
    ↓
at::empty(size, device='cuda')                   # 创建未初始化 CUDA 张量
    ↓
result.normal_(0, 1, generator)                  # 原地填充标准正态分布
    ↓
at::native::templates::normal_impl_()            # 模板分派
    ↓
    ├─ CHECK_NORMAL_STD(std)                     # 检查 std >= 0
    ├─ 处理复数张量（可选）
    └─ normal_kernel<RNG>()
        ↓
【Generator 第一次使用】
get_generator_or_default<CUDAGeneratorImpl>()    # 获取 CUDA 生成器
    ↓
templates::cuda::normal_kernel()                 # CUDA 模板实现
    ↓
    ├─ 创建 TensorIterator
    ├─ 定义 normal_func: N(0,1) → N(mean, std)
    └─ normal_and_transform()
        ↓
        ├─ 选择 curand_normal4() 或 curand_normal2_double()
        └─ distribution_nullary_kernel()
            ↓
            ├─ calc_execution_policy()          # 计算 grid/block
            │
            ├─【Generator 第二次使用】
            ├─ gen->philox_cuda_state(offset)   # 获取 Philox 状态
            │       ↓
            │   PhiloxCudaState(seed, offset)
            │
            └─ 启动 CUDA kernel <<<grid, block>>>
                ↓
distribution_elementwise_grid_stride_kernel()
    ↓
    ├─ 解包 Philox 状态: [seed, offset]
    ├─ curand_init(seed, threadIdx, offset)     # 初始化 Philox
    │
    └─ Grid-Stride Loop:
        for (idx; idx < numel; idx += stride)
            ↓
            ├─ curand_normal4(&state)           # 生成 4 个标准正态随机数
            │       ↓
            │   Philox 生成 uniform → Box-Muller 变换
            │       ↓
            │   返回 float4{z0, z1, z2, z3} ~ N(0, 1)
            │
            └─ 应用 transform_func
                    ↓
                value = rand * std + mean       # N(0,1) → N(mean, std)
                output[idx] = value
```

**与 torch.rand 流程的对比**:

| 步骤 | torch.rand (uniform) | torch.randn (normal) |
|------|---------------------|---------------------|
| **工厂函数** | `result.uniform_(0, 1)` | `result.normal_(0, 1)` |
| **分布层** | `uniform_impl_()` | `normal_impl_()` |
| **CUDA kernel** | `uniform_kernel()` | `normal_kernel()` |
| **变换函数** | `uniform_and_transform()` | `normal_and_transform()` |
| **cuRAND 函数** | `curand_uniform4()` | `curand_normal4()` |
| **算法** | 线性映射 | **Box-Muller 变换** |
| **输出** | [0, 1) 均匀分布 | N(0, 1) 正态分布 |
| **最终变换** | `rand * range + from` | `rand * std + mean` |

## 8. 与 uniform 的对比

### 8.1 实现差异

| 特性 | uniform | normal |
|------|---------|--------|
| **分布** | 均匀分布 U[a, b) | 正态分布 N(μ, σ²) |
| **cuRAND 函数** | `curand_uniform4` | `curand_normal4` |
| **底层算法** | 线性映射 | Box-Muller 变换 |
| **计算复杂度** | O(1) 线性运算 | O(log + trig) 超越函数 |
| **输出范围** | (0, 1] | (-∞, +∞) |
| **边界处理** | 需要反转 (0,1] → [0,1) | 无需边界处理 |
| **缓存优化** | 无缓存 | CPU 缓存第二个样本 |

### 8.2 性能对比

**理论性能**:
```
uniform: ~10 GFLOPS (线性运算)
normal:  ~5 GFLOPS  (Box-Muller 的 log/cos/sin)
```

**实际性能** (Tesla V100, float32, 10M 元素):
```
torch.rand:  ~2.5 GB/s
torch.randn: ~1.8 GB/s
```

**性能瓶颈**:
- uniform: 内存带宽
- normal: 超越函数计算（log, cos, sin）

### 8.3 使用场景差异

| 场景 | 推荐 API | 原因 |
|------|----------|------|
| **权重初始化** | `torch.randn` + 缩放 | Xavier/Kaiming 需要正态分布 |
| **Dropout 掩码** | `torch.rand` | 二值化阈值，均匀分布即可 |
| **数据增强噪声** | `torch.randn` | 高斯噪声模拟真实噪声 |
| **采样索引** | `torch.randint` | 离散整数，不用分布 |
| **强化学习探索** | `torch.rand` / `torch.randn` | 均匀探索 vs ε-greedy |
| **生成对抗网络** | `torch.randn` | 潜在空间通常假设正态分布 |

### 8.4 代码对比示例

```python
# 1. Xavier 初始化（需要正态分布）
# uniform 版本（不推荐）
limit = math.sqrt(6.0 / (fan_in + fan_out))
weight = torch.rand(fan_in, fan_out) * 2 * limit - limit  # 繁琐

# normal 版本（推荐）
std = math.sqrt(2.0 / (fan_in + fan_out))
weight = torch.randn(fan_in, fan_out) * std  # 简洁

# 2. Dropout 掩码（均匀分布更高效）
mask = torch.rand(x.shape) > dropout_rate  # 推荐
mask = torch.randn(x.shape) > threshold    # 不推荐

# 3. 高斯噪声注入
noise = torch.randn_like(x) * noise_std  # 推荐
```

## 9. torch.normal 的多种变体

`torch.normal` 支持灵活的参数广播机制：

### 9.1 标量 mean + 标量 std

```python
# 生成固定分布的张量
result = torch.normal(mean=10.0, std=2.0, size=(3, 4))
# 等价于
result = torch.randn(3, 4) * 2.0 + 10.0
```

### 9.2 张量 mean + 标量 std

**文件**: `aten/src/ATen/native/Distributions.cpp:284`

```python
means = torch.tensor([1.0, 2.0, 3.0])
result = torch.normal(means, std=0.5)
# 每个元素使用不同的均值，相同的标准差
```

**实现逻辑**:
```cpp
Tensor& normal_out(const Tensor& mean, double std, std::optional<Generator> gen, Tensor& output) {
  // 1. 生成 N(0, 1)
  normal_impl_<NormalStub, Generator>(output, 0, 1, gen);
  // 2. 缩放：output *= std
  output.mul_(std);
  // 3. 平移：output += mean（广播）
  output.add_(mean);
  return output;
}
```

### 9.3 标量 mean + 张量 std

```python
stds = torch.tensor([0.1, 0.2, 0.3])
result = torch.normal(mean=5.0, std=stds)
# 每个元素使用相同的均值，不同的标准差
```

**实现逻辑**:
```cpp
Tensor& normal_out(double mean, const Tensor& std, std::optional<Generator> gen, Tensor& output) {
  // 1. 生成 N(0, 1)
  normal_impl_<NormalStub, Generator>(output, 0, 1, gen);
  // 2. 逐元素乘法：output *= std（广播）
  output.mul_(std);
  // 3. 加常数：output += mean
  output.add_(mean);
  return output;
}
```

### 9.4 张量 mean + 张量 std

```python
means = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
stds = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
result = torch.normal(means, stds)
# 每个元素使用独立的均值和标准差
```

**实现逻辑**:
```cpp
Tensor& normal_out(const Tensor& mean, const Tensor& std, std::optional<Generator> gen, Tensor& output) {
  // 1. 推断输出形状（广播规则）
  auto shape = at::infer_size(mean.sizes(), std.sizes());
  at::native::resize_output(output, shape);

  // 2. 生成 N(0, 1)
  normal_impl_<NormalStub, Generator>(output, 0, 1, gen);

  // 3. 仿射变换：output = output * std + mean
  output.mul_(std).add_(mean);

  return output;
}
```

**广播规则**:
```python
mean = torch.tensor([1.0, 2.0])     # shape: (2,)
std = torch.tensor([[0.1], [0.2]])  # shape: (2, 1)
result = torch.normal(mean, std)    # shape: (2, 2) 广播后
# result[0, :] ~ N([1.0, 2.0], 0.1²)
# result[1, :] ~ N([1.0, 2.0], 0.2²)
```

## Appendix A: Box-Muller 变换的数学证明

### A.1 问题陈述

给定两个独立的均匀随机变量 $U_1, U_2 \sim \text{Uniform}(0, 1)$，证明以下变换产生两个独立的标准正态随机变量：

$$
\begin{cases}
Z_0 = \sqrt{-2 \ln U_1} \cdot \cos(2\pi U_2) \\
Z_1 = \sqrt{-2 \ln U_1} \cdot \sin(2\pi U_2)
\end{cases}
$$

且 $Z_0, Z_1 \sim N(0, 1)$ 独立。

### A.2 极坐标变换

设 $Z_0, Z_1$ 是两个独立的标准正态随机变量，其联合概率密度函数为：

$$
f_{Z_0, Z_1}(z_0, z_1) = \frac{1}{2\pi} e^{-\frac{z_0^2 + z_1^2}{2}}
$$

引入极坐标变换：
$$
\begin{cases}
R = \sqrt{Z_0^2 + Z_1^2} \\
\Theta = \arctan\left(\frac{Z_1}{Z_0}\right)
\end{cases}
$$

### A.3 雅可比行列式

雅可比矩阵：
$$
J = \begin{vmatrix}
\frac{\partial z_0}{\partial r} & \frac{\partial z_0}{\partial \theta} \\
\frac{\partial z_1}{\partial r} & \frac{\partial z_1}{\partial \theta}
\end{vmatrix}
= \begin{vmatrix}
\cos\theta & -r\sin\theta \\
\sin\theta & r\cos\theta
\end{vmatrix}
= r
$$

极坐标下的联合密度：
$$
f_{R, \Theta}(r, \theta) = f_{Z_0, Z_1}(r\cos\theta, r\sin\theta) \cdot |J|
= \frac{1}{2\pi} e^{-\frac{r^2}{2}} \cdot r
= \frac{r}{2\pi} e^{-\frac{r^2}{2}}
$$

### A.4 边缘分布

$$
f_R(r) = \int_0^{2\pi} f_{R, \Theta}(r, \theta) d\theta = r e^{-\frac{r^2}{2}}, \quad r \geq 0
$$

$$
f_{\Theta}(\theta) = \int_0^{\infty} f_{R, \Theta}(r, \theta) dr = \frac{1}{2\pi}, \quad 0 \leq \theta < 2\pi
$$

可见 $\Theta \sim \text{Uniform}(0, 2\pi)$，且 $R$ 和 $\Theta$ 独立。

### A.5 逆变换采样

**对于 $\Theta$**:

$\Theta \sim \text{Uniform}(0, 2\pi)$，故：
$$
\Theta = 2\pi U_2, \quad U_2 \sim \text{Uniform}(0, 1)
$$

**对于 $R$**:

$R$ 的累积分布函数：
$$
F_R(r) = \int_0^r x e^{-\frac{x^2}{2}} dx = 1 - e^{-\frac{r^2}{2}}
$$

逆变换采样：
$$
U_1 = F_R(R) = 1 - e^{-\frac{R^2}{2}}
$$

解出 $R$：
$$
R = \sqrt{-2 \ln(1 - U_1)}
$$

因为 $1 - U_1 \sim \text{Uniform}(0, 1)$，可以简化为：
$$
R = \sqrt{-2 \ln U_1}
$$

### A.6 最终变换

结合极坐标关系：
$$
\begin{cases}
Z_0 = R \cos\Theta = \sqrt{-2 \ln U_1} \cdot \cos(2\pi U_2) \\
Z_1 = R \sin\Theta = \sqrt{-2 \ln U_1} \cdot \sin(2\pi U_2)
\end{cases}
$$

**证毕**：$Z_0, Z_1$ 独立且都服从 $N(0, 1)$。

### A.7 为什么这个变换有效？

**关键洞察**:
1. 两个独立标准正态随机变量的联合分布在笛卡尔坐标下是圆对称的
2. 在极坐标下，$R$ 和 $\Theta$ 是独立的
3. $\Theta$ 是均匀分布，可以直接从 $U_2$ 得到
4. $R^2$ 服从指数分布，可以通过逆变换从 $U_1$ 得到

**直观理解**:
```
笛卡尔坐标: (Z₀, Z₁) ~ 圆对称的高斯云
极坐标:     (R, Θ) ~ R 决定距离，Θ 均匀分布方向
逆变换:     (U₁, U₂) ~ R 从指数分布采样，Θ 从均匀分布采样
```

## Appendix B: 正态分布的统计特性

### B.1 基本性质

**标准正态分布** $Z \sim N(0, 1)$:
- **PDF**: $f(z) = \frac{1}{\sqrt{2\pi}} e^{-\frac{z^2}{2}}$
- **CDF**: $\Phi(z) = \int_{-\infty}^z \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}} dt$（无解析解）
- **期望**: $E[Z] = 0$
- **方差**: $Var(Z) = 1$
- **偏度**: $Skew(Z) = 0$（对称）
- **峰度**: $Kurt(Z) = 3$（超额峰度 = 0）

**一般正态分布** $X \sim N(\mu, \sigma^2)$:
- **PDF**: $f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$
- **标准化**: $Z = \frac{X - \mu}{\sigma} \sim N(0, 1)$
- **期望**: $E[X] = \mu$
- **方差**: $Var(X) = \sigma^2$

### B.2 线性变换性质

若 $Z \sim N(0, 1)$，则：
$$
X = \sigma Z + \mu \sim N(\mu, \sigma^2)
$$

**证明**:
$$
\begin{align*}
E[X] &= E[\sigma Z + \mu] = \sigma \cdot 0 + \mu = \mu \\
Var(X) &= Var(\sigma Z + \mu) = \sigma^2 Var(Z) = \sigma^2
\end{align*}
$$

这正是 PyTorch `normal_` 的实现原理。

### B.3 和的性质

若 $X_1 \sim N(\mu_1, \sigma_1^2)$，$X_2 \sim N(\mu_2, \sigma_2^2)$ 独立，则：
$$
X_1 + X_2 \sim N(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)
$$

**应用**: 深度学习中的 Dropout 噪声累积。

### B.4 复数正态分布

若 $Z = X + iY$ 是复正态分布 $CN(\mu, \sigma^2)$，则：
- $X \sim N(\text{Re}(\mu), \sigma^2/2)$
- $Y \sim N(\text{Im}(\mu), \sigma^2/2)$
- $X$ 和 $Y$ 独立

**为什么方差是 $\sigma^2/2$？**
$$
Var(Z) = Var(X) + Var(Y) = \frac{\sigma^2}{2} + \frac{\sigma^2}{2} = \sigma^2
$$

这正是 `normal_impl_` 中复数张量处理的依据：
```cpp
normal_kernel(float_tensor, mean, std/sqrt(2), gen);
```

### B.5 Box-Muller 的统计保证

Box-Muller 变换生成的 $Z_0, Z_1$：
1. **独立性**: $Cov(Z_0, Z_1) = 0$
2. **同分布**: $Z_0, Z_1 \sim N(0, 1)$
3. **精确性**: 理论上完全精确（不是近似）

**与拒绝采样的对比**:
- Box-Muller: 确定性变换，无拒绝
- Marsaglia: 拒绝采样，平均拒绝率 21%

### B.6 深度学习中的应用

| 应用 | 分布参数 | 原因 |
|------|----------|------|
| **Xavier 初始化** | $N(0, \frac{2}{n_{in}+n_{out}})$ | 保持梯度方差 |
| **Kaiming 初始化** | $N(0, \frac{2}{n_{in}})$ | ReLU 激活后的方差 |
| **批归一化** | $N(0, 1)$ 初始化 $\gamma$ | 恒等变换起点 |
| **Dropout 噪声** | $N(0, \frac{p}{1-p})$ | 期望不变性 |
| **VAE 潜在空间** | $N(0, I)$ 先验 | 标准正态假设 |
| **数据增强** | $N(0, \epsilon^2)$ | 小幅扰动 |

---

**文档版本**: v1.0
**对应 PyTorch 版本**: >= 2.0
**最后更新**: 2024-12-22
