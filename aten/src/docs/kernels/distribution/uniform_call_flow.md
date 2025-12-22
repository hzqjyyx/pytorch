# torch.rand/randn 随机数生成调用流程详解

## 概述

本文档详细介绍了 PyTorch 随机数生成 API（以 `torch.rand` 和 `torch.randn` 为例）从 Python 用户 API 到底层随机数引擎的完整调用流程。PyTorch 的随机数生成系统是深度学习训练中至关重要的组件，它提供了高性能、可复现的随机数生成能力。

**本文重点关注 CUDA 实现**，因为 GPU 加速的随机数生成在深度学习中使用最为广泛，其实现也更加复杂和精巧。

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
硬件执行层 (CPU: MT19937 / GPU: Philox)
```

## 1. Python 层入口

### 1.1 用户API

```python
# 生成均匀分布的随机数 [0, 1)
result = torch.rand(3, 4)

# 生成标准正态分布的随机数
result = torch.randn(3, 4)

# 生成指定范围的随机整数
result = torch.randint(0, 10, (3, 4))

# 使用自定义生成器
generator = torch.Generator()
generator.manual_seed(42)
result = torch.rand(3, 4, generator=generator)

# CUDA 张量
result = torch.rand(3, 4, device='cuda')
```

### 1.2 API绑定

这些函数是对 C++ 扩展的直接绑定：
- `torch.rand` → `torch._C.rand`
- `torch.randn` → `torch._C.randn`
- `torch.randint` → `torch._C.randint`

**文件位置**: `torch/_torch_docs.py`

## 2. C++ 分派层

### 2.1 torch.rand 入口函数

**文件**: `aten/src/ATen/native/TensorFactories.cpp:1047`

```cpp
Tensor rand(
    IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::rand(
      size,
      static_cast<std::optional<Generator>>(std::nullopt),
      dtype,
      layout,
      device,
      pin_memory);
}
```

### 2.2 带生成器的版本

**文件**: `aten/src/ATen/native/TensorFactories.cpp:1062`

```cpp
Tensor rand(
    IntArrayRef size,
    std::optional<Generator> generator,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 构建TensorOptions
  TensorOptions options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(
          pin_memory);

  // 创建空张量
  auto result = at::empty(size, options);

  // 关键调用：使用 uniform_ 填充 [0, 1) 的均匀分布
  return result.uniform_(0, 1, std::move(generator));
}
```

**主要功能**:
- 构建 `TensorOptions` 对象，封装数据类型、布局、设备等信息
- 创建指定形状的空张量（未初始化）
- 调用 `uniform_` 原地操作填充 [0, 1) 区间的均匀分布随机数

### 2.3 输出参数版本

**文件**: `aten/src/ATen/native/TensorFactories.cpp:1082`

```cpp
Tensor& rand_out(
    IntArrayRef size,
    std::optional<Generator> generator,
    Tensor& result) {
  result.resize_(size);
  return result.uniform_(0, 1, std::move(generator));
}
```

**优化点**: 不创建新张量，直接使用用户提供的输出张量，避免内存分配

## 3. 核心分布层

### 3.1 uniform_ 实现

**文件**: `aten/src/ATen/native/Distributions.cpp:250`

```cpp
Tensor& uniform_(Tensor& self, double from, double to, std::optional<Generator> gen) {
  return at::native::templates::uniform_impl_<UniformStub, Generator>(self, from, to, std::move(gen));
}
```

这里使用了模板化设计：
- `UniformStub`: 分发器存根，用于分派到 CPU/CUDA 实现
- `Generator`: 随机数生成器类型

### 3.2 uniform_impl_ 模板函数

**文件**: `aten/src/ATen/native/DistributionTemplates.h:286`

```cpp
template<template<typename> class uniform_kernel, typename RNG>
at::Tensor& uniform_impl_(at::Tensor& self, double from, double to, std::optional<Generator> generator) {
  // 处理复数张量（将其视为实数的view）
  if (self.is_complex()) {
    CHECK_EMPTY_AND_RETURN(self);
    auto float_tensor = at::view_as_real(self);
    uniform_impl_<uniform_kernel, RNG>(float_tensor, from, to, generator);
  } else {
    // 浮点类型的边界检查
    AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                     self.scalar_type(), "check_uniform_bounds", [&] {
      const auto dtype = self.dtype();
      const auto min = static_cast<double>(std::numeric_limits<scalar_t>::lowest());
      const auto max = static_cast<double>(std::numeric_limits<scalar_t>::max());

      // 检查边界
      CHECK_OUT_OF_BOUNDS(from, "from", min, max, dtype);
      CHECK_OUT_OF_BOUNDS(to, "to", min, max, dtype);
      TORCH_CHECK(from <= to, "uniform_ expects to return a [from, to) range");

      // 调整边界到有效范围
      from = std::min(std::max(from, min), max);
      to = std::max(std::min(to, max), min);
    });

    CHECK_EMPTY_AND_RETURN(self);

    // 创建 TensorIterator（用于高效遍历）
    auto iter = at::TensorIterator::borrowing_nullary_op(self);

    // 分派到具体设备的 uniform_kernel
    uniform_kernel<RNG>()(iter, from, to, generator);
  }
  return self;
}
```

**关键功能**:
- **复数支持**: 复数张量被视为实数张量的 view，实部和虚部独立生成
- **边界检查**: 确保 from/to 在数据类型的有效范围内
- **TensorIterator**: 创建高效的张量遍历器，处理各种内存布局
- **设备分派**: 根据张量设备类型分派到 CPU 或 CUDA 实现

## 4. CPU 实现（简介）

CPU 实现相对简单，使用 MT19937 引擎和多线程并行。

**文件**: `aten/src/ATen/native/cpu/DistributionKernels.cpp:211`

```cpp
void uniform_kernel(TensorIteratorBase& iter, double from, double to, std::optional<Generator> gen) {
  CPUGeneratorImpl* generator = get_generator_or_default<CPUGeneratorImpl>(gen, detail::getDefaultCPUGenerator());
  templates::cpu::uniform_kernel(iter, from, to, generator);
}
```

**实现细节**:
- 使用互斥锁保护生成器状态
- `uniform_real_distribution` 将 RNG 输出转换为均匀分布
- `cpu_serial_kernel` 自动根据张量大小选择串行或并行执行

## 5. CUDA 实现（重点）

### 5.1 CUDA uniform_kernel 入口

**文件**: `aten/src/ATen/native/cuda/DistributionUniform.cu:8`

```cpp
void uniform_kernel(TensorIteratorBase& iter, double from, double to, std::optional<Generator> gen) {
  // 获取或创建 CUDA 生成器
  auto generator = get_generator_or_default<CUDAGeneratorImpl>(gen, cuda::detail::getDefaultCUDAGenerator());

  // 调用模板化的 CUDA 实现
  templates::cuda::uniform_kernel(iter, from, to, generator);
}
```

**关键点**：
- **Generator 的第一次使用**: 在这里将 `std::optional<Generator>` 转换为具体的 `CUDAGeneratorImpl*`
- 如果用户没有提供 generator，使用默认的 CUDA 生成器

### 5.2 CUDA 模板实现

**文件**: `aten/src/ATen/native/cuda/DistributionTemplates.h:484`

```cpp
template<typename RNG>
void uniform_kernel(TensorIteratorBase& iter, double from_, double to_, RNG gen) {
  AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16,
                                  iter.dtype(), "uniform_kernel_cuda", [&] {
    auto from = static_cast<scalar_t>(from_);
    auto to = static_cast<scalar_t>(to_);
    using opmath_t = at::opmath_type<scalar_t>;
    auto range = static_cast<opmath_t>(to-from);

    // 定义变换 lambda（在设备端执行）
    auto uniform_func = [range, from, to] __device__ (opmath_t rand) {
      // 计算输出值: rand ∈ (0,1] → value ∈ [from, to)
      auto value = static_cast<scalar_t>(rand * range + from);

      // 将 curand 的 (0, 1] 反转为 [0, 1)
      // cuRAND 生成 (0, 1]，但 PyTorch 语义要求 [0, 1)
      auto reverse_bound_value = value == to ? from : value;
      return reverse_bound_value;
    };

    // 调用通用的 uniform_and_transform
    // 这是 CUDA 实现的核心入口
    uniform_and_transform<scalar_t, opmath_t>(iter, gen, uniform_func);
   });
}
```

**CUDA 特殊处理**:
- **边界反转**: cuRAND 生成 (0, 1]，需要转换为 [0, 1)
- **混合精度计算**: 使用 `opmath_t` 进行中间计算，提高精度
  - Half/BFloat16 → float 计算
  - Float → float 计算
  - Double → double 计算
- **设备端 lambda**: `__device__` 变换函数在 GPU 上执行

### 5.3 uniform_and_transform 实现

**文件**: `aten/src/ATen/native/cuda/DistributionTemplates.h:428`

这是 CUDA 随机数生成的核心函数，将 cuRAND 的原始输出转换为目标分布。

```cpp
template<typename scalar_t, typename accscalar_t, typename RNG, typename transform_t>
void uniform_and_transform(TensorIteratorBase& iter, RNG gen, transform_t transform) {
  if (std::is_same_v<scalar_t, double>) {
    // double 类型：使用 curand_uniform2_double，一次生成 2 个 double
    distribution_nullary_kernel<scalar_t, accscalar_t, double2>(iter,
      gen,
      [] __device__ (curandStatePhilox4_32_10_t* state) -> double2 {
        return curand_uniform2_double(state);
      },
      transform);
  } else {
    // float/half/bfloat16 类型：使用 curand_uniform4，一次生成 4 个 float
    distribution_nullary_kernel<scalar_t, accscalar_t, float4>(iter,
      gen,
      [] __device__ (curandStatePhilox4_32_10_t* state) -> float4 {
        return curand_uniform4(state);
      },
      transform);
  }
}
```

**关键设计**:
1. **类型特化**: double 和 float 使用不同的 cuRAND 函数
2. **向量化生成**:
   - `curand_uniform4()`: 一次生成 4 个 float 随机数
   - `curand_uniform2_double()`: 一次生成 2 个 double 随机数
3. **分离关注点**:
   - cuRAND 负责生成 [0,1] 的均匀分布
   - `transform` lambda 负责转换为目标分布

### 5.4 distribution_nullary_kernel（CUDA 核心）

**文件**: `aten/src/ATen/native/cuda/DistributionTemplates.h:112`

这是 CUDA 随机数生成的最核心函数，负责：
1. 计算 Grid/Block 配置
2. 从 Generator 获取 Philox 状态（**Generator 的第二次使用**）
3. 启动 CUDA kernel

```cpp
template<typename scalar_t,
         typename accscalar_t,
         typename dist_func_return_t,
         typename RNG,
         typename dist_t,
         typename transform_t>
void distribution_nullary_kernel(at::TensorIteratorBase& iter,
                                 RNG gen,
                                 const dist_t& dist_func,
                                 const transform_t transform_func) {
  const int unroll_factor = sizeof(dist_func_return_t) / sizeof(accscalar_t);
  TORCH_CHECK(unroll_factor >= 1, "unroll_factor must be >= 1.");
  int64_t numel = iter.numel();
  if (numel == 0) {
    return;
  }

  // 计算执行策略：grid size, block size, offset
  auto [counter_offset, grid, block] = calc_execution_policy(numel, unroll_factor);

  // **Generator 的关键使用**：获取 Philox CUDA 状态
  PhiloxCudaState rng_engine_inputs;
  {
    // 线程安全：获取生成器锁
    std::lock_guard<std::mutex> lock(gen->mutex_);

    // 从 Generator 获取 Philox 状态并增加偏移量
    rng_engine_inputs = gen->philox_cuda_state(counter_offset);
  }

  // 处理大张量：递归分块处理
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      distribution_nullary_kernel<scalar_t, accscalar_t, dist_func_return_t>(sub_iter,
        gen, dist_func, transform_func);
    }
    return;
  }

  char* out_data = (char*)iter.data_ptr(0);
  auto stream = at::cuda::getCurrentCUDAStream();

  // 启动 CUDA kernel
  if (iter.is_trivial_1d()) {
    // 优化路径：连续内存
    auto strides = iter.get_inner_strides();
    int stride0 = strides[0];
    distribution_elementwise_grid_stride_kernel<accscalar_t, unroll_factor><<<grid, block, 0, stream>>>(
      numel,
      rng_engine_inputs,  // Philox 状态传递给 kernel
      dist_func,
      [=]__device__(int idx, accscalar_t rand) {
        scalar_t* out = (scalar_t*)&out_data[stride0 * idx];
        *out = transform_func(rand);
      }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  } else {
    // 通用路径：非连续内存
    auto offset_calc = make_offset_calculator<1>(iter);
    distribution_elementwise_grid_stride_kernel<accscalar_t, unroll_factor><<<grid, block, 0, stream>>>(
      numel,
      rng_engine_inputs,
      dist_func,
      [=]__device__(int idx, accscalar_t rand) {
        auto offsets = offset_calc.get(idx);
        scalar_t* out = (scalar_t*)&out_data[offsets[0]];
        *out = transform_func(rand);
      }
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}
```

**Generator 使用流程总结**：
1. **第一次使用**: `get_generator_or_default()` 获取生成器实例
2. **第二次使用**: `gen->philox_cuda_state(counter_offset)` 获取 Philox 状态
3. **状态传递**: `PhiloxCudaState` 传递给 CUDA kernel
4. **线程内使用**: 每个 CUDA 线程用状态初始化 `curandStatePhilox4_32_10_t`

### 5.5 CUDA Grid-Stride Loop 内核

**文件**: `aten/src/ATen/native/cuda/DistributionTemplates.h:66`

```cpp
C10_LAUNCH_BOUNDS_2(block_size_bound, grid_size_bound)
__global__ void distribution_elementwise_grid_stride_kernel(
    int64_t numel,
    PhiloxCudaState philox_args,
    const dist_t dist_func,
    const transform_t transform_func) {

  // 1. 解包 Philox 状态（seed + offset）
  auto [seed, offset] = at::cuda::philox::unpack(philox_args);
  int64_t idx = ((int64_t) blockIdx.x) * blockDim.x + threadIdx.x;

  // 2. 初始化每个线程的 Philox 状态
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);

  // 3. Grid-stride loop（提高GPU利用率）
  int64_t rounded_size = ((numel - 1)/(blockDim.x * gridDim.x * unroll_factor)+1) *
      blockDim.x * gridDim.x * unroll_factor;

  for(int64_t linear_index = idx; linear_index < rounded_size;
      linear_index += blockDim.x * gridDim.x * unroll_factor) {

    // 4. 一次生成多个随机数（例如 float4）
    auto rand = dist_func(&state);

    // 5. 展开循环处理每个随机数
    #pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int64_t li = linear_index + blockDim.x * gridDim.x * ii;
      if (li < numel) {
        // 6. 应用变换并写入输出
        transform_func(li, static_cast<accscalar_t>((&rand.x)[ii]));
      }
    }
    __syncthreads();
  }
}
```

### 5.6 Grid-Stride Loop 的优势

Grid-Stride Loop 是 CUDA 随机数生成的核心优化策略，相比传统的一线程一元素模式有显著优势：

#### 优势 1: 提高 GPU 占用率（Occupancy）

```
传统模式：
  线程数 = 元素数
  问题：元素数可能远小于 GPU 核心数，导致 GPU 利用率低

Grid-Stride Loop：
  线程数 = min(optimal_threads, 元素数)
  每个线程处理多个元素
  优势：即使元素数很少，也能充分利用 GPU
```

**示例**:
```cpp
// 假设有 1000 个元素，GPU 有 2048 个核心
// 传统模式：只启动 1000 个线程，1048 个核心闲置
// Grid-Stride Loop：启动 2048 个线程，每个线程处理 1-2 个元素
```

#### 优势 2: 减少 Kernel 启动开销

```cpp
// 对于大张量，避免多次 kernel 启动
// 所有工作在一个 kernel 中完成
// Grid-stride loop 自动处理任意大小的输入
```

#### 优势 3: 更好的缓存利用

```cpp
// 每个线程连续处理多个元素
// 提高数据局部性和缓存命中率
for(int64_t linear_index = idx;
    linear_index < rounded_size;
    linear_index += blockDim.x * gridDim.x * unroll_factor) {
  // 连续访问内存
}
```

#### 优势 4: 灵活的负载均衡

```cpp
// 自动适应不同的张量大小
// 大张量：每个线程处理更多元素
// 小张量：每个线程处理更少元素
// 无需手动调优 grid/block 配置
```

#### 优势 5: 向量化内存访问

```cpp
// 一次生成 float4（4 个随机数）
auto rand = curand_uniform4(&state);  // 128 位向量化生成

#pragma unroll
for (int ii = 0; ii < 4; ii++) {
  // 展开循环，编译器优化为向量化内存访问
  output[idx + ii] = (&rand.x)[ii];
}
```

### 5.7 执行策略计算

**文件**: `aten/src/ATen/native/cuda/DistributionTemplates.h:49`

```cpp
std::tuple<uint64_t, dim3, dim3> calc_execution_policy(const int64_t total_elements, const uint32_t unroll_factor) {
  const uint64_t numel = static_cast<uint64_t>(total_elements);
  const uint32_t block_size = 256;  // 每个 block 256 个线程

  dim3 dim_block(block_size);
  dim3 grid((numel + block_size - 1) / block_size);

  // 限制 grid 大小，避免过度启动
  uint32_t blocks_per_sm = at::cuda::getCurrentDeviceProperties()->maxThreadsPerMultiProcessor / block_size;
  grid.x = std::min(
      static_cast<uint32_t>(at::cuda::getCurrentDeviceProperties()->multiProcessorCount) * blocks_per_sm,
      grid.x);

  // 计算 Philox counter offset
  // 每个线程最多需要 counter_offset 个随机数
  uint64_t counter_offset = ((numel - 1) / (block_size * grid.x * unroll_factor) + 1) * max_generator_offsets_per_curand_call;

  return std::make_tuple(counter_offset, grid, dim_block);
}
```

**计算逻辑**:
- **Block size**: 固定 256 线程（经验值，平衡占用率和资源使用）
- **Grid size**: 根据 SM 数量和元素数量计算
- **Counter offset**: 确保每个线程有足够的随机数种子空间

## 6. 随机数生成器引擎层

### 6.1 CPU 生成器 (MT19937)

**文件**: `aten/src/ATen/CPUGeneratorImpl.h:10`

```cpp
struct TORCH_API CPUGeneratorImpl : public c10::GeneratorImpl {
  CPUGeneratorImpl(uint64_t seed_in = default_rng_seed_val);

  uint32_t random();     // 生成 32 位随机数
  uint64_t random64();   // 生成 64 位随机数

  at::mt19937 engine();  // 底层 MT19937 引擎

private:
  at::mt19937 engine_;   // Mersenne Twister 19937 引擎
  std::optional<float> next_float_normal_sample_;   // 缓存的正态分布样本
  std::optional<double> next_double_normal_sample_; // Box-Muller 优化
};
```

**MT19937 引擎**:
- **算法**: Mersenne Twister，周期为 2^19937-1
- **状态大小**: 624 个 32 位整数
- **性能**: CPU 上高质量、高性能的 PRNG
- **可复现性**: 相同的种子产生完全相同的序列

### 6.2 CUDA 生成器 (Philox) - 重点

**文件**: `aten/src/ATen/cuda/CUDAGeneratorImpl.h:125`

```cpp
struct TORCH_CUDA_CPP_API CUDAGeneratorImpl : public c10::GeneratorImpl {
  CUDAGeneratorImpl(DeviceIndex device_index = -1);

  // 生成 Philox CUDA 状态（核心方法）
  PhiloxCudaState philox_cuda_state(uint64_t increment);

  void set_philox_offset_per_thread(uint64_t offset);
  uint64_t philox_offset_per_thread() const;

private:
  c10::intrusive_ptr<CUDAGeneratorState> state_;
};
```

**Philox 状态**:
```cpp
struct CUDAGeneratorState {
  uint64_t seed_;                      // 种子（Philox 密钥）
  uint64_t philox_offset_per_thread_;  // 每线程偏移量
  uint32_t offset_intragraph_;         // 图内偏移量（用于 CUDA Graph）
  bool capturing_{};                   // 是否正在捕获 CUDA Graph
};
```

**philox_cuda_state 实现**:
```cpp
PhiloxCudaState CUDAGeneratorImpl::philox_cuda_state(uint64_t increment) {
  // 获取当前偏移量并增加
  uint64_t offset = state_->philox_offset_per_thread_;
  state_->philox_offset_per_thread_ += increment;

  // 返回 Philox 状态（seed + offset）
  return PhiloxCudaState(state_->seed_, offset);
}
```

## 7. 完整调用流程图（CUDA）

```
torch.rand(3, 4, device='cuda')
    ↓
torch._C.rand                                    # Python C扩展绑定
    ↓
at::rand(size, device='cuda', ...)               # C++ API
    ↓
at::empty(size, device='cuda')                   # 创建未初始化 CUDA 张量
    ↓
result.uniform_(0, 1, generator)                 # 原地填充
    ↓
at::native::templates::uniform_impl_()           # 模板分派
    ↓
TensorIterator::borrowing_nullary_op()           # 创建遍历器
    ↓
uniform_kernel<CUDAGeneratorImpl>()              # CUDA 分派
    ↓
【Generator 第一次使用】
get_generator_or_default<CUDAGeneratorImpl>()    # 获取 CUDA 生成器
    ↓
templates::cuda::uniform_kernel()                # CUDA 模板实现
    ↓
    ├─ 定义 uniform_func: [0,1] → [from, to)
    └─ uniform_and_transform()
        ↓
        ├─ 选择 curand_uniform4() 或 curand_uniform2_double()
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
            ├─ curand_uniform4(&state)          # 生成 4 个随机数
            │       ↓
            │   Philox 算法执行
            │       ↓
            │   返回 float4{x, y, z, w}
            │
            └─ 应用 transform_func
                    ↓
                value = rand * range + from
                output[idx] = (value == to) ? from : value
```

## 9. 常见分布实现对比

| 分布 | CPU 算法 | CUDA 算法 | cuRAND 函数 | 特点 |
|------|---------|----------|-------------|------|
| **uniform** | MT19937 + 线性映射 | Philox + 线性映射 | `curand_uniform4` | 最简单，最快，向量化 |
| **normal** | Box-Muller | Box-Muller | `curand_normal4` | cuRAND 内置，高效 |
| **exponential** | 逆变换采样 | 逆变换采样 | `curand_uniform4` + transform | `-log(1-U)` |
| **cauchy** | 逆变换采样 | 逆变换采样 | `curand_uniform4` + transform | `tan(π(U-0.5))` |
| **log_normal** | `exp(normal)` | `exp(normal)` | `curand_normal4` + exp | 复合分布 |
| **bernoulli** | 比较采样 | 比较采样 | `curand_uniform4` + 比较 | `U < p` |
| **geometric** | 逆变换采样 | 逆变换采样 | `curand_uniform4` + transform | `⌈log(U)/log(1-p)⌉` |

## Appendix A: Philox 算法原理

### A.1 为什么 CUDA 不用 MT19937？

**MT19937 的问题**:
1. **巨大的状态**: 624 个 32 位整数 = 2496 字节/线程
   - 1024 个线程 = 2.4 MB 状态
   - 消耗宝贵的寄存器/共享内存
2. **串行依赖**: 生成下一个随机数依赖前一个状态
   - GPU 并行性无法利用
3. **难以跳跃**: 无法高效跳到序列的任意位置
   - 每个线程需要独立初始化

**Philox 的优势**:
1. **极小状态**: 只需 128 位 counter + 128 位 key
2. **无状态计算**: `random(counter, key)` 是纯函数
3. **完美并行**: 每个线程独立，无同步需求
4. **快速跳跃**: 改变 counter 即可跳到任意位置

### A.2 Philox 算法原理

Philox 是一种 **Counter-Based Random Number Generator (CBRNG)**，基于密码学的思想。

#### 核心思想

Philox4x32 意味着它一次处理 4 个 32 位整数（共 128 位）。每次生成时，它需要两个核心输入：
- Counter (明文): $C = [L_0, R_0, L_1, R_1]$
  - 这是 4 个 32 位无符号整数。
  - 在 PyTorch CUDA 中，这就是你的线程索引（Thread ID）加上当前的全局偏移量（Offset）。
- Key (密钥): $K = [k_0, k_1]$
  - 这是 2 个 32 位无符号整数（共 64 位）。
  - 这就是你用 torch.manual_seed() 设置的种子。

Philox4x32-10 会运行 10 轮 相同的逻辑。每一轮都在疯狂地搅拌这些比特。 每一轮包含三个步骤：乘法 (S-Box)、异或 (XOR)、置换 (Permutation)。
```
random = Philox_Round( ... Philox_Round(counter, key) ...)
                        \_____ 10 rounds _____/
```

**关键特性**:
- **确定性**: 相同的 (counter, key) 总是产生相同的输出
- **雪崩效应**: counter 的微小变化导致输出完全不同
- **统计质量**: 通过 BigCrush 测试（最严格的随机性测试）

#### Philox4x32 详细步骤

Philox4x32 = 4 个 32-bit 输出，基于 32-bit 运算

```cpp
struct Philox4x32State {
  uint32_t counter[4];  // 128-bit counter
  uint32_t key[2];      // 64-bit key (可扩展到 128-bit)
};

// 单轮 Philox 变换
__device__ void philox_single_round(uint32_t counter[4], uint32_t key[2]) {
  // S-box 层：非线性变换
  uint64_t prod0 = (uint64_t)0xD2511F53 * counter[0];
  uint64_t prod1 = (uint64_t)0xCD9E8D57 * counter[2];

  // 提取高低 32 位
  uint32_t hi0 = prod0 >> 32;
  uint32_t lo0 = prod0 & 0xFFFFFFFF;
  uint32_t hi1 = prod1 >> 32;
  uint32_t lo1 = prod1 & 0xFFFFFFFF;

  // P-box 层：排列变换
  // 注意：这里发生了交叉混合（Lane Mixing），第 0 组的结果混入了第 1 组，第 1 组的结果混入了第 0 组。
  counter[0] = hi1 ^ counter[1] ^ key[0];
  counter[1] = lo1;
  counter[2] = hi0 ^ counter[3] ^ key[1];
  counter[3] = lo0;
}

// 密钥更新
// 这种简单的加法能保证密钥序列在 $2^{64}$ 轮内不会重复，且分布均匀。
__device__ void bump_key(uint32_t key[2]) {
  key[0] += 0x9E3779B9;  // 黄金比例常数
  key[1] += 0xBB67AE85;  // sqrt(3)相关的常数
}

// 完整 Philox 生成（10 轮）
// uint4:4 个 32 位无符号整数
// uint2:2 个 32 位无符号整数
__device__ uint4 philox4x32_10(uint4 counter, uint2 key) {
  uint32_t cnt[4] = {counter.x, counter.y, counter.z, counter.w};
  uint32_t k[2] = {key.x, key.y};

  // 10 轮混淆
  for (int i = 0; i < 10; i++) {
    philox_single_round(cnt, k);
    if (i < 9) bump_key(k);  // 最后一轮不更新密钥
  }

  return make_uint4(cnt[0], cnt[1], cnt[2], cnt[3]);
}
```

#### PyTorch 中的使用

```cpp
// 在 CUDA kernel 中
__global__ void random_kernel(PhiloxCudaState philox_args) {
  auto [seed, offset] = at::cuda::philox::unpack(philox_args);

  // 概念上，每个线程有独立的 counter 和 key，实际上由 curand_init 自动管理
  // uint64_t counter = offset + threadIdx.x;  // 线程独立
  // uint64_t key = seed;                      // 全局种子

  curandStatePhilox4_32_10_t state;
  curand_init(seed, threadIdx.x, offset, &state);

  // 生成随机数
  float4 rand = curand_uniform4(&state);
  // curand_uniform4 内部调用 philox4x32_10
  // 并将 uint4 转换为 float4 ∈ (0, 1]

  // 在这个 kernel 之前或之后需要对全局 offset 增加 4（atomic）
}
```
