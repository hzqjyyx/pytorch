# torch.randint/random 离散整数随机数生成详解

> **面向**: PyTorch 开发者和高级用户
> **目的**: 深入理解 PyTorch 离散整数随机数生成的完整实现，重点关注 CUDA 实现

---

## 文档导航

本文档是 PyTorch 随机数生成系统文档的一部分：
- **[架构总览](distribution_architecture.md)** - 随机数系统整体架构
- **[torch.rand/uniform 详解](random_call_flow.md)** - 均匀分布随机数生成
- **[torch.randn/normal 详解](normal_call_flow.md)** - 正态分布随机数生成
- **本文档** - 离散整数随机数生成

---

## 概述

PyTorch 提供了三个主要的离散整数随机数生成 API：

| API | 说明 | 输出范围 | 典型用途 |
|-----|------|----------|----------|
| `torch.randint(low, high, size)` | 生成指定范围的随机整数张量 | [low, high) | 随机索引、类别标签 |
| `torch.randperm(n)` | 生成 0 到 n-1 的随机排列 | [0, n) 的排列 | 数据打乱、批次划分 |
| `tensor.random_(from, to)` | 原地填充随机整数 | [from, to) | randint 的底层实现 |

**核心设计**：
- `torch.randint` 内部调用 `empty() + random_(low, high)`
- `torch.randperm` 使用**基于排序的算法**而非 Fisher-Yates shuffle
- CUDA 实现使用**向量化**的 Philox 生成器（一次生成 4 个随机数）
- 整数生成使用**简单 modulo 方法**（在特定条件下会有轻微偏差）

**与连续分布的区别**：
| 特性 | 连续分布 (uniform/normal) | 离散整数分布 |
|------|---------------------------|--------------|
| 算法 | 直接变换（linear/Box-Muller） | modulo + 偏移 |
| 偏差 | 无偏 | 有轻微 modulo bias（可控） |
| 输出范围 | 浮点数 [from, to) | 整数 [low, high) |

---

## 1. Python 层入口

### 1.1 torch.randint API

`torch.randint` 是生成整数随机数的工厂函数，有两种重载形式：

```python
# 形式 1: 指定 [low, high) 范围
torch.randint(low, high, size, *, generator=None, dtype=torch.int64,
              device=None, layout=torch.strided, requires_grad=False)

# 形式 2: 默认从 0 开始，指定 [0, high) 范围
torch.randint(high, size, *, generator=None, dtype=torch.int64,
              device=None, layout=torch.strided, requires_grad=False)
```

**使用示例**：
```python
# 生成 [0, 10) 的 3x4 随机整数张量
x = torch.randint(10, (3, 4))
# 输出示例: tensor([[3, 7, 2, 9],
#                    [1, 5, 8, 0],
#                    [6, 4, 2, 7]])

# 生成 [5, 15) 的随机整数
y = torch.randint(5, 15, (3, 4))
# 输出示例: tensor([[12, 8, 14, 6],
#                    [9, 11, 7, 13],
#                    [10, 5, 8, 12]])

# 使用自定义生成器
gen = torch.Generator().manual_seed(42)
z = torch.randint(100, (3, 4), generator=gen)

# CUDA 张量
cuda_tensor = torch.randint(10, (1000, 1000), device='cuda')
```

### 1.2 torch.randperm API

`torch.randperm` 生成 0 到 n-1 的随机排列，常用于数据打乱。

```python
torch.randperm(n, *, generator=None, dtype=torch.int64,
               device=None, layout=torch.strided, requires_grad=False)
```

**使用示例**：
```python
# 生成 [0, 10) 的随机排列
perm = torch.randperm(10)
# 输出示例: tensor([3, 7, 1, 9, 0, 4, 6, 2, 8, 5])

# 打乱数据集
data = torch.arange(100)
shuffled_data = data[torch.randperm(100)]

# 随机批次划分
batch_size = 32
indices = torch.randperm(len(dataset))
batch_indices = indices[:batch_size]
```

### 1.3 tensor.random_ API

`tensor.random_` 是原地操作，直接修改张量内容，有三种重载：

```python
# 形式 1: 指定 [from, to) 范围
tensor.random_(from, to, *, generator=None)

# 形式 2: 指定 [0, to) 范围
tensor.random_(to, *, generator=None)

# 形式 3: 使用数据类型的完整范围
tensor.random_(*, generator=None)
```

**使用示例**：
```python
# 复用张量避免内存分配
buffer = torch.empty(1000, 1000, dtype=torch.int64, device='cuda')
for epoch in range(100):
    buffer.random_(0, 100)  # 原地填充 [0, 100)
    # 使用 buffer...

# 使用数据类型的完整范围
x = torch.empty(10, dtype=torch.int32)
x.random_()  # 填充 int32 的完整范围 [-2^31, 2^31)
```

### 1.4 API 关系图

```
torch.randint(low, high, size)
    ↓
at::randint(low, high, size, generator, ...)
    ↓
result = at::empty(size, ...)        # 1. 创建未初始化张量
result.random_(low, high, generator) # 2. 原地填充随机整数
    ↓
return result
```

**关键点**：
- `torch.randint` 是**工厂函数**，创建新张量
- `tensor.random_` 是**原地操作**，`randint` 的底层实现
- `torch.randperm` 使用**独立的实现**（基于排序，非 Fisher-Yates）

---

## 2. C++ 分派层

### 2.1 randint 入口函数

文件位置: `aten/src/ATen/native/TensorFactories.cpp:1108-1165`

PyTorch 提供了 4 个 `randint` 重载，最终都归约到带完整参数的版本：

```cpp
// 形式 1: randint(high, size, ...)
Tensor randint(
    int64_t high,
    IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::randint(
      high,
      size,
      std::nullopt /* generator*/,  // 使用默认生成器
      dtype, layout, device, pin_memory);
}

// 形式 2: randint(high, size, generator, ...)
Tensor randint(
    int64_t high,
    IntArrayRef size,
    std::optional<Generator> generator,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 默认 low = 0
  return native::randint(
      0, high, size, std::move(generator), dtype, layout, device, pin_memory);
}

// 形式 3: randint(low, high, size, ...)
Tensor randint(
    int64_t low,
    int64_t high,
    IntArrayRef size,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  return native::randint(
      low, high, size, std::nullopt, dtype, layout, device, pin_memory);
}

// 形式 4: randint(low, high, size, generator, ...) - 核心实现
Tensor randint(
    int64_t low,
    int64_t high,
    IntArrayRef size,
    std::optional<Generator> generator,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 构造 TensorOptions
  TensorOptions options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 1. 创建未初始化张量
  auto result = at::empty(size, options);

  // 2. 原地填充随机整数（调用 random_）
  return result.random_(low, high, std::move(generator));
}
```

**输出参数版本**：
```cpp
// randint_out: 复用现有张量
Tensor& randint_out(
    int64_t high,
    IntArrayRef size,
    std::optional<Generator> generator,
    Tensor& result) {
  result.resize_(size);  // 调整大小
  return result.random_(0, high, std::move(generator));
}
```

**关键点**：
- 所有 `randint` 最终都调用 `empty() + random_()`
- 默认 dtype 是 `torch.int64`
- Generator 为 `std::nullopt` 时使用默认生成器

### 2.2 randperm 入口函数

文件位置: `aten/src/ATen/native/TensorFactories.cpp:1371-1402`

```cpp
Tensor randperm(
    int64_t n,
    std::optional<Generator> generator,
    std::optional<ScalarType> dtype,
    std::optional<Layout> layout,
    std::optional<Device> device,
    std::optional<bool> pin_memory) {
  // 默认 dtype 是 int64
  if (!dtype.has_value()) {
    dtype = ScalarType::Long;
  }

  TensorOptions options =
      TensorOptions().dtype(dtype).layout(layout).device(device).pinned_memory(pin_memory);

  // 1. 创建长度为 n 的张量
  auto tensor = at::empty(n, options);

  // 2. 调用 randperm_out 填充排列
  return at::randperm_out(tensor, n, std::move(generator));
}
```

**关键点**：
- `randperm` 有独立的实现路径，不使用 `random_`
- CPU 和 CUDA 有不同的 `randperm_out` 实现
- CUDA 版本使用**基于排序的算法**（见第 4.3 节）

### 2.3 random_ 的三种重载

文件位置: `aten/src/ATen/native/Distributions.cpp:353-372`

```cpp
// 形式 1: random_(generator) - 使用数据类型的完整范围
Tensor& random_(Tensor& self, std::optional<Generator> gen) {
  return at::native::templates::random_impl<RandomStub, Generator>(
      self, std::move(gen));
}

// 形式 2: random_(to, generator) - [0, to)
Tensor& random_(Tensor& self, int64_t to, std::optional<Generator> gen) {
  return random_(self, 0, to, std::move(gen));
}

// 形式 3: random_(from, to, generator) - [from, to) - 核心实现
Tensor& random_(Tensor& self, int64_t from, std::optional<int64_t> to,
                std::optional<Generator> gen) {
  return at::native::templates::random_from_to_impl<RandomFromToStub, Generator>(
      self, from, to, std::move(gen));
}
```

**Stub 分派结构**：
```cpp
template<typename RNG>
struct RandomFromToStub {
  // 有范围版本
  void operator()(TensorIteratorBase& iter, uint64_t range, int64_t from,
                  std::optional<Generator> gen) {
    random_from_to_stub(iter.device_type(), iter, range, from, gen);
    // 根据 device_type 分派到 CPU 或 CUDA 实现
  }

  // 完整 64 位范围版本
  void operator()(TensorIteratorBase& iter, std::optional<Generator> gen) {
    random_full_64_bits_range_stub(iter.device_type(), iter, gen);
  }
};
```

**关键点**：
- 使用 **Dispatcher 分派机制**根据设备类型（CPU/CUDA）选择实现
- `random_from_to_stub` 和 `random_full_64_bits_range_stub` 是分派 stub
- TensorIterator 用于处理多种张量布局和步幅

---

## 3. 核心实现层

### 3.1 random_from_to_impl 模板函数

`random_from_to_impl` 处理范围检查和参数规范化，然后调用设备特定的 kernel。

**核心逻辑**（简化版）：
```cpp
template<typename Stub, typename Generator>
Tensor& random_from_to_impl(
    Tensor& self,
    int64_t from,
    std::optional<int64_t> to_opt,
    std::optional<Generator> gen) {

  // 1. 获取数据类型的有效范围
  int64_t min_value = /* dtype 的最小值 */;
  int64_t max_value = /* dtype 的最大值 */;

  // 2. 处理 to 参数
  int64_t to = to_opt.has_value() ? to_opt.value() : (max_value + 1);

  // 3. 范围检查
  TORCH_CHECK(from < to, "from must be less than to");
  TORCH_CHECK(from >= min_value && to <= max_value + 1, "out of range");

  // 4. 计算范围
  uint64_t range = static_cast<uint64_t>(to) - static_cast<uint64_t>(from);

  // 5. 创建 TensorIterator
  auto iter = TensorIterator::borrowing_nullary_op(self);

  // 6. 分派到 CPU 或 CUDA kernel
  if (range == 0) {
    // 特殊情况：完整 64 位范围
    Stub()(iter, gen);  // 调用 random_full_64_bits_range_kernel
  } else {
    Stub()(iter, range, from, gen);  // 调用 random_from_to_kernel
  }

  return self;
}
```

**关键决策点**：
- **range == 0** 表示完整 64 位范围（例如 int64 的 [-2^63, 2^63)）
- **range < 2^28**（非 FBCODE）或 **range < 2^32**（FBCODE）：使用 32 位随机数
- **range >= 2^28/2^32**：使用 64 位随机数（组合两个 32 位）

### 3.2 数据类型处理

不同数据类型有不同的有效范围：

| 数据类型 | 最小值 | 最大值 | 范围 |
|----------|--------|--------|------|
| `int8` | -128 | 127 | 256 |
| `int16` | -32768 | 32767 | 65536 |
| `int32` | -2^31 | 2^31 - 1 | 2^32 |
| `int64` | -2^63 | 2^63 - 1 | 2^64 |
| `uint8` | 0 | 255 | 256 |
| `bool` | 0 | 1 | 2 |
| `float16` | - | - | 2^11 (精度位数) |
| `bfloat16` | - | - | 2^8 |

**浮点数类型的特殊处理**：
- 浮点数存储整数值时受**精度位数**限制
- `float` 有 24 位精度，`double` 有 53 位精度
- 超过精度的整数会有**舍入误差**

---

## 4. CUDA 实现（重点）

### 4.1 CUDA random_from_to_kernel 入口

文件位置: `aten/src/ATen/native/cuda/DistributionRandomKernel.cu:8-11`

```cpp
void random_from_to_kernel(TensorIteratorBase& iter, uint64_t range,
                           int64_t base, std::optional<Generator> gen_) {
  // 获取 CUDA 生成器
  auto gen = get_generator_or_default<CUDAGeneratorImpl>(
      gen_, cuda::detail::getDefaultCUDAGenerator());

  // 调用模板实现
  at::native::templates::cuda::random_from_to_kernel(iter, range, base, gen);
}
```

### 4.2 CUDA 模板实现

文件位置: `aten/src/ATen/native/cuda/DistributionTemplates.h:281-347`

CUDA 实现根据 `range` 的大小选择不同策略：

```cpp
template<typename RNG>
void random_from_to_kernel(TensorIteratorBase& iter, uint64_t range,
                           int64_t base, RNG gen) {
  AT_DISPATCH_V2(iter.dtype(), "random_from_to_kernel_cuda", AT_WRAP([&] {
    #ifdef FBCODE_CAFFE2
      // Facebook 内部构建：range >= 2^32 时使用 64 位
      if (range >= 1ULL << 32) {
        // 使用 64 位随机数
        auto random_func = [range, base] __device__ (uint64_t rand) {
          return transformation::uniform_int_from_to<scalar_t>(rand, range, base);
        };

        distribution_nullary_kernel<scalar_t, uint64_t, ulonglong2>(iter, gen,
          // dist_func: 生成两个 64 位随机数
          [] __device__ (curandStatePhilox4_32_10_t* state) -> ulonglong2 {
            ulonglong2 ret;
            uint4 rand_val = curand4(state);  // 生成 4 个 32 位
            // 组合成两个 64 位
            ret.x = (static_cast<uint64_t>(rand_val.x) << 32) | rand_val.y;
            ret.y = (static_cast<uint64_t>(rand_val.z) << 32) | rand_val.w;
            return ret;
          },
          random_func);
      } else {
        // 使用 32 位随机数
        auto random_func = [range, base] __device__ (uint32_t rand) {
          return transformation::uniform_int_from_to<scalar_t>(rand, range, base);
        };

        distribution_nullary_kernel<scalar_t, uint32_t, uint4>(iter, gen,
          // dist_func: 生成 4 个 32 位随机数
          [] __device__ (curandStatePhilox4_32_10_t* state) -> uint4 {
            return curand4(state);
          },
          random_func);
      }
    #else
      // 开源构建：range >= 2^28 时使用 64 位（允许约 5% 偏差）
      if (range >= 1ULL << 28) {
        // ... 64 位路径（同上）
      } else {
        // ... 32 位路径（同上）
      }
    #endif
  }), /* 支持的数据类型 */);
}
```

**关键设计决策**：
- **FBCODE 版本**：`range >= 2^32` 时使用 64 位（严格无偏）
- **开源版本**：`range >= 2^28` 时使用 64 位（允许 ~5% modulo bias）
- **向量化**：`curand4` 一次生成 4 个 32 位随机数

### 4.3 uniform_int_from_to 变换函数

文件位置: `aten/src/ATen/core/TransformationHelper.h:41-43`

将 uniform 随机数转换为指定范围的整数：

```cpp
template <typename T, typename V>
C10_HOST_DEVICE inline T uniform_int_from_to(V val, uint64_t range, int64_t base) {
  // 简单 modulo + 偏移
  return static_cast<T>(static_cast<int64_t>((val % range) + base));
}
```

**算法分析**：
- **优点**：简单、快速、GPU 友好
- **缺点**：存在 **modulo bias**（见 Appendix A）
- **偏差大小**：当 `range` 远小于 `2^32` 或 `2^64` 时，偏差可忽略

**示例**：
```cpp
// 假设 val 是 32 位 uniform 随机数 [0, 2^32)
// range = 10, base = 5
// 期望输出：[5, 15) 的均匀整数

uint32_t val = 3141592653;  // 某个随机数
uint64_t range = 10;
int64_t base = 5;

int64_t result = (val % range) + base;
// result = (3141592653 % 10) + 5 = 3 + 5 = 8
```

**Modulo Bias 示例**：
```cpp
// 假设生成 3 位随机数 [0, 8)，映射到 [0, 5)
// 0 → 0, 1 → 1, 2 → 2, 3 → 3, 4 → 4
// 5 → 0, 6 → 1, 7 → 2
// 结果：0, 1, 2 出现概率 3/8，3, 4 出现概率 2/8（有偏！）

// PyTorch 的缓解措施：使用足够大的随机数范围
// 当 range << 2^32 时，偏差 < 0.01%
```

### 4.4 distribution_nullary_kernel 框架

文件位置: `aten/src/ATen/native/cuda/DistributionTemplates.h:106-169`

这是所有分布 kernel 的通用框架，负责：
1. 计算 Grid/Block 配置
2. 获取 Philox 状态
3. 启动 Grid-Stride Loop kernel

```cpp
template<typename scalar_t, typename accscalar_t, typename dist_func_return_t,
         typename RNG, typename dist_t, typename transform_t>
void distribution_nullary_kernel(
    at::TensorIteratorBase& iter,
    RNG gen,
    const dist_t& dist_func,        // 生成随机数的函数
    const transform_t transform_func) {  // 变换函数

  // 1. 计算 unroll_factor（一次处理多少个元素）
  const int unroll_factor = sizeof(dist_func_return_t) / sizeof(accscalar_t);
  // 例如：dist_func_return_t = uint4（16 字节），accscalar_t = uint32_t（4 字节）
  // unroll_factor = 16 / 4 = 4

  int64_t numel = iter.numel();
  if (numel == 0) return;

  // 2. 计算执行策略（Grid/Block 配置 + counter_offset）
  auto [counter_offset, grid, block] = calc_execution_policy(numel, unroll_factor);

  // 3. 获取 Philox 状态（seed + offset）
  PhiloxCudaState rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(counter_offset);
  }

  // 4. 处理超过 32 位索引的情况
  if (!iter.can_use_32bit_indexing()) {
    for (auto& sub_iter : iter.with_32bit_indexing()) {
      distribution_nullary_kernel<...>(sub_iter, gen, dist_func, transform_func);
    }
    return;
  }

  // 5. 启动 kernel
  char* out_data = (char*)iter.data_ptr(0);
  auto stream = at::cuda::getCurrentCUDAStream();

  if (iter.is_trivial_1d()) {
    // 连续内存，简化索引计算
    int stride0 = iter.get_inner_strides()[0];
    distribution_elementwise_grid_stride_kernel<accscalar_t, unroll_factor>
        <<<grid, block, 0, stream>>>(
      numel, rng_engine_inputs, dist_func,
      [=]__device__(int idx, accscalar_t rand) {
        scalar_t* out = (scalar_t*)&out_data[stride0 * idx];
        *out = transform_func(rand);
      }
    );
  } else {
    // 非连续内存，使用 offset calculator
    auto offset_calc = make_offset_calculator<1>(iter);
    distribution_elementwise_grid_stride_kernel<accscalar_t, unroll_factor>
        <<<grid, block, 0, stream>>>(
      numel, rng_engine_inputs, dist_func,
      [=]__device__(int idx, accscalar_t rand) {
        auto offsets = offset_calc.get(idx);
        scalar_t* out = (scalar_t*)&out_data[offsets[0]];
        *out = transform_func(rand);
      }
    );
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
```

**关键点**：
- **模板参数**：支持任意分布函数和变换函数
- **Grid-Stride Loop**：高效利用 GPU，处理任意大小张量
- **向量化**：`unroll_factor` 支持一次处理多个元素
- **灵活索引**：支持连续和非连续张量布局

### 4.5 distribution_elementwise_grid_stride_kernel

文件位置: `aten/src/ATen/native/cuda/DistributionTemplates.h:64-88`

这是实际执行的 CUDA kernel：

```cpp
template<typename accscalar_t, int unroll_factor, typename dist_t, typename transform_t>
C10_LAUNCH_BOUNDS_2(block_size_bound, grid_size_bound)
__global__ void distribution_elementwise_grid_stride_kernel(
    int64_t numel,
    PhiloxCudaState philox_args,
    const dist_t dist_func,
    const transform_t transform_func) {

  // 1. 解包 Philox 状态
  auto [seed, offset] = at::cuda::philox::unpack(philox_args);

  // 2. 计算线程全局索引
  int64_t idx = ((int64_t)blockIdx.x) * blockDim.x + threadIdx.x;

  // 3. 初始化 Philox 状态
  curandStatePhilox4_32_10_t state;
  curand_init(seed, idx, offset, &state);

  // 4. Grid-Stride Loop
  int64_t rounded_size = ((numel - 1) / (blockDim.x * gridDim.x * unroll_factor) + 1) *
                         blockDim.x * gridDim.x * unroll_factor;

  for (int64_t linear_index = idx;
       linear_index < rounded_size;
       linear_index += blockDim.x * gridDim.x * unroll_factor) {

    // 5. 生成随机数（例如 uint4，包含 4 个 uint32_t）
    auto rand = dist_func(&state);  // 调用 curand4(&state)

    // 6. Unroll loop：处理 4 个元素
    #pragma unroll
    for (int ii = 0; ii < unroll_factor; ii++) {
      int64_t li = linear_index + blockDim.x * gridDim.x * ii;
      if (li < numel) {
        // 7. 变换并写入输出
        // (&rand.x)[ii] 访问 rand.x, rand.y, rand.z, rand.w
        transform_func(li, static_cast<accscalar_t>((&rand.x)[ii]));
      }
    }

    __syncthreads();  // 同步（通常不必要，但保留用于调试）
  }
}
```

**Grid-Stride Loop 详解**：
```
假设：
- numel = 10000（要生成的随机数个数）
- gridDim.x = 100（Grid 大小）
- blockDim.x = 256（Block 大小）
- unroll_factor = 4（一次处理 4 个）

线程 0 的迭代：
- Iteration 1: 处理索引 0, 25600, 51200, 76800（每次跳 blockDim * gridDim * unroll_factor）
- Iteration 2: 处理索引 102400, ... （超出范围，结束）

总共：100 * 256 = 25600 个线程
每个线程迭代次数：ceil(10000 / 25600) = 1 次
每次迭代处理：min(4, 剩余元素) 个元素
```

**向量化示例**：
```cpp
// dist_func 返回 uint4（4 个 uint32_t）
uint4 rand = curand4(&state);
// rand.x, rand.y, rand.z, rand.w 包含 4 个随机数

// unroll loop 展开：
transform_func(li + 0 * stride, rand.x);  // 处理第 1 个元素
transform_func(li + 1 * stride, rand.y);  // 处理第 2 个元素
transform_func(li + 2 * stride, rand.z);  // 处理第 3 个元素
transform_func(li + 3 * stride, rand.w);  // 处理第 4 个元素
```

### 4.6 randperm CUDA 实现（特殊算法）

文件位置: `aten/src/ATen/native/cuda/Randperm.cu:58-131`

`torch.randperm` 使用**基于排序的算法**而非传统的 Fisher-Yates shuffle：

```cpp
Tensor& randperm_out_cuda(int64_t n, std::optional<Generator> generator, Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
  result.resize_({n});

  // 1. 生成 arange 序列 [0, 1, 2, ..., n-1]
  auto range = at::arange(n, result.options());

  // 2. 计算需要的随机 key 位数
  // 见 [Algorithm of randperm] 注释
  const double log_threshold_12 = std::log(0.9) * 12;
  double nd = static_cast<double>(n);
  int bits = std::min(64,
      static_cast<int>(std::ceil(std::log2(nd - (6 * nd * nd + 1) / log_threshold_12))));

  if (n == 0) return result;

  Tensor shuffled_data = /* 输出数据指针 */;

  if (bits <= 32) {
    // 3a. 使用 32 位随机 key
    auto keys = at::empty(result.sizes(), TensorOptions().device(result.device()).dtype(kInt))
                  .random_(std::numeric_limits<int>::min(),
                          std::numeric_limits<int>::max(), generator);
    auto keys_tmp = at::empty_like(keys);
    auto keys_out = keys_tmp.mutable_data_ptr<int>();

    AT_DISPATCH_ALL_TYPES_AND(kHalf, result.scalar_type(), "randperm_out_cuda", [&] {
      // 4. 基数排序（radix sort）：按 keys 对 range 排序
      at::cuda::cub::radix_sort_pairs<int, dtype>(
        keys.const_data_ptr<int>(), keys_out,
        range_data, shuffled_data_,
        n, false, 0, bits);

      // 5. 处理重复的 key（使用 Fisher-Yates）
      randperm_handle_duplicate_keys(keys_out, shuffled_data_, bits, n, generator);
    });
  } else {
    // 3b. 使用 64 位随机 key（同样的流程）
    auto keys = at::empty(...).random_(std::numeric_limits<int64_t>::min(),
                                       std::numeric_limits<int64_t>::max(), generator);
    // ... 64 位排序 ...
  }

  return result;
}
```

**算法步骤**：
1. 生成 arange 序列 `[0, 1, 2, ..., n-1]`
2. 生成 n 个随机 key（32 位或 64 位）
3. 按 key 对 arange 进行排序（使用 CUB 的 radix sort）
4. 处理 key 重复的"岛"（使用 Fisher-Yates shuffle）

**为什么不直接用 Fisher-Yates？**
- Fisher-Yates 是串行算法，难以并行
- 基于排序的算法可以利用 GPU 的并行排序（CUB radix sort）
- 排序复杂度 O(n log n)，但 GPU 并行后实际更快

**处理重复 key**：

文件位置: `aten/src/ATen/native/cuda/Randperm.cuh:13-39`

```cpp
template<typename T, typename scalar_t>
__global__ void randperm_handle_duplicate_keys_kernel(
    T *keys, scalar_t *data, T mask, int n, PhiloxCudaState philox_args) {

  int tid = threadIdx.x + blockDim.x * blockIdx.x;

  // 1. 找到"岛"的开头（连续相同 key 的区域）
  if (tid >= n - 1) return;
  if ((keys[tid] & mask) != (keys[tid + 1] & mask)) return;  // 不在岛中
  if (tid != 0 && (keys[tid] & mask) == (keys[tid - 1] & mask)) return;  // 不是开头

  // 2. 计算岛的大小
  int island_size = 0;
  do { island_size++; }
  while ((tid + island_size < n) && (keys[tid + island_size] & mask) == (keys[tid] & mask));

  // 3. 对岛内元素进行 Fisher-Yates shuffle
  data += tid;
  const auto [seed, offset] = at::cuda::philox::unpack(philox_args);
  curandStatePhilox4_32_10_t state;
  curand_init(seed, tid, offset, &state);

  for (int i = island_size - 1; i > 0; i--) {
    unsigned int r = curand(&state) % (i + 1);
    if (i != r) {
      scalar_t tmp = data[i];
      data[i] = data[r];
      data[r] = tmp;
    }
  }
}
```

**岛检测示例**：
```
排序后的 keys（假设 bits=3，只看低 3 位）：
索引:  0   1   2   3   4   5   6   7   8
keys: 001 010 011 011 011 100 101 110 110
                ↑___岛___↑         ↑_岛_↑

线程 2 检测到岛的开头（keys[2] == keys[3] 且 keys[1] != keys[2]）
岛大小 = 3（索引 2, 3, 4）
对 data[2], data[3], data[4] 进行 Fisher-Yates shuffle
```

---

## 5. CPU 实现（简介）

### 5.1 random_from_to_kernel CPU 版本

文件位置: `aten/src/ATen/native/cpu/DistributionTemplates.h:21-34`

CPU 实现相对简单，使用 TensorIterator 的 lambda：

```cpp
template<typename RNG>
void random_from_to_kernel(TensorIteratorBase& iter, uint64_t range, int64_t base, RNG generator) {
  AT_DISPATCH_V2(iter.dtype(), "random_from_to_kernel_cpu", AT_WRAP([&] {
    std::lock_guard<std::mutex> lock(generator->mutex_);

    // 使用 cpu_serial_kernel 处理每个元素
    cpu_serial_kernel(iter, [range, base, generator]() -> scalar_t {
      // 创建 uniform_int_from_to_distribution
      uniform_int_from_to_distribution<scalar_t> random(range, base);
      // 调用 generator 生成随机数
      return random(generator);
    });
  }), /* 数据类型 */);
}
```

**uniform_int_from_to_distribution**：

文件位置: `aten/src/ATen/core/DistributionsHelper.h:37-62`

```cpp
template <typename T>
struct uniform_int_from_to_distribution {
  uint64_t range_;
  int64_t base_;

  template <typename RNG>
  C10_HOST_DEVICE inline T operator()(RNG generator) {
    // 根据 range 大小选择 32 位或 64 位随机数
    if (range_ >= 1ULL << 28) {  // 开源版本的阈值
      return transformation::uniform_int_from_to<T>(
          generator->random64(), range_, base_);
    } else {
      return transformation::uniform_int_from_to<T>(
          generator->random(), range_, base_);
    }
  }
};
```

**CPU vs CUDA 对比**：
| 特性 | CPU | CUDA |
|------|-----|------|
| RNG 引擎 | MT19937 | Philox |
| 并行性 | 串行（可用 OpenMP） | 大规模并行 |
| 向量化 | 无（一次 1 个） | 有（一次 4 个） |
| 内存访问 | 按序遍历 | Grid-Stride Loop |
| 算法 | 完全相同（modulo） | 完全相同（modulo） |

### 5.2 randperm CPU 实现

文件位置: `aten/src/ATen/native/TensorFactories.cpp:1404-1449`

CPU 版本使用**经典的 Fisher-Yates shuffle**：

```cpp
Tensor& randperm_out_cpu(
    int64_t n,
    std::optional<Generator> generator,
    Tensor& result) {
  TORCH_CHECK(n >= 0, "n must be non-negative, got", n);
  result.resize_({n});

  auto gen = get_generator_or_default<CPUGeneratorImpl>(generator, ...);

  // 1. 填充 [0, 1, 2, ..., n-1]
  AT_DISPATCH_ALL_TYPES_AND(ScalarType::Half, result.scalar_type(), "randperm_out_cpu", [&] {
    scalar_t *r__data = result.data_ptr<scalar_t>();

    // 填充 arange
    for (const auto i : c10::irange(n)) {
      r__data[i] = static_cast<scalar_t>(i);
    }

    // 2. Fisher-Yates shuffle
    std::lock_guard<std::mutex> lock(gen->mutex_);
    for (int64_t i = 0; i < n - 1; i++) {
      // 生成 [0, n-i) 的随机索引
      int64_t z = gen->random() % (n - i);

      // 交换 r__data[i] 和 r__data[i + z]
      scalar_t sav = r__data[i];
      r__data[i] = r__data[i + z];
      r__data[i + z] = sav;
    }
  });

  return result;
}
```

**Fisher-Yates 算法**（详见 Appendix B）：
```python
for i in range(n - 1):
    j = random.randint(i, n - 1)  # [i, n-1] 的随机索引
    swap(arr[i], arr[j])
```

**CPU vs CUDA randperm 对比**：
| 特性 | CPU | CUDA |
|------|-----|------|
| 算法 | Fisher-Yates shuffle | 基于排序 + 岛处理 |
| 复杂度 | O(n) | O(n log n) |
| 并行性 | 串行 | 高度并行 |
| 实际性能 | n < 10000 较快 | n > 10000 较快 |

---

## 6. 完整调用流程图（CUDA）

### 6.1 torch.randint 流程图

```
用户代码:
torch.randint(5, 15, (1000, 1000), device='cuda')
    ↓
Python 层: torch._C.randint
    ↓
C++ 分派层: at::native::randint(low=5, high=15, size=[1000, 1000], generator=nullptr, ...)
    ↓
TensorFactories.cpp:1149-1165
    ↓ 步骤 1: 创建未初始化张量
result = at::empty([1000, 1000], device='cuda', dtype=torch.int64)
    ↓ 步骤 2: 原地填充随机整数
result.random_(5, 15, nullptr)
    ↓
Distributions.cpp:366-368
at::native::templates::random_from_to_impl<RandomFromToStub, Generator>(result, 5, 15, nullptr)
    ↓
【核心实现层】
    ↓ 计算 range = 15 - 5 = 10
    ↓ 创建 TensorIterator
iter = TensorIterator::borrowing_nullary_op(result)
    ↓
    ↓ 分派到 CUDA kernel
RandomFromToStub()(iter, range=10, base=5, gen=nullptr)
    ↓
DistributionRandomKernel.cu:8-11
random_from_to_kernel(iter, range=10, base=5, gen_)
    ↓ 获取默认 CUDA 生成器
gen = get_generator_or_default<CUDAGeneratorImpl>(gen_, ...)
    ↓
DistributionTemplates.h:281-347
templates::cuda::random_from_to_kernel(iter, range=10, base=5, gen)
    ↓
【关键决策】range=10 < 2^28，使用 32 位随机数路径
    ↓
AT_DISPATCH_V2 分派到 int64 类型处理
    ↓
    ↓ 定义变换函数
auto random_func = [range=10, base=5] __device__ (uint32_t rand) {
    return transformation::uniform_int_from_to<int64_t>(rand, 10, 5);
};
    ↓
    ↓ 调用通用 kernel 框架
distribution_nullary_kernel<int64_t, uint32_t, uint4>(
    iter, gen,
    dist_func = [] __device__ (curandStatePhilox4_32_10_t* state) {
        return curand4(state);  // 生成 4 个 uint32_t
    },
    transform_func = random_func
)
    ↓
【Kernel 启动前准备】
    ↓ 步骤 1: 计算执行策略
auto [counter_offset, grid, block] = calc_execution_policy(numel=1000000, unroll_factor=4)
// 假设结果: counter_offset=16, grid=dim3(108), block=dim3(256)
    ↓
    ↓ 步骤 2: 获取 Philox 状态【Generator 第一次使用】
{
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_cuda_state(counter_offset=16);
    // 返回 PhiloxCudaState{seed=..., offset=...}
    // 同时更新 gen 内部 offset += 16
}
    ↓
    ↓ 步骤 3: 启动 CUDA kernel
distribution_elementwise_grid_stride_kernel<uint32_t, 4>
    <<<grid=dim3(108), block=dim3(256), 0, stream>>>(
    numel=1000000,
    philox_args=rng_engine_inputs,
    dist_func=curand4,
    transform_func=[](int idx, uint32_t rand) {
        int64_t* out = &result[idx];
        *out = uniform_int_from_to<int64_t>(rand, 10, 5);
    }
)
    ↓
【设备端执行】
    ↓
Kernel 内部: distribution_elementwise_grid_stride_kernel
    ↓ 步骤 1: 解包 Philox 状态
auto [seed, offset] = at::cuda::philox::unpack(philox_args);
    ↓
    ↓ 步骤 2: 初始化线程的 Philox 状态
int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;  // 线程全局索引
curandStatePhilox4_32_10_t state;
curand_init(seed, idx, offset, &state);
    ↓
    ↓ 步骤 3: Grid-Stride Loop
for (int64_t linear_index = idx; linear_index < 1000000; linear_index += 27648) {
    ↓
    ↓ 步骤 3a: 生成 4 个随机数
    uint4 rand = curand4(&state);  // rand.x, rand.y, rand.z, rand.w
    ↓
    ↓ 步骤 3b: Unroll loop 处理 4 个元素
    #pragma unroll
    for (int ii = 0; ii < 4; ii++) {
        int64_t li = linear_index + 27648 * ii;
        if (li < 1000000) {
            uint32_t rand_val = (&rand.x)[ii];  // 取出一个随机数
            ↓
            ↓ 步骤 3c: 变换并写入
            int64_t output = (rand_val % 10) + 5;  // [5, 15)
            result[li] = output;
        }
    }
}
    ↓
【完成】
CUDA kernel 执行完毕
    ↓
返回 result（包含 [5, 15) 的随机整数）
```

**关键步骤说明**：
1. **内存分配**：`at::empty` 创建未初始化的 CUDA 张量
2. **Generator 使用**：只在主机端获取 seed 和 offset
3. **向量化生成**：`curand4` 一次生成 4 个 32 位随机数
4. **Grid-Stride Loop**：每个线程处理多个元素，高效利用 GPU
5. **Modulo 变换**：`(rand % range) + base` 转换为目标范围

### 6.2 torch.randperm 流程图

```
用户代码:
torch.randperm(1000, device='cuda')
    ↓
Python 层: torch._C.randperm
    ↓
C++ 分派层: at::native::randperm(n=1000, generator=nullptr, dtype=torch.int64, device='cuda')
    ↓
TensorFactories.cpp:1380-1398
    ↓ 步骤 1: 创建空张量
tensor = at::empty(1000, device='cuda', dtype=torch.int64)
    ↓ 步骤 2: 调用 randperm_out
at::randperm_out(tensor, n=1000, generator=nullptr)
    ↓
【分派到 CUDA 实现】
Randperm.cu:58
randperm_out_cuda(n=1000, generator=nullptr, result=tensor)
    ↓
【算法准备】
    ↓ 步骤 1: 生成 arange 序列
range = at::arange(1000, device='cuda', dtype=torch.int64)
// range = [0, 1, 2, ..., 999]
    ↓
    ↓ 步骤 2: 计算随机 key 的位数
nd = 1000.0
bits = ceil(log2(nd - (6 * nd^2 + 1) / (12 * log(0.9))))
     = ceil(log2(1000 - ...))
     = 20  // 需要 20 位随机 key（假设）
    ↓
    ↓ 步骤 3: 生成随机 keys（32 位路径）
keys = at::empty(1000, device='cuda', dtype=torch.int32)
       .random_(std::numeric_limits<int>::min(),  // -2^31
                std::numeric_limits<int>::max(),  // 2^31 - 1
                generator)
// keys = [随机 int32 数组]
    ↓
【排序阶段】
    ↓ 步骤 4: Radix Sort（CUB 库）
keys_tmp = at::empty_like(keys)
keys_out = keys_tmp.mutable_data_ptr<int>()

at::cuda::cub::radix_sort_pairs<int, int64_t>(
    keys_in = keys.data_ptr<int>(),
    keys_out = keys_out,
    values_in = range.data_ptr<int64_t>(),    // [0, 1, 2, ..., 999]
    values_out = result.data_ptr<int64_t>(),  // 输出排列
    num_items = 1000,
    begin_bit = 0,
    end_bit = 20  // 只比较低 20 位
)

// 排序后:
// keys_out 是排序后的 keys（只看低 20 位）
// result 是根据 keys 排序的 range（这就是排列！）
    ↓
【处理重复 key】
    ↓ 步骤 5: 检测并处理 key 重复的"岛"
randperm_handle_duplicate_keys(keys_out, result.data_ptr(), bits=20, n=1000, generator)
    ↓
Randperm.cuh:43-56
    ↓ 获取 Philox 状态
gen = get_generator_or_default<CUDAGeneratorImpl>(generator, ...)
rng_engine_inputs = gen->philox_cuda_state(counter_offset=1000)
    ↓
    ↓ 启动岛处理 kernel
mask = (1 << 20) - 1  // 0x000FFFFF（低 20 位掩码）
randperm_handle_duplicate_keys_kernel<<<(1000+511)/512, 512>>>(
    keys_out, result.data_ptr(), mask, n=1000, rng_engine_inputs
)
    ↓
【设备端：岛处理 kernel】
Randperm.cuh:13-39
    ↓
for each thread tid in [0, 1000):
    ↓
    ↓ 步骤 1: 检测是否是岛的开头
    if (keys_out[tid] & mask) != (keys_out[tid+1] & mask):
        continue  // 不在岛中
    if tid > 0 and (keys_out[tid-1] & mask) == (keys_out[tid] & mask):
        continue  // 不是岛的开头

    ↓ 步骤 2: 计算岛的大小
    island_size = 0
    while (keys_out[tid + island_size] & mask) == (keys_out[tid] & mask):
        island_size++

    ↓ 步骤 3: 对岛内元素进行 Fisher-Yates shuffle
    curandStatePhilox4_32_10_t state
    curand_init(seed, tid, offset, &state)

    for i in range(island_size - 1, 0, -1):
        r = curand(&state) % (i + 1)
        swap(result[tid + i], result[tid + r])
    ↓
【完成】
返回 result（[0, 999] 的随机排列）
```

**关键点说明**：
1. **基于排序**：利用 GPU 的高效并行排序（CUB radix sort）
2. **概率保证**：通过精心选择 bits，保证 key 重复概率 < 10%
3. **岛处理**：只有少数"岛"需要 Fisher-Yates，大部分元素已正确排列
4. **复杂度**：排序 O(n log n)，但 GPU 并行后实际比串行 Fisher-Yates 快

---

## 7. 实现对比

### 7.1 torch.randint vs torch.rand

| 特性 | torch.randint | torch.rand |
|------|---------------|------------|
| **输出类型** | 整数 | 浮点数 |
| **输出范围** | [low, high) 离散 | [0, 1) 连续 |
| **CUDA 生成函数** | `curand4()` (uint32) | `curand_uniform4()` (float) |
| **变换方法** | modulo + 偏移 | 线性映射 |
| **偏差** | 有轻微 modulo bias | 无偏 |
| **典型用途** | 索引、标签 | 权重初始化、掩码 |

**实现对比**：
```cpp
// torch.rand 的变换
float uniform_real(uint32_t rand) {
    constexpr uint32_t MASK = (1 << 24) - 1;  // 24 位精度
    constexpr float DIVISOR = 1.0f / (1 << 24);
    return (rand & MASK) * DIVISOR;  // [0, 1)
}

// torch.randint 的变换
int64_t uniform_int_from_to(uint32_t rand, uint64_t range, int64_t base) {
    return static_cast<int64_t>((rand % range) + base);  // [base, base+range)
}
```

### 7.2 torch.randperm CPU vs CUDA

| 特性 | CPU | CUDA |
|------|-----|------|
| **算法** | Fisher-Yates shuffle | 基于排序 + 岛处理 |
| **时间复杂度** | O(n) | O(n log n) |
| **空间复杂度** | O(n) | O(n) + 排序缓冲 |
| **并行性** | 串行 | 高度并行 |
| **实际性能** | n < 10000 更快 | n > 10000 更快 |
| **确定性** | 完全确定 | 依赖 key 碰撞概率 |

**性能转折点**（近似）：
```python
import torch
import time

# CPU 实现
n = 10000
start = time.time()
torch.randperm(n, device='cpu')
cpu_time = time.time() - start

# CUDA 实现
start = time.time()
torch.cuda.synchronize()
torch.randperm(n, device='cuda')
torch.cuda.synchronize()
cuda_time = time.time() - start

print(f"CPU: {cpu_time:.6f}s, CUDA: {cuda_time:.6f}s")
# n=1000:   CPU 更快
# n=10000:  接近
# n=100000: CUDA 更快
```

### 7.3 random_ 的三种形式对比

| 形式 | 参数 | 范围 | 使用场景 |
|------|------|------|----------|
| `random_(gen)` | 仅 generator | dtype 的完整范围 | 测试、初始化 |
| `random_(to, gen)` | to, generator | [0, to) | 常用形式（randint 默认） |
| `random_(from, to, gen)` | from, to, generator | [from, to) | 完整控制（randint 核心） |

**示例对比**：
```python
import torch

x = torch.empty(10, dtype=torch.int32)

# 形式 1: 完整范围 [-2^31, 2^31)
x.random_()
# 可能输出: tensor([-1234567890, 987654321, ...])

# 形式 2: [0, 100)
x.random_(100)
# 可能输出: tensor([23, 87, 5, 99, 12, ...])

# 形式 3: [50, 150)
x.random_(50, 150)
# 可能输出: tensor([78, 123, 56, 149, 91, ...])
```

---

## 8. 与其他分布的实现对比

### 8.1 算法对比

| 分布 | 基础 RNG | 变换方法 | 复杂度 | 偏差 |
|------|----------|----------|--------|------|
| **uniform** (float) | `curand_uniform4` | 直接生成 | O(1) | 无 |
| **randint** | `curand4` → uint32 | modulo + 偏移 | O(1) | 轻微 |
| **normal** | `curand_normal4` | cuRAND 内置 | O(1) | 无 |
| **exponential** | `curand_uniform4` | -log(1-U)/λ | O(log) | 无 |
| **multinomial** | `curand_uniform4` | 累积概率搜索 | O(k log n) | 无 |
| **randperm** | `curand4` | 排序 + shuffle | O(n log n) | 可忽略 |

### 8.2 性能对比

以生成 1000×1000 张量为例（CUDA，单位: ms）：

| 操作 | 时间 | 相对速度 |
|------|------|----------|
| `torch.rand(1000, 1000, device='cuda')` | 0.05 | 1.0× |
| `torch.randint(0, 100, (1000, 1000), device='cuda')` | 0.06 | 1.2× |
| `torch.randn(1000, 1000, device='cuda')` | 0.08 | 1.6× |
| `torch.randperm(1000000, device='cuda')` | 2.5 | 50× |

**性能说明**：
- **uniform/randint/normal** 性能相近（都是向量化生成）
- **randperm** 慢得多（需要排序）
- **实际性能**取决于 GPU 型号和张量大小

---

## Appendix A: Modulo Bias 问题

### A.1 什么是 Modulo Bias

当使用 modulo 操作将大范围的随机数映射到小范围时，会产生**不均匀分布**。

**简单示例**：
```python
# 假设生成 3 位随机数 [0, 8)，映射到 [0, 5)
# 使用 modulo: result = rand % 5

rand = 0 → 0 % 5 = 0
rand = 1 → 1 % 5 = 1
rand = 2 → 2 % 5 = 2
rand = 3 → 3 % 5 = 3
rand = 4 → 4 % 5 = 4
rand = 5 → 5 % 5 = 0  ← 重复！
rand = 6 → 6 % 5 = 1  ← 重复！
rand = 7 → 7 % 5 = 2  ← 重复！

概率分布:
0: 2/8 = 25%   ← 偏高
1: 2/8 = 25%   ← 偏高
2: 2/8 = 25%   ← 偏高
3: 1/8 = 12.5% ← 偏低
4: 1/8 = 12.5% ← 偏低
```

**偏差大小**：
```
bias = (rand_range % target_range) / rand_range

例如：rand_range = 8, target_range = 5
bias = (8 % 5) / 8 = 3/8 = 37.5%  ← 严重偏差！
```

### A.2 PyTorch 的缓解措施

PyTorch 使用**足够大的随机数范围**来减小偏差：

| rand_range | target_range | 最大偏差 |
|------------|--------------|----------|
| 2^32 | 100 | 100 / 2^32 ≈ 0.000002% |
| 2^32 | 10^6 | 10^6 / 2^32 ≈ 0.02% |
| 2^32 | 2^31 | 2^31 / 2^32 = 50% ← 需要 64 位！ |
| 2^64 | 2^32 | 2^32 / 2^64 ≈ 0.000002% |

**PyTorch 的策略**：
```cpp
// 开源版本：range >= 2^28 时使用 64 位
if (range >= 1ULL << 28) {  // 268435456
    // 使用 64 位随机数
    uint64_t rand = (curand4().x << 32) | curand4().y;
    return (rand % range) + base;
} else {
    // 使用 32 位随机数
    uint32_t rand = curand4().x;
    return (rand % range) + base;
}

// FBCODE 版本：range >= 2^32 时使用 64 位（更严格）
```

**偏差示例**：
```python
import torch

# 小范围：偏差可忽略
x = torch.randint(0, 100, (1000000,))
# 理论概率: 1/100 = 1%
# 实际频率: ~1% ± 0.0001%

# 大范围：需要 64 位
y = torch.randint(0, 2**31, (1000000,), dtype=torch.int64)
# 使用 32 位会有 50% 偏差
# PyTorch 自动切换到 64 位，偏差 < 0.001%
```

### A.3 拒绝采样（Rejection Sampling）

更严格的无偏方法是**拒绝采样**：

```cpp
// 无偏的整数生成（PyTorch 未采用）
uint64_t uniform_int_unbiased(uint64_t range) {
    uint64_t limit = UINT64_MAX - (UINT64_MAX % range);
    uint64_t r;
    do {
        r = random_uint64();
    } while (r >= limit);  // 拒绝超过 limit 的值
    return r % range;
}
```

**为什么 PyTorch 不用拒绝采样？**
1. **性能**：拒绝采样需要循环，GPU 上难以高效实现
2. **偏差可控**：modulo bias 在实际应用中可忽略（< 0.01%）
3. **简单**：modulo 方法简单、快速、向量化友好

**拒绝概率**：
```
rejection_rate = (rand_range % target_range) / rand_range

例如：range = 100, rand_range = 2^32
rejection_rate = 100 / 2^32 ≈ 0.000002%  ← 几乎不拒绝
```

---

## Appendix B: Fisher-Yates Shuffle 算法

### B.1 算法原理

Fisher-Yates shuffle 是生成均匀随机排列的标准算法。

**算法步骤**：
```python
def fisher_yates_shuffle(arr):
    n = len(arr)
    for i in range(n - 1):
        # 从 [i, n-1] 中随机选择一个索引
        j = random.randint(i, n - 1)
        # 交换 arr[i] 和 arr[j]
        arr[i], arr[j] = arr[j], arr[i]
    return arr
```

**示例**：
```
初始: [0, 1, 2, 3, 4]

i=0: j=random(0, 4)=2, swap(arr[0], arr[2]) → [2, 1, 0, 3, 4]
i=1: j=random(1, 4)=4, swap(arr[1], arr[4]) → [2, 4, 0, 3, 1]
i=2: j=random(2, 4)=3, swap(arr[2], arr[3]) → [2, 4, 3, 0, 1]
i=3: j=random(3, 4)=4, swap(arr[3], arr[4]) → [2, 4, 3, 1, 0]

结果: [2, 4, 3, 1, 0]（一个随机排列）
```

### B.2 正确性证明

**定理**：Fisher-Yates 算法生成 n! 种排列中的任意一种，每种概率为 1/n!

**证明（归纳法）**：
- **基础情况** (n=2)：
  - i=0, j∈{0,1}，概率各 1/2
  - j=0: [0,1], j=1: [1,0]
  - 两种排列概率都是 1/2 = 1/2! ✓

- **归纳假设**：对 n-1 个元素，算法正确
- **归纳步骤**：对 n 个元素
  - 第 i 步从 [i, n-1] 选 j，概率 1/(n-i)
  - 最终某个元素出现在位置 i 的概率：
    ```
    P = (1/n) × (1/(n-1)) × ... × (1/1) = 1/n!
    ```
  - 任意排列的概率：1/n! ✓

### B.3 常见错误实现

**错误 1**：每次从整个范围随机选择
```python
# ❌ 错误！
def wrong_shuffle(arr):
    n = len(arr)
    for i in range(n):
        j = random.randint(0, n - 1)  # 应该是 random(i, n-1)
        arr[i], arr[j] = arr[j], arr[i]
```
**问题**：生成 n^n 种不同的交换序列，但只有 n! 种排列，分布不均匀。

**错误 2**：使用排序
```python
# ⚠️ 偏差取决于排序算法的稳定性
def sort_shuffle(arr):
    keys = [random.random() for _ in arr]
    return [x for _, x in sorted(zip(keys, arr))]
```
**问题**：
- 如果排序是**稳定的**（相同 key 保持原序），有偏差
- 如果 key 有碰撞，分布不均匀
- PyTorch CUDA 使用此方法，但通过**岛处理**修正偏差

### B.4 PyTorch 的实现

#### CPU 版本（标准 Fisher-Yates）

文件位置: `aten/src/ATen/native/TensorFactories.cpp:1404-1449`

```cpp
Tensor& randperm_out_cpu(int64_t n, std::optional<Generator> generator, Tensor& result) {
  result.resize_({n});

  AT_DISPATCH_ALL_TYPES_AND(ScalarType::Half, result.scalar_type(), "randperm_out_cpu", [&] {
    scalar_t *r__data = result.data_ptr<scalar_t>();

    // 1. 填充 [0, 1, ..., n-1]
    for (const auto i : c10::irange(n)) {
      r__data[i] = static_cast<scalar_t>(i);
    }

    // 2. Fisher-Yates shuffle
    std::lock_guard<std::mutex> lock(gen->mutex_);
    for (int64_t i = 0; i < n - 1; i++) {
      int64_t z = gen->random() % (n - i);  // [0, n-i)

      // 交换 r__data[i] 和 r__data[i + z]
      scalar_t sav = r__data[i];
      r__data[i] = r__data[i + z];
      r__data[i + z] = sav;
    }
  });

  return result;
}
```

#### CUDA 版本（岛内 Fisher-Yates）

文件位置: `aten/src/ATen/native/cuda/Randperm.cuh:13-39`

```cpp
// 只对重复 key 的"岛"使用 Fisher-Yates
template<typename T, typename scalar_t>
__global__ void randperm_handle_duplicate_keys_kernel(...) {
  int tid = ...;

  // 检测岛的开头
  if (/* 是岛的第一个线程 */) {
    // 计算岛的大小
    int island_size = ...;

    // Fisher-Yates shuffle（岛内）
    for (int i = island_size - 1; i > 0; i--) {
      unsigned int r = curand(&state) % (i + 1);
      if (i != r) {
        scalar_t tmp = data[i];
        data[i] = data[r];
        data[r] = tmp;
      }
    }
  }
}
```

### B.5 并行化的困难

Fisher-Yates 算法**本质上是串行的**：

```python
for i in range(n - 1):
    j = random(i, n - 1)
    swap(arr[i], arr[j])  # 这一步依赖前面所有的交换
```

**并行化尝试**：
1. **并行交换**：不行，交换之间有依赖
2. **分块处理**：可以，但需要全局同步
3. **基于排序**：可以，这就是 PyTorch CUDA 的方法！

**PyTorch 的解决方案**：
- 使用**并行排序**（CUB radix sort）代替串行 shuffle
- 只对少数重复 key 的"岛"使用串行 Fisher-Yates
- 大部分元素已经通过排序正确排列

---

## 常见问题（FAQ）

### Q1: torch.randint 的输出是均匀分布的吗？

**A**: 几乎是均匀的，但有**极轻微的 modulo bias**。

- 当 `range << 2^32`（例如 range=100），偏差 < 0.000002%，**可忽略**
- 当 `range >= 2^28`，PyTorch 自动切换到 64 位随机数，偏差仍 < 0.01%
- 在实际应用中（机器学习），这个偏差**完全不影响结果**

### Q2: 为什么 torch.randperm CUDA 比 CPU 慢？

**A**: 对于**小 n**（< 10000），CPU 的 Fisher-Yates 更快，因为：
- Fisher-Yates 是 O(n) 算法，排序是 O(n log n)
- 小数据量下，GPU kernel 启动开销占主导

对于**大 n**（> 100000），CUDA 的并行排序优势显现，**比 CPU 快得多**。

### Q3: random_ 的完整范围模式是如何工作的？

**A**: 不同数据类型有不同的完整范围：

```python
import torch

# int32: [-2^31, 2^31)
x = torch.empty(10, dtype=torch.int32)
x.random_()
print(x.min(), x.max())  # 接近 -2147483648, 2147483647

# int64: [-2^63, 2^63)
y = torch.empty(10, dtype=torch.int64)
y.random_()
print(y.min(), y.max())  # 接近 -9223372036854775808, 9223372036854775807

# bool: {0, 1}
z = torch.empty(10, dtype=torch.bool)
z.random_()
print(z)  # tensor([True, False, True, ...])
```

### Q4: 如何避免 modulo bias？

**A**: PyTorch 已经自动处理，但如果你需要**完全无偏的整数**：

```python
# 方法 1: 使用浮点数 uniform 再转换（PyTorch 推荐）
# （注意：这不是 randint 的实现，但可以作为替代）
import torch

def unbiased_randint(low, high, size):
    # 生成 [0, 1) 的浮点数，乘以范围并取整
    return (torch.rand(size) * (high - low)).long() + low

x = unbiased_randint(0, 100, (1000,))
# 完全无偏，但比 torch.randint 慢（需要浮点运算）

# 方法 2: 直接使用 torch.randint（偏差可忽略）
y = torch.randint(0, 100, (1000,))
# 有极轻微偏差（< 0.000002%），但更快
```

### Q5: 如何生成可复现的随机整数？

**A**: 使用自定义 Generator：

```python
import torch

# 方法 1: 全局种子
torch.manual_seed(42)
x = torch.randint(0, 100, (10,))
# 每次运行结果相同

# 方法 2: 独立生成器
gen = torch.Generator().manual_seed(42)
y = torch.randint(0, 100, (10,), generator=gen)
# 不影响全局随机数流

# 方法 3: CUDA 生成器
gen_cuda = torch.Generator(device='cuda').manual_seed(42)
z = torch.randint(0, 100, (10,), device='cuda', generator=gen_cuda)
```

---

## 总结

本文档详细介绍了 PyTorch 离散整数随机数生成的完整实现：

1. **三个主要 API**：
   - `torch.randint`: 工厂函数，创建新张量
   - `torch.randperm`: 生成随机排列（CUDA 使用排序算法）
   - `tensor.random_`: 原地操作，`randint` 的底层实现

2. **核心算法**：
   - **整数生成**：简单 modulo 方法（`(rand % range) + base`）
   - **偏差控制**：使用足够大的随机数范围（32 位或 64 位）
   - **CUDA 优化**：向量化生成（`curand4` 一次 4 个）+ Grid-Stride Loop

3. **randperm 的特殊实现**：
   - **CPU**：经典 Fisher-Yates shuffle（O(n)）
   - **CUDA**：基于排序 + 岛处理（O(n log n)，但并行）

4. **关键差异**：
   - 连续分布（uniform/normal）使用精确变换，无偏差
   - 离散分布（randint）使用 modulo，有轻微偏差但可控

5. **性能要点**：
   - randint 与 rand/randn 性能相近
   - randperm 慢得多（需要排序）
   - CUDA 在大规模数据上有显著优势

**下一步阅读**：
- [torch.rand/uniform 详解](random_call_flow.md) - 了解连续分布实现
- [torch.randn/normal 详解](normal_call_flow.md) - 了解 Box-Muller 变换
- [架构总览](distribution_architecture.md) - 了解整体设计

---

**文档版本**: v1.0
**对应 PyTorch 版本**: >= 2.0
**最后更新**: 2024-12-22
