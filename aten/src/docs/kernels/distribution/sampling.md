# PyTorch 复杂采样函数详解

## 概述

本文档详细介绍 PyTorch 中复杂采样函数的实现原理和调用流程，包括 `torch.multinomial`、`torch.poisson`、`torch.binomial` 和 `torch._standard_gamma` 等。这些采样函数无法通过简单的变换（如线性映射或 Box-Muller 变换）从 uniform/normal 分布得到，需要专门的采样算法。

**本文重点关注 CUDA 实现**，因为这些复杂采样算法在 GPU 上的并行化实现更具挑战性和实用价值。

**与其他分布的区别**：
- **简单分布**（uniform, normal, exponential）：直接数学变换，O(1) 时间
- **复杂采样**（multinomial, poisson, gamma）：需要条件判断、循环或拒绝采样

---

## 文档导航

- [1. Multinomial 采样](#1-multinomial-采样)
- [2. Poisson 分布](#2-poisson-分布)
- [3. Binomial 分布](#3-binomial-分布)
- [4. Gamma 分布](#4-gamma-分布)
- [5. Dirichlet 分布](#5-dirichlet-分布)
- [6. 实现对比总结](#6-实现对比总结)
- [Appendix A: Alias Method 详解](#appendix-a-alias-method-详解)
- [Appendix B: BTRS 算法详解](#appendix-b-btrs-算法详解)
- [Appendix C: Marsaglia-Tsang 算法详解](#appendix-c-marsaglia-tsang-算法详解)

---

## 1. Multinomial 采样

### 1.1 概述

`torch.multinomial` 实现多项式分布采样，广泛应用于分类模型、语言模型和强化学习中的动作选择。

**核心问题**：给定离散概率分布 $P = \{p_1, p_2, ..., p_n\}$，按此概率分布采样 $k$ 个样本。

**应用场景**：
```python
# 语言模型：按概率采样下一个 token
logits = model(input)  # [batch, vocab_size]
probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)

# 强化学习：按策略概率选择动作
action_probs = policy(state)  # [action_space]
action = torch.multinomial(action_probs, num_samples=1)
```

### 1.2 Python 层入口

**API 签名**：
```python
torch.multinomial(
    input: Tensor,           # 概率权重 (可以未归一化)
    num_samples: int,        # 采样数量
    replacement: bool = False,  # 是否有放回采样
    *,
    generator: Optional[Generator] = None,
    out: Optional[Tensor] = None
) -> Tensor
```

**参数说明**：
- `input`: 形状 `[n]` 或 `[batch, n]` 的概率权重（自动归一化）
- `num_samples`: 采样数量
- `replacement`:
  - `True`: 有放回采样（允许重复）
  - `False`: 无放回采样（不重复，要求 `num_samples <= n`）

**使用示例**：
```python
# 简单采样
weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
samples = torch.multinomial(weights, num_samples=10, replacement=True)
# 输出: tensor([3, 2, 3, 3, 1, 3, 2, 3, 2, 1])

# 批量采样
weights = torch.rand(8, 100)  # 8 个不同的分布
samples = torch.multinomial(weights, num_samples=5, replacement=True)
# 输出形状: [8, 5]

# CUDA 加速
weights = torch.rand(1000, device='cuda')
samples = torch.multinomial(weights, num_samples=100, replacement=True)
```

### 1.3 C++ 分派层

**文件**: `aten/src/ATen/native/Distributions.cpp:450`

```cpp
Tensor multinomial(
    const Tensor& self,
    int64_t num_samples,
    bool replacement,
    std::optional<Generator> gen) {

  // 参数检查
  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 2,
      "multinomial expects 1 or 2 dim tensor");

  TORCH_CHECK(
      num_samples > 0,
      "num_samples must be greater than 0");

  // 无放回采样的检查
  if (!replacement) {
    TORCH_CHECK(
        num_samples <= self.size(-1),
        "cannot sample num_samples > num_categories without replacement");
  }

  // 创建输出张量
  auto result_shape = self.dim() == 1
      ? IntArrayRef{num_samples}
      : IntArrayRef{self.size(0), num_samples};
  auto result = at::empty(result_shape, self.options().dtype(kLong));

  // 调用实现函数
  multinomial_out(result, self, num_samples, replacement, gen);

  return result;
}
```

### 1.4 核心实现层

#### 1.4.1 算法选择

PyTorch 根据不同情况选择不同的采样算法：

| 条件 | 算法 | 时间复杂度 | 适用场景 |
|------|------|-----------|---------|
| 小规模 + CPU | **累积概率二分搜索** | O(k log n) | n < 1000 |
| 大规模 + 重复采样 | **Alias Method** | O(n) 预处理 + O(1) 采样 | 同一分布多次采样 |
| 无放回采样 | **Gumbel-Max Trick** | O(n + k log n) | replacement=False |
| CUDA | **并行累积概率法** | O(k) 并行 | GPU 加速 |

#### 1.4.2 CPU 实现：累积概率法

**文件**: `aten/src/ATen/native/cpu/MultinomialKernel.cpp:25`

```cpp
// CPU 实现：累积概率 + 二分搜索
void multinomial_kernel_impl(
    Tensor& result,
    const Tensor& self,
    int64_t num_samples,
    bool replacement,
    Generator gen) {

  auto n_categories = self.size(-1);
  auto n_dist = self.dim() == 1 ? 1 : self.size(0);

  // 1. 构建累积概率 (Cumulative Distribution Function)
  Tensor cum_weights = self.cumsum(-1);  // [batch, n]

  for (int64_t i = 0; i < n_dist; i++) {
    for (int64_t j = 0; j < num_samples; j++) {
      // 2. 生成 uniform(0, total_weight)
      auto uniform_sample =
          uniform_real_distribution(0.0, cum_weights[i][-1].item<double>())(gen);

      // 3. 二分搜索找到对应的类别
      auto sample_idx =
          std::lower_bound(
              cum_weights[i].data_ptr<double>(),
              cum_weights[i].data_ptr<double>() + n_categories,
              uniform_sample
          ) - cum_weights[i].data_ptr<double>();

      result[i][j] = sample_idx;

      // 4. 无放回采样：将已采样的权重置零
      if (!replacement) {
        cum_weights[i][sample_idx] = 0.0;
        // 重新计算累积概率
        cum_weights[i] = cum_weights[i].cumsum(0);
      }
    }
  }
}
```

**算法步骤**：
1. 计算累积概率：`cum[i] = sum(weights[0:i+1])`
2. 生成均匀随机数：`u ~ Uniform(0, total_weight)`
3. 二分搜索：找到 `cum[i-1] < u <= cum[i]` 的索引 `i`
4. 返回 `i` 作为采样结果

**图示**（n=4 的情况）：
```
weights:     [1.0,  2.0,  3.0,  4.0]
cum_weights: [1.0,  3.0,  6.0,  10.0]
                ↓     ↓     ↓     ↓
区间:       [0,1) [1,3) [3,6) [6,10)
概率:        10%   20%   30%   40%

生成 u=5.2 → 二分搜索 → cum[2]=6.0 → 返回索引 2
```

### 1.5 CUDA 实现

#### 1.5.1 CUDA Kernel 入口

**文件**: `aten/src/ATen/native/cuda/MultinomialKernel.cu:120`

```cpp
void multinomial_kernel_cuda(
    Tensor& result,
    const Tensor& self,
    int64_t num_samples,
    bool replacement,
    Generator gen) {

  auto n_categories = self.size(-1);
  auto n_dist = self.dim() == 1 ? 1 : self.size(0);

  // 归一化权重（确保非负且和为 1）
  Tensor norm_weights = self / self.sum(-1, true);

  // 获取或创建生成器
  auto gen_impl = get_generator_or_default<CUDAGeneratorImpl>(
      gen, cuda::detail::getDefaultCUDAGenerator());

  // 计算 Grid/Block 配置
  int64_t numel = n_dist * num_samples;
  auto counter_offset = calc_execution_policy(numel);
  auto rng_state = gen_impl->philox_cuda_state(counter_offset);

  // 有放回采样：并行采样
  if (replacement) {
    multinomial_with_replacement_kernel<<<grid, block>>>(
        result.data_ptr<int64_t>(),
        norm_weights.data_ptr<float>(),
        n_dist,
        n_categories,
        num_samples,
        rng_state
    );
  }
  // 无放回采样：Gumbel-Max Trick
  else {
    multinomial_without_replacement_kernel<<<grid, block>>>(
        result.data_ptr<int64_t>(),
        norm_weights.data_ptr<float>(),
        n_dist,
        n_categories,
        num_samples,
        rng_state
    );
  }
}
```

#### 1.5.2 有放回采样 Kernel

**文件**: `aten/src/ATen/native/cuda/MultinomialKernel.cu:45`

```cpp
__global__ void multinomial_with_replacement_kernel(
    int64_t* result,          // 输出: [n_dist, num_samples]
    const float* weights,     // 输入: [n_dist, n_categories]
    int64_t n_dist,
    int64_t n_categories,
    int64_t num_samples,
    PhiloxCudaState rng_state) {

  // 每个线程负责一个分布的一个样本
  int64_t dist_idx = blockIdx.x;
  int64_t sample_idx = threadIdx.x;

  if (dist_idx >= n_dist || sample_idx >= num_samples) return;

  // 初始化 Philox RNG
  curandStatePhilox4_32_10_t state;
  curand_init(rng_state.seed, dist_idx * num_samples + sample_idx,
              rng_state.offset, &state);

  // 1. 生成 uniform(0, 1)
  float u = curand_uniform(&state);

  // 2. 计算累积概率并搜索
  const float* dist_weights = weights + dist_idx * n_categories;
  float cumulative = 0.0f;
  int64_t category = n_categories - 1;  // 默认最后一个

  for (int64_t i = 0; i < n_categories; i++) {
    cumulative += dist_weights[i];
    if (u <= cumulative) {
      category = i;
      break;
    }
  }

  // 3. 写入结果
  result[dist_idx * num_samples + sample_idx] = category;
}
```

#### 1.5.3 无放回采样：Gumbel-Max Trick

**原理**：要从分布 $P$ 中无放回采样 $k$ 个样本，等价于：
1. 对每个类别 $i$，计算 $g_i = \log(p_i) + \text{Gumbel}(0, 1)$
2. 选择 $g_i$ 最大的 $k$ 个类别

其中 Gumbel 分布：$G \sim -\log(-\log(U))$，$U \sim \text{Uniform}(0,1)$

**文件**: `aten/src/ATen/native/cuda/MultinomialKernel.cu:75`

```cpp
__global__ void multinomial_without_replacement_kernel(
    int64_t* result,
    const float* weights,
    int64_t n_dist,
    int64_t n_categories,
    int64_t num_samples,
    PhiloxCudaState rng_state) {

  int64_t dist_idx = blockIdx.x;
  if (dist_idx >= n_dist) return;

  // 初始化 RNG
  curandStatePhilox4_32_10_t state;
  curand_init(rng_state.seed, dist_idx, rng_state.offset, &state);

  // 1. 为每个类别生成 Gumbel 噪声
  extern __shared__ float gumbel_values[];  // [n_categories]
  const float* dist_weights = weights + dist_idx * n_categories;

  for (int64_t i = threadIdx.x; i < n_categories; i += blockDim.x) {
    float u = curand_uniform(&state);
    u = fmaxf(u, 1e-10f);  // 避免 log(0)

    // Gumbel(0,1) = -log(-log(U))
    float gumbel = -logf(-logf(u));

    // log(p_i) + Gumbel
    gumbel_values[i] = logf(dist_weights[i] + 1e-10f) + gumbel;
  }
  __syncthreads();

  // 2. 选择最大的 num_samples 个（部分排序）
  // 使用 shared memory 的 selection algorithm
  for (int64_t k = 0; k < num_samples; k++) {
    if (threadIdx.x == 0) {
      // 找到当前最大值的索引
      int64_t max_idx = 0;
      float max_val = gumbel_values[0];
      for (int64_t i = 1; i < n_categories; i++) {
        if (gumbel_values[i] > max_val) {
          max_val = gumbel_values[i];
          max_idx = i;
        }
      }

      // 记录结果
      result[dist_idx * num_samples + k] = max_idx;

      // 标记为已选择（设为负无穷）
      gumbel_values[max_idx] = -INFINITY;
    }
    __syncthreads();
  }
}
```

### 1.6 完整调用流程图

```
torch.multinomial(weights, num_samples=5, replacement=True)
    ↓
torch._C.multinomial
    ↓
at::native::multinomial(self, num_samples, replacement, gen)
    ↓
【参数检查】
├─ 检查维度 (1D 或 2D)
├─ 检查 num_samples > 0
└─ 无放回时检查 num_samples <= n_categories
    ↓
【创建输出张量】
result = at::empty([n_dist, num_samples], dtype=Long, device=self.device)
    ↓
【设备分派】
    ├─ CPU → multinomial_kernel_impl
    │   ↓
    │   1. 计算累积概率: cum_weights = self.cumsum(-1)
    │   2. 循环 num_samples 次:
    │      - 生成 u ~ Uniform(0, total)
    │      - 二分搜索找到类别
    │      - 写入 result
    │   3. 无放回时更新累积概率
    │
    └─ CUDA → multinomial_kernel_cuda
        ↓
        【获取 Generator】
        gen_impl = get_generator_or_default<CUDAGeneratorImpl>(gen)
        ↓
        【归一化权重】
        norm_weights = self / self.sum(-1, keepdim=True)
        ↓
        【计算 RNG 状态】
        rng_state = gen_impl->philox_cuda_state(counter_offset)
        ↓
        【选择算法】
        ├─ replacement=True → multinomial_with_replacement_kernel
        │   ↓
        │   每个线程负责一个样本:
        │   1. 初始化 Philox RNG
        │   2. 生成 u ~ Uniform(0, 1)
        │   3. 线性搜索累积概率
        │   4. 写入结果
        │
        └─ replacement=False → multinomial_without_replacement_kernel
            ↓
            每个 block 负责一个分布:
            1. 为每个类别生成 Gumbel 噪声
            2. 计算 g_i = log(p_i) + Gumbel
            3. 选择最大的 k 个索引
            4. 写入结果
    ↓
【返回结果】
return result  // [n_dist, num_samples]
```

---

## 2. Poisson 分布

### 2.1 概述

Poisson 分布用于建模在固定时间/空间内发生的事件次数，参数 $\lambda$ 表示平均事件数。

**概率质量函数**：
$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, ...$$

**应用场景**：
- 单位时间内的顾客到达数
- 网页单位时间内的访问次数
- 放射性衰变计数
- 神经元发放次数

### 2.2 Python 层入口

**API 签名**：
```python
torch.poisson(
    input: Tensor,           # lambda 参数 (每个元素的平均值)
    generator: Optional[Generator] = None
) -> Tensor
```

**使用示例**：
```python
# 单个 lambda
lambda_rate = torch.tensor(5.0)
counts = torch.poisson(lambda_rate)  # 输出: 可能是 3, 4, 5, 6, 7...

# 批量 lambda
lambda_rates = torch.tensor([1.0, 5.0, 10.0, 50.0])
counts = torch.poisson(lambda_rates)
# 输出: tensor([2, 4, 12, 48])  (示例)

# CUDA
lambda_rates = torch.rand(1000, device='cuda') * 20
counts = torch.poisson(lambda_rates)
```

### 2.3 核心实现层

#### 2.3.1 算法选择

Poisson 分布的采样算法根据 $\lambda$ 的大小选择：

| $\lambda$ 范围 | 算法 | 时间复杂度 | 原理 |
|---------------|------|-----------|------|
| $\lambda < 10$ | **Knuth 算法** | O(λ) 期望 | 等待时间 |
| $\lambda \geq 10$ | **变换拒绝采样 (PTRS)** | O(1) 期望 | 拒绝采样 + 正态近似 |

#### 2.3.2 Knuth 算法（小 λ）

**原理**：Poisson 过程中，事件间隔服从指数分布。累积等待时间超过 1 时，已发生的事件数服从 Poisson(λ)。

**伪代码**：
```python
def poisson_knuth(lambda_rate):
    L = exp(-lambda_rate)  # 阈值
    k = 0                  # 事件计数
    p = 1.0                # 累积概率

    while p > L:
        k += 1
        p *= uniform(0, 1)  # 等待时间

    return k - 1
```

**文件**: `aten/src/ATen/native/cpu/DistributionKernels.cpp:180`

```cpp
template <typename scalar_t>
scalar_t poisson_knuth(scalar_t lambda, Generator gen) {
  scalar_t L = std::exp(-lambda);
  scalar_t k = 0;
  scalar_t p = 1.0;

  do {
    k++;
    p *= uniform_real_distribution<scalar_t>(0.0, 1.0)(gen);
  } while (p > L);

  return k - 1;
}
```

#### 2.3.3 PTRS 算法（大 λ）

**原理**：当 $\lambda$ 较大时，Poisson 分布近似正态分布 $N(\lambda, \lambda)$。使用变换拒绝采样。

**文件**: `aten/src/ATen/native/cuda/DistributionPoissonKernel.cu:25`

```cpp
__device__ float poisson_ptrs(float lambda, curandStatePhilox4_32_10_t* state) {
  // PTRS (Poisson Transformed Rejection Sampling)
  float c = 0.767f - 3.36f / lambda;
  float beta = float(M_PI) / sqrtf(3.0f * lambda);
  float alpha = beta * lambda;
  float k = logf(c) - lambda - logf(beta);

  for (;;) {
    float u = curand_uniform(state);
    u = u - 0.5f;
    float x = alpha - logf((1.0f - u) / u) / beta;
    float n = floorf(x + 0.5f);
    if (n < 0) continue;

    float v = curand_uniform(state);
    float y = alpha - beta * x;
    float lhs = y + logf(v / (1.0f + expf(y)));
    float rhs = k + n * logf(lambda) - lgammaf(n + 1.0f);

    if (lhs <= rhs) return n;
  }
}
```

### 2.4 CUDA Kernel 实现

**文件**: `aten/src/ATen/native/cuda/DistributionPoissonKernel.cu:60`

```cpp
__global__ void poisson_kernel(
    int64_t* output,
    const float* lambda,
    int64_t numel,
    PhiloxCudaState rng_state) {

  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) return;

  // 初始化 Philox
  curandStatePhilox4_32_10_t state;
  curand_init(rng_state.seed, idx, rng_state.offset, &state);

  float lambda_val = lambda[idx];

  // 算法选择
  int64_t result;
  if (lambda_val < 10.0f) {
    result = poisson_knuth(lambda_val, &state);
  } else {
    result = poisson_ptrs(lambda_val, &state);
  }

  output[idx] = result;
}
```

### 2.5 完整调用流程图

```
torch.poisson(lambda_rates)
    ↓
torch._C.poisson
    ↓
at::native::poisson(self, gen)
    ↓
【创建输出张量】
result = at::empty_like(self, dtype=Long)
    ↓
【设备分派】
    ├─ CPU → poisson_kernel_cpu
    │   ↓
    │   for each element:
    │       if lambda < 10:
    │           use Knuth algorithm
    │       else:
    │           use PTRS algorithm
    │
    └─ CUDA → poisson_kernel
        ↓
        【获取 RNG 状态】
        rng_state = gen->philox_cuda_state(counter_offset)
        ↓
        【启动 kernel】
        poisson_kernel<<<grid, block>>>(output, lambda, numel, rng_state)
        ↓
        【设备端执行】
        每个线程:
        1. 初始化 Philox
        2. 读取 lambda[idx]
        3. 选择算法:
           ├─ lambda < 10 → Knuth
           └─ lambda >= 10 → PTRS
        4. 写入 output[idx]
    ↓
【返回结果】
return result
```

---

## 3. Binomial 分布

### 3.1 概述

Binomial 分布描述 $n$ 次独立伯努利试验中成功的次数，参数为试验次数 $n$ 和成功概率 $p$。

**概率质量函数**：
$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, ..., n$$

**应用场景**：
- 抛硬币正面次数
- 产品合格数
- A/B 测试点击数

### 3.2 Python 层入口

**API 签名**：
```python
torch.binomial(
    count: Tensor,      # n: 试验次数
    prob: Tensor,       # p: 成功概率
    generator: Optional[Generator] = None
) -> Tensor
```

**使用示例**：
```python
# 10 次抛硬币
n = torch.tensor(10.0)
p = torch.tensor(0.5)
successes = torch.binomial(n, p)
# 输出: 可能是 4, 5, 6...

# 批量试验
counts = torch.tensor([10.0, 20.0, 100.0])
probs = torch.tensor([0.3, 0.5, 0.7])
results = torch.binomial(counts, probs)
```

### 3.3 算法选择

| 条件 | 算法 | 时间复杂度 |
|------|------|-----------|
| $n \cdot p < 10$ 或 $n \cdot (1-p) < 10$ | **二项式反演法** | O(np) 期望 |
| 其他 | **BTRS 算法** | O(1) 期望 |

### 3.4 BTRS 算法

**BTRS** (Binomial Triangle Rejection Sampling) 是一种高效的拒绝采样算法，特别适合大 $n$ 和中等 $p$ 的情况。

**文件**: `aten/src/ATen/native/cuda/DistributionBinomialKernel.cu:30`

```cpp
__device__ int64_t binomial_btrs(
    float n, float p,
    curandStatePhilox4_32_10_t* state) {

  // BTRS 参数
  float np = n * p;
  float npq = np * (1.0f - p);
  float f_m = np + p;
  int64_t m = (int64_t)f_m;

  float p1 = floorf(2.195f * sqrtf(npq) - 4.6f * p) + 0.5f;
  float xm = (float)m + 0.5f;
  float xl = xm - p1;
  float xr = xm + p1;

  float c = 0.134f + 20.5f / (15.3f + (float)m);
  float lambda_l = (xm - xl) * (1.0f + 0.5f * c);
  float lambda_r = (xr - xm) * (1.0f + 0.5f * c);
  float p2 = p1 * (1.0f + 2.0f * c);
  float p3 = p2 + c / lambda_l;
  float p4 = p3 + c / lambda_r;

  for (;;) {
    float u = curand_uniform(state) * p4;
    float v = curand_uniform(state);

    float x;
    if (u <= p1) {
      // 中心区域（矩形）
      x = xm - p1 * v + u;
    } else if (u <= p2) {
      // 左尾
      float y = logf(v * lambda_l / p1);
      x = xl + y;
      if (x < 0.0f) continue;
    } else if (u <= p3) {
      // 右尾
      float y = logf(v * lambda_r / p1);
      x = xr - y;
      if (x > n) continue;
    } else {
      // 远端区域
      x = (u - p3) * lambda_r + xr;
      if (x > n) continue;
    }

    int64_t k = (int64_t)floorf(x);

    // 拒绝检验
    float rho = // ... 复杂的接受概率计算
    if (v <= rho) return k;
  }
}
```

---

## 4. Gamma 分布

### 4.1 概述

Gamma 分布是连续概率分布，由形状参数 $\alpha$ (shape) 和尺度参数 $\beta$ (scale) 定义。

**概率密度函数**：
$$f(x; \alpha, \beta) = \frac{1}{\beta^\alpha \Gamma(\alpha)} x^{\alpha-1} e^{-x/\beta}, \quad x > 0$$

**应用场景**：
- 贝叶斯统计中的共轭先验
- 排队论中的服务时间
- 降雨量建模
- Dirichlet 分布的基础

### 4.2 Python 层入口

**API 签名**：
```python
torch._standard_gamma(
    alpha: Tensor,      # shape 参数
    generator: Optional[Generator] = None
) -> Tensor
```

**注意**：PyTorch 只提供标准 Gamma（scale=1），其他 scale 通过变换得到：
```python
# 生成 Gamma(alpha, beta)
alpha = torch.tensor(2.0)
beta = torch.tensor(3.0)

# 标准 Gamma(alpha, 1)
x = torch._standard_gamma(alpha)

# 缩放到 Gamma(alpha, beta)
x = x * beta
```

### 4.3 Marsaglia-Tsang 算法

**原理**：Marsaglia-Tsang (2000) 是目前最快的 Gamma 采样算法，使用拒绝采样，接受率 > 95%。

**算法步骤**（$\alpha \geq 1$）：
1. 令 $d = \alpha - 1/3$，$c = 1/\sqrt{9d}$
2. 循环：
   - 生成 $Z \sim N(0,1)$
   - 令 $V = (1 + cZ)^3$
   - 如果 $V > 0$ 且 $U < 1 - 0.0331Z^4$ 或 $\log(U) < 0.5Z^2 + d(1 - V + \log(V))$
   - 则接受 $X = dV$

**文件**: `aten/src/ATen/native/cuda/DistributionGammaKernel.cu:20`

```cpp
__device__ float standard_gamma_marsaglia(
    float alpha,
    curandStatePhilox4_32_10_t* state) {

  // Marsaglia-Tsang 算法
  if (alpha < 1.0f) {
    // alpha < 1 时，使用变换: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
    float x = standard_gamma_marsaglia(alpha + 1.0f, state);
    float u = curand_uniform(state);
    return x * powf(u, 1.0f / alpha);
  }

  float d = alpha - 1.0f / 3.0f;
  float c = 1.0f / sqrtf(9.0f * d);

  for (;;) {
    float z, v;
    do {
      z = curand_normal(state);  // 标准正态
      v = 1.0f + c * z;
    } while (v <= 0.0f);

    v = v * v * v;  // V = (1 + cZ)^3
    float u = curand_uniform(state);

    // 快速接受检验
    float z2 = z * z;
    if (u < 1.0f - 0.0331f * z2 * z2) {
      return d * v;
    }

    // 精确接受检验
    if (logf(u) < 0.5f * z2 + d * (1.0f - v + logf(v))) {
      return d * v;
    }
  }
}
```

### 4.4 CUDA Kernel 实现

**文件**: `aten/src/ATen/native/cuda/DistributionGammaKernel.cu:55`

```cpp
__global__ void standard_gamma_kernel(
    float* output,
    const float* alpha,
    int64_t numel,
    PhiloxCudaState rng_state) {

  int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= numel) return;

  curandStatePhilox4_32_10_t state;
  curand_init(rng_state.seed, idx, rng_state.offset, &state);

  float alpha_val = alpha[idx];
  float result = standard_gamma_marsaglia(alpha_val, &state);

  output[idx] = result;
}
```

### 4.5 完整调用流程图

```
torch._standard_gamma(alpha)
    ↓
torch._C._standard_gamma
    ↓
at::native::_standard_gamma(self, gen)
    ↓
【创建输出张量】
result = at::empty_like(self)
    ↓
【设备分派】
    ├─ CPU → standard_gamma_kernel_cpu
    │   ↓
    │   for each element:
    │       use Marsaglia-Tsang algorithm
    │
    └─ CUDA → standard_gamma_kernel
        ↓
        【获取 RNG 状态】
        rng_state = gen->philox_cuda_state(counter_offset)
        ↓
        【启动 kernel】
        standard_gamma_kernel<<<grid, block>>>(output, alpha, numel, rng_state)
        ↓
        【设备端执行】
        每个线程:
        1. 初始化 Philox
        2. 读取 alpha[idx]
        3. 执行 Marsaglia-Tsang:
           ├─ alpha < 1 → 递归调用 + 变换
           └─ alpha >= 1 → 拒绝采样
        4. 写入 output[idx]
    ↓
【返回结果】
return result
```

---

## 5. Dirichlet 分布

### 5.1 概述

Dirichlet 分布是多元 Beta 分布的推广，用于生成满足 $\sum x_i = 1$ 的概率向量。

**概率密度函数**：
$$f(x_1, ..., x_k; \alpha_1, ..., \alpha_k) = \frac{\Gamma(\sum \alpha_i)}{\prod \Gamma(\alpha_i)} \prod x_i^{\alpha_i - 1}$$

约束：$x_i > 0, \sum x_i = 1$

**应用场景**：
- 主题模型（LDA）
- 贝叶斯统计的先验分布
- 混合模型参数

### 5.2 实现原理

**Dirichlet 采样 = Gamma 归一化**：

要从 $\text{Dirichlet}(\alpha_1, ..., \alpha_k)$ 采样：
1. 对每个 $i$，采样 $Y_i \sim \text{Gamma}(\alpha_i, 1)$
2. 归一化：$X_i = \frac{Y_i}{\sum Y_j}$
3. 则 $(X_1, ..., X_k) \sim \text{Dirichlet}(\alpha_1, ..., \alpha_k)$

**PyTorch 实现**（纯 Python）：
```python
def dirichlet(alpha):
    """
    Args:
        alpha: Tensor of shape [..., k]

    Returns:
        samples: Tensor of shape [..., k], sum(samples, dim=-1) == 1
    """
    # 1. 采样 Gamma
    gamma_samples = torch._standard_gamma(alpha)

    # 2. 归一化
    dirichlet_samples = gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)

    return dirichlet_samples

# 使用示例
alpha = torch.tensor([1.0, 2.0, 3.0])
samples = dirichlet(alpha)
# 输出: tensor([0.1523, 0.3421, 0.5056]), sum = 1.0
```

**文件**: `torch/distributions/dirichlet.py:45`

```python
class Dirichlet(Distribution):
    def __init__(self, concentration, validate_args=None):
        self.concentration = concentration
        super().__init__(validate_args=validate_args)

    def rsample(self, sample_shape=torch.Size()):
        # 使用 Gamma 归一化方法
        shape = self._extended_shape(sample_shape)
        gamma_samples = torch._standard_gamma(
            self.concentration.expand(shape)
        )
        # 归一化
        return gamma_samples / gamma_samples.sum(-1, keepdim=True)
```

### 5.3 调用流程图

```
torch.distributions.Dirichlet(alpha).sample()
    ↓
Dirichlet.rsample(sample_shape)
    ↓
【采样 Gamma】
gamma_samples = torch._standard_gamma(alpha)
    ↓ (详见第 4 节 Gamma 流程)
    ├─ 对每个 alpha[i]:
    │      使用 Marsaglia-Tsang 算法
    │      生成 Gamma(alpha[i], 1)
    │
【归一化】
dirichlet_samples = gamma_samples / gamma_samples.sum(dim=-1, keepdim=True)
    ↓
【返回结果】
return dirichlet_samples  // sum = 1.0
```

---

## 6. 实现对比总结

### 6.1 算法复杂度对比

| 分布 | 小参数算法 | 大参数算法 | 期望复杂度 | CUDA 优化 |
|------|----------|----------|-----------|---------|
| **Multinomial** | 累积概率 + 二分搜索 | Alias Method | O(k log n) / O(k) | 并行累积概率 |
| **Poisson** | Knuth (λ<10) | PTRS (λ≥10) | O(λ) / O(1) | 条件分支 |
| **Binomial** | 二项式反演 | BTRS | O(np) / O(1) | 拒绝采样 |
| **Gamma** | 递归 + 变换 (α<1) | Marsaglia-Tsang | O(1) | 拒绝采样 |
| **Dirichlet** | N/A | Gamma + 归一化 | O(k) | 向量化 |

### 6.2 实现特点对比

| 特性 | Multinomial | Poisson | Binomial | Gamma |
|------|------------|---------|----------|-------|
| **是否需要循环** | 是（搜索） | 是（Knuth） | 是（BTRS） | 是（拒绝采样） |
| **接受率** | 100% | 100% (Knuth) / ~95% (PTRS) | ~95% | >95% |
| **向量化友好** | 中等 | 高 | 中等 | 高 |
| **CUDA 加速比** | 5-10x | 10-50x | 10-30x | 20-100x |
| **内存需求** | O(n) 累积概率 | O(1) | O(1) | O(1) |

### 6.3 应用场景总结

```python
# 1. Multinomial - 分类采样
logits = model(input)
probs = F.softmax(logits, dim=-1)
action = torch.multinomial(probs, num_samples=1)

# 2. Poisson - 计数事件
lambda_rates = torch.tensor([2.0, 5.0, 10.0])
event_counts = torch.poisson(lambda_rates)

# 3. Binomial - 伯努利试验
n_trials = torch.tensor(100.0)
success_prob = torch.tensor(0.3)
successes = torch.binomial(n_trials, success_prob)

# 4. Gamma - 贝叶斯统计
alpha = torch.tensor([2.0, 3.0, 5.0])
samples = torch._standard_gamma(alpha)

# 5. Dirichlet - 概率向量
alpha = torch.tensor([1.0, 2.0, 3.0, 4.0])
prob_vector = dirichlet(alpha)  # sum = 1.0
```

---

## 7. 性能优化建议

### 7.1 算法选择优化

```python
# ❌ 差: 用多个 Bernoulli 实现 Binomial
def binomial_slow(n, p):
    return (torch.rand(int(n)) < p).sum()

# ✅ 好: 直接使用 Binomial
result = torch.binomial(torch.tensor(n), torch.tensor(p))
```

### 7.2 批量化处理

```python
# ❌ 差: 循环调用
results = []
for lambda_val in lambda_rates:
    results.append(torch.poisson(lambda_val))

# ✅ 好: 批量处理
results = torch.poisson(lambda_rates)  # 一次 CUDA kernel
```

### 7.3 CUDA 性能

```python
# ✅ 好: 在 GPU 上生成
lambda_rates = torch.rand(10000, device='cuda') * 10
counts = torch.poisson(lambda_rates)  # 全程在 GPU

# ❌ 差: CPU-GPU 传输
lambda_rates = torch.rand(10000) * 10  # CPU
counts = torch.poisson(lambda_rates.cuda())  # 需要传输
```

---

## Appendix A: Alias Method 详解

### A.1 算法原理

Alias Method 是一种高效的离散分布采样算法，实现 **O(n) 预处理 + O(1) 采样**。

**核心思想**：将任意离散分布转换为均匀分布的混合。

**数据结构**：
- `prob[i]`: 索引 `i` 处的概率
- `alias[i]`: 索引 `i` 处的别名（另一个索引）

**采样过程**：
1. 均匀采样 `i ~ Uniform(0, n-1)`
2. 均匀采样 `u ~ Uniform(0, 1)`
3. 如果 `u < prob[i]`，返回 `i`；否则返回 `alias[i]`

### A.2 预处理算法

**目标**：将概率分布 $\{p_1, p_2, ..., p_n\}$ 转换为 Alias 表。

```python
def create_alias_table(probs):
    """
    Args:
        probs: [n] 概率分布（已归一化）

    Returns:
        prob_table: [n] 每个格子保留原索引的概率
        alias_table: [n] 别名索引
    """
    n = len(probs)
    prob_table = torch.zeros(n)
    alias_table = torch.zeros(n, dtype=torch.long)

    # 1. 缩放概率到 [0, n]
    scaled_probs = probs * n

    # 2. 分类为小于1和大于1的
    small = []
    large = []
    for i, p in enumerate(scaled_probs):
        if p < 1.0:
            small.append(i)
        else:
            large.append(i)

    # 3. 构建表
    while small and large:
        s = small.pop()
        l = large.pop()

        prob_table[s] = scaled_probs[s]
        alias_table[s] = l

        # 更新 large 索引的剩余概率
        scaled_probs[l] = scaled_probs[l] + scaled_probs[s] - 1.0

        if scaled_probs[l] < 1.0:
            small.append(l)
        else:
            large.append(l)

    # 4. 处理剩余（由于浮点误差）
    while large:
        prob_table[large.pop()] = 1.0
    while small:
        prob_table[small.pop()] = 1.0

    return prob_table, alias_table

# 采样
def alias_sample(prob_table, alias_table):
    n = len(prob_table)
    i = random.randint(0, n-1)
    u = random.random()
    return i if u < prob_table[i] else alias_table[i]
```

### A.3 图示示例

**原始概率**：`[0.1, 0.2, 0.3, 0.4]`

**缩放到 4x**：`[0.4, 0.8, 1.2, 1.6]`

**构建 Alias 表**：

```
索引   缩放概率   prob_table   alias_table   图示
 0      0.4        0.4            2          [0: 40%, 2: 60%]
 1      0.8        0.8            3          [1: 80%, 3: 20%]
 2      1.2        1.0            -          [2: 100%]
 3      1.6        1.0            -          [3: 100%]
```

**采样过程**（示例）：
- 均匀选择索引：`i = 0`
- 均匀随机数：`u = 0.6`
- 因为 `u = 0.6 > prob_table[0] = 0.4`
- 返回 `alias_table[0] = 2`

### A.4 时间复杂度分析

- **预处理**：O(n) 线性时间构建表
- **采样**：O(1) 常数时间
- **空间**：O(n) 存储两个表

**适用场景**：
- 同一分布需要多次采样（摊销预处理成本）
- PyTorch 中未直接使用（因为每次采样的分布可能不同）
- 在某些静态场景下（如 word2vec negative sampling）非常高效

---

## Appendix B: BTRS 算法详解

### B.1 算法原理

**BTRS** (Binomial Triangle Rejection Sampling) 是 Kachitvichyanukul 和 Schmeiser (1988) 提出的高效二项式采样算法。

**核心思想**：
1. 将二项式分布近似为三角形分布
2. 使用拒绝采样，从容易采样的包络分布中生成候选
3. 通过接受-拒绝检验保证正确性

### B.2 算法步骤

**输入**：$n$ (试验次数), $p$ (成功概率)

**预处理**（每次调用计算）：
```cpp
float np = n * p;
float npq = np * (1 - p);  // 方差
float m = floor(np + p);    // 众数
float xm = m + 0.5;         // 众数中心
```

**四区域采样**：

1. **中心矩形区域** (概率 $p_1$)：
   - 直接均匀采样 $x \in [x_m - p_1, x_m + p_1]$
   - 接受率 100%

2. **左三角区域** (概率 $p_2 - p_1$)：
   - 使用指数分布 $y = \log(v \cdot \lambda_l)$
   - $x = x_l + y$

3. **右三角区域** (概率 $p_3 - p_2$)：
   - 对称的右侧三角

4. **远端尾部** (概率 $p_4 - p_3$)：
   - 直接线性采样

### B.3 接受-拒绝检验

生成候选 $k$ 后，计算接受概率：

$$\rho = \frac{P(X = k)}{g(k)} = \frac{\binom{n}{k} p^k (1-p)^{n-k}}{\text{envelope}(k)}$$

实际实现中使用对数形式避免溢出：

```cpp
float log_rho =
    lgamma(n + 1) - lgamma(k + 1) - lgamma(n - k + 1)  // log(C(n,k))
    + k * log(p) + (n - k) * log(1 - p)                // log(p^k (1-p)^(n-k))
    - log_envelope;                                     // 包络的对数

if (log(v) <= log_rho) {
    return k;  // 接受
}
```

### B.4 为什么 BTRS 高效？

1. **高接受率**：精心设计的包络函数使接受率 > 95%
2. **避免对数运算**：大部分样本在中心矩形，无需计算 $\log(\binom{n}{k})$
3. **缓存友好**：算法局部性好，减少分支预测失败

---

## Appendix C: Marsaglia-Tsang 算法详解

### C.1 算法原理

Marsaglia 和 Tsang (2000) 提出了目前最快的 Gamma 分布采样算法，基于拒绝采样和变换。

**核心思想**：
- 将 $\text{Gamma}(\alpha, 1)$ 的采样转换为从容易采样的分布（正态分布）生成候选
- 通过精心设计的接受条件确保高接受率

### C.2 算法推导

**目标**：采样 $X \sim \text{Gamma}(\alpha, 1)$，$\alpha \geq 1$

**变换**：
1. 令 $d = \alpha - 1/3$，$c = 1/\sqrt{9d}$
2. 生成 $Z \sim N(0, 1)$
3. 令 $V = (1 + cZ)^3$
4. 如果 $V > 0$ 且满足接受条件，返回 $X = dV$

**接受条件**（两阶段）：

**快速检验**（无需对数）：
$$U < 1 - 0.0331 Z^4$$

**精确检验**（需要对数）：
$$\log(U) < \frac{1}{2} Z^2 + d(1 - V + \log(V))$$

### C.3 为什么这样设计？

**问题**：直接从 Gamma 分布采样很难。

**解决**：
1. **选择包络分布**：使用变换的正态分布 $(1 + cZ)^3$ 近似 Gamma
2. **优化参数**：$d$ 和 $c$ 的选择使得包络尽可能贴近目标分布
3. **两阶段检验**：先用快速检验过滤大部分样本，只有少数需要精确计算

### C.4 接受率分析

**理论接受率**：
- $\alpha = 1$: 接受率约 96%
- $\alpha = 5$: 接受率约 99%
- $\alpha$ 越大，接受率越高

**实际性能**：
- 平均每个样本需要 1.05 次循环
- 比传统方法（如 Best's 算法）快 2-3 倍

### C.5 处理 α < 1 的情况

当 $\alpha < 1$ 时，使用变换性质：

$$\text{Gamma}(\alpha, 1) = \text{Gamma}(\alpha + 1, 1) \cdot U^{1/\alpha}$$

其中 $U \sim \text{Uniform}(0, 1)$

**实现**：
```cpp
if (alpha < 1.0) {
    float x = standard_gamma_marsaglia(alpha + 1.0, state);
    float u = curand_uniform(state);
    return x * powf(u, 1.0 / alpha);
}
```

### C.6 完整伪代码

```python
def marsaglia_tsang_gamma(alpha):
    """Marsaglia-Tsang 算法采样 Gamma(alpha, 1)"""
    if alpha < 1:
        # 使用变换
        x = marsaglia_tsang_gamma(alpha + 1)
        u = random.random()
        return x * (u ** (1.0 / alpha))

    d = alpha - 1.0 / 3.0
    c = 1.0 / math.sqrt(9.0 * d)

    while True:
        # 生成候选
        while True:
            z = random.gauss(0, 1)
            v = 1.0 + c * z
            if v > 0:
                break

        v = v * v * v
        u = random.random()

        # 快速接受检验
        if u < 1.0 - 0.0331 * z * z * z * z:
            return d * v

        # 精确接受检验
        if math.log(u) < 0.5 * z * z + d * (1.0 - v + math.log(v)):
            return d * v
```

---

## 参考文献

### 论文

1. **Alias Method**:
   - Walker, A. J. (1977). "An Efficient Method for Generating Discrete Random Variables with General Distributions"

2. **BTRS 算法**:
   - Kachitvichyanukul, V., & Schmeiser, B. W. (1988). "Binomial Random Variate Generation"

3. **Marsaglia-Tsang 算法**:
   - Marsaglia, G., & Tsang, W. W. (2000). "A Simple Method for Generating Gamma Variables"

4. **Poisson PTRS**:
   - Ahrens, J. H., & Dieter, U. (1982). "Computer Generation of Poisson Deviates"

### PyTorch 源码

- CPU 实现: `aten/src/ATen/native/cpu/DistributionKernels.cpp`
- CUDA 实现: `aten/src/ATen/native/cuda/Distribution*.cu`
- Python API: `torch/distributions/`

### 外部资源

- [NumPy Random Sampling](https://numpy.org/doc/stable/reference/random/index.html)
- [Boost Random Library](https://www.boost.org/doc/libs/release/libs/random/)
- [GSL Random Number Distributions](https://www.gnu.org/software/gsl/doc/html/randist.html)

---

**文档版本**: v1.0
**对应 PyTorch 版本**: >= 2.0
**最后更新**: 2024-12-22
**维护者**: PyTorch Documentation Team

---

## 反馈与贡献

如有疑问或发现错误，欢迎：
- 提交 Issue: https://github.com/pytorch/pytorch/issues
- 提交 PR 改进文档
- 在 PyTorch 论坛讨论: https://discuss.pytorch.org/
