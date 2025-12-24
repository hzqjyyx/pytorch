# PyTorch 随机数生成系统架构总览

## 文档导航

欢迎来到 PyTorch 随机数生成系统文档！本文档提供整体架构概览和 API 导航。

### 已有文档
- **[torch.rand/uniform 详解](uniform_call_flow.md)** - 均匀分布随机数生成的完整调用流程（CUDA 重点）
- **[torch.randn/normal 详解](normal_call_flow.md)** - 正态分布随机数生成的完整调用流程（CUDA 重点）
- **[torch.randint/random 详解](discrete_call_flow.md)** - 离散整数随机数生成的完整调用流程（CUDA 重点）
- **[复杂采样函数详解](sampling.md)** - multinomial, poisson, binomial, gamma, dirichlet 采样算法
- **[特殊分布详解](special_distributions.md)** - exponential, cauchy, geometric, log_normal, bernoulli 等基于变换的分布

---

## 概述

PyTorch 的随机数生成系统是深度学习训练中至关重要的组件，它提供了高性能、可复现的随机数生成能力。本文档将带你了解：

- PyTorch 提供的所有随机数 API 及其分类
- 整体架构设计
- 不同设备（CPU/CUDA）的实现差异
- Generator 机制和状态管理
- 如何选择合适的 API

---

## PyTorch 随机数 API 分类

### 1. 工厂函数（Factory Functions）- 创建新张量

这些函数直接创建包含随机数的新张量，是最常用的 API。

#### 1.1 均匀分布
| API | 说明 | 输出范围 | 详细文档 |
|-----|------|----------|----------|
| `torch.rand(size, ...)` | 生成均匀分布随机数 | [0, 1) | [详见](uniform_call_flow.md) |
| `torch.rand_like(input, ...)` | 按输入形状生成 | [0, 1) | [详见](uniform_call_flow.md) |

**使用示例**：
```python
# 生成 3x4 的均匀分布张量
x = torch.rand(3, 4)

# 生成与 y 相同形状的张量
z = torch.rand_like(y)

# 使用自定义生成器
gen = torch.Generator().manual_seed(42)
x = torch.rand(3, 4, generator=gen)

# CUDA 张量
x = torch.rand(3, 4, device='cuda')
```

#### 1.2 正态分布
| API | 说明 | 分布参数 | 详细文档 |
|-----|------|----------|----------|
| `torch.randn(size, ...)` | 生成标准正态分布 | N(0, 1) | 待编写 |
| `torch.randn_like(input, ...)` | 按输入形状生成 | N(0, 1) | 待编写 |
| `torch.normal(mean, std, ...)` | 生成任意正态分布 | N(mean, std) | 待编写 |

**使用示例**：
```python
# 标准正态分布
x = torch.randn(3, 4)

# 指定均值和标准差
x = torch.normal(mean=10.0, std=2.0, size=(3, 4))

# 每个元素不同的参数（广播）
means = torch.tensor([1.0, 2.0, 3.0])
stds = torch.tensor([0.1, 0.2, 0.3])
x = torch.normal(means, stds)
```

#### 1.3 整数随机数
| API | 说明 | 输出范围 | 详细文档 |
|-----|------|----------|----------|
| `torch.randint(low, high, size, ...)` | 生成随机整数 | [low, high) | [详见](discrete_call_flow.md) |
| `torch.randint_like(input, low, high, ...)` | 按输入形状生成 | [low, high) | [详见](discrete_call_flow.md) |
| `torch.randperm(n, ...)` | 生成随机排列 | [0, n) 的排列 | [详见](discrete_call_flow.md) |

**使用示例**：
```python
# 生成 [0, 10) 的随机整数
x = torch.randint(0, 10, (3, 4))

# 生成 [0, 100) 的随机排列（用于打乱数据）
perm = torch.randperm(100)
shuffled_data = data[perm]
```

---

### 2. 原地操作（Inplace Operations）- 修改现有张量

这些函数以 `_` 结尾，直接修改张量内容，避免内存分配。**这些是工厂函数的底层实现**。

#### 2.1 基础分布
| API | 说明 | 分布/范围 | 关系 |
|-----|------|-----------|------|
| `tensor.uniform_(from, to)` | 填充均匀分布 | [from, to) | **`torch.rand` 的核心** |
| `tensor.normal_(mean, std)` | 填充正态分布 | N(mean, std) | **`torch.randn` 的核心** |
| `tensor.random_(from, to)` | 填充随机整数 | [from, to) | **`torch.randint` 的核心** |

**架构关系**：
```python
# torch.rand 的实现逻辑
def rand(size, ...):
    result = torch.empty(size, ...)    # 1. 创建未初始化张量
    result.uniform_(0, 1)              # 2. 原地填充 [0, 1) 均匀分布
    return result
```

**使用示例**：
```python
# 复用已有张量，避免内存分配
buffer = torch.empty(1000, 1000)
for epoch in range(100):
    buffer.uniform_(0, 1)  # 原地填充，无额外内存分配
    # 使用 buffer...
```

#### 2.2 其他分布
| API | 说明 | 分布 | 详细文档 |
|-----|------|------|----------|
| `tensor.log_normal_(mean, std)` | 对数正态分布 | exp(N(mean, std)) | 待编写 |
| `tensor.exponential_(lambda)` | 指数分布 | Exp(λ) | 待编写 |
| `tensor.cauchy_(median, sigma)` | 柯西分布 | Cauchy(median, σ) | 待编写 |
| `tensor.geometric_(p)` | 几何分布 | Geom(p) | 待编写 |
| `tensor.bernoulli_(p)` | 伯努利分布 | Bernoulli(p) | 待编写 |

**应用场景**：
- **log_normal**: 资产价格、生物体大小等正偏态数据
- **exponential**: 等待时间、事件间隔（泊松过程）
- **cauchy**: 重尾分布，机器学习中的鲁棒估计
- **geometric**: 首次成功前的失败次数
- **bernoulli**: 二分类、掩码生成

---

### 3. 特殊采样函数

这些函数实现更复杂的采样算法，不能简单地通过变换 uniform/normal 得到。

| API | 说明 | 应用场景 | 详细文档 |
|-----|------|----------|----------|
| `torch.multinomial(weights, num_samples)` | 多项式采样 | 分类模型、强化学习动作选择 | [详见](sampling.md#1-multinomial-采样) |
| `torch.poisson(rates)` | 泊松分布 | 计数事件、排队论 | [详见](sampling.md#2-poisson-分布) |
| `torch.binomial(count, prob)` | 二项分布 | 伯努利试验次数 | [详见](sampling.md#3-binomial-分布) |
| `torch._standard_gamma(alpha)` | Gamma 分布 | 贝叶斯统计、等待时间 | [详见](sampling.md#4-gamma-分布) |

**使用示例**：
```python
# multinomial: 按概率权重采样（常用于语言模型）
logits = model(input)  # [batch, vocab_size]
probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)

# poisson: 生成计数数据
lambda_rates = torch.tensor([2.0, 5.0, 10.0])
counts = torch.poisson(lambda_rates)
```

---

### 4. Generator 管理

控制随机数生成器的状态，用于可复现性和高级控制。

| API | 说明 | 用途 |
|-----|------|------|
| `torch.Generator(device)` | 创建生成器对象 | 独立的随机数流 |
| `torch.manual_seed(seed)` | 设置全局种子 | 实验可复现性 |
| `torch.initial_seed()` | 获取初始种子 | 调试、日志记录 |
| `torch.get_rng_state()` | 获取 RNG 状态 | 保存/恢复状态 |
| `torch.set_rng_state(state)` | 设置 RNG 状态 | 保存/恢复状态 |
| `torch.random.fork_rng(devices)` | 临时分叉 RNG | 数据增强、A/B 测试 |

**使用示例**：
```python
# 全局种子设置（影响所有设备）
torch.manual_seed(42)

# 独立生成器（不影响全局状态）
gen1 = torch.Generator().manual_seed(123)
gen2 = torch.Generator().manual_seed(456)
x = torch.rand(10, generator=gen1)
y = torch.rand(10, generator=gen2)

# 保存和恢复状态
state = torch.get_rng_state()
x = torch.rand(10)
torch.set_rng_state(state)
y = torch.rand(10)  # y == x

# 临时分叉（常用于数据增强）
with torch.random.fork_rng():
    # 这里的随机操作不影响外部状态
    augmented = random_augment(data)
# 外部状态恢复
```

---

## 整体架构

### 分层架构

```
┌─────────────────────────────────────────────────────────┐
│  用户层 (Python API)                                     │
│  torch.rand, torch.randn, torch.randint, ...            │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│  API 绑定层 (Python C Extension)                        │
│  torch._C.rand, torch._C.randn, ...                     │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│  工厂层 (C++ TensorFactories)                           │
│  at::rand() → empty() + uniform_(0, 1)                  │
│  at::randn() → empty() + normal_(0, 1)                  │
└────────────────────┬────────────────────────────────────┘
                     ↓
┌─────────────────────────────────────────────────────────┐
│  分布层 (Distributions)                                  │
│  uniform_(), normal_(), exponential_(), ...             │
└────────────────────┬────────────────────────────────────┘
                     ↓
         ┌───────────┴───────────┐
         ↓                       ↓
┌──────────────────┐    ┌──────────────────┐
│  CPU 实现         │    │  CUDA 实现        │
│  MT19937 引擎     │    │  Philox 引擎      │
└──────────────────┘    └──────────────────┘
```

### 核心设计原则

1. **设备无关性**: 相同的 API 在 CPU 和 CUDA 上都能工作
2. **分派机制**: 根据张量设备自动选择 CPU 或 CUDA 实现
3. **性能优化**:
   - CUDA 使用向量化生成（一次 4 个随机数）
   - Grid-Stride Loop 提高 GPU 利用率
4. **可复现性**: 相同种子产生相同序列

---

## CPU vs CUDA 实现对比

### RNG 引擎选择

| 特性 | CPU (MT19937) | CUDA (Philox) |
|------|---------------|---------------|
| **算法类型** | 线性反馈移位寄存器 | 计数器模式密码学 |
| **状态大小** | 2496 字节 (624×32bit) | 16 字节 (128bit) |
| **并行性** | 串行生成 | 完美并行 |
| **跳跃性能** | O(n) | O(1) |
| **统计质量** | 高质量 PRNG | 通过 BigCrush 测试 |
| **适用场景** | 单线程高质量 | 大规模并行 |

### 为什么 CUDA 不用 MT19937？

**MT19937 的问题**：
1. **巨大状态**: 每个 CUDA 线程需要 2496 字节，1024 个线程就是 2.4 MB
2. **串行依赖**: 下一个随机数依赖前一个状态，无法并行
3. **难以跳跃**: 无法高效跳到序列任意位置

**Philox 的优势**：
1. **极小状态**: 只需 128 位 counter + 64/128 位 key
2. **无状态计算**: `random(counter, key)` 是纯函数
3. **完美并行**: 每个线程独立，无同步
4. **快速跳跃**: 改变 counter 即可跳到任意位置

详细原理参见：[torch.rand 调用流程 - Appendix A: Philox 算法原理](uniform_call_flow.md#appendix-a-philox-算法原理)

---

## Generator 机制详解

### Generator 的作用

Generator 封装了 RNG 引擎的状态，提供了三个核心功能：

1. **独立的随机数流**: 多个 Generator 互不干扰
2. **状态管理**: 保存和恢复 RNG 状态
3. **可复现性**: 相同种子产生相同序列

### Generator 的类型

| Generator 类型 | 设备 | 底层引擎 | 状态 |
|---------------|------|----------|------|
| `CPUGeneratorImpl` | CPU | MT19937 | 624×32bit + 缓存 |
| `CUDAGeneratorImpl` | CUDA | Philox | seed + offset |

### Generator 在调用链中的使用

以 `torch.rand(3, 4, device='cuda')` 为例：

```
1. Python 层: torch.rand(3, 4, device='cuda')
   ↓
2. C++ 工厂层: at::rand(size, device='cuda')
   ↓ 没有传入 generator，使用 nullptr
3. 分布层: uniform_(result, 0, 1, generator=nullptr)
   ↓
4. CUDA kernel 入口: uniform_kernel(iter, 0, 1, gen)
   ↓ 【Generator 第一次使用】
5. 获取默认生成器:
   gen = get_generator_or_default<CUDAGeneratorImpl>(nullptr)
   ↓ 返回 CUDA 默认生成器
6. 计算 Grid/Block 配置: calc_execution_policy(numel)
   ↓ 返回 counter_offset, grid, block
7. 【Generator 第二次使用】
   获取 Philox 状态:
   rng_state = gen->philox_cuda_state(counter_offset)
   ↓ 返回 PhiloxCudaState{seed, offset}
   ↓ 同时增加 gen 内部的 offset
8. 启动 CUDA kernel:
   kernel<<<grid, block>>>(numel, rng_state, ...)
   ↓
9. 设备端: 每个线程用 rng_state 初始化 Philox
   curand_init(seed, threadIdx, offset, &state)
   ↓
10. 生成随机数:
    curand_uniform4(&state) → float4
```

**关键点**：
- Generator 只在 **主机端（Host）** 使用，用于获取种子和偏移量
- 真正的随机数生成在 **设备端（Device）**，使用 Philox 算法
- 每次调用会更新 Generator 的 offset，确保不同调用产生不同随机数

详细流程参见：[torch.rand 调用流程 - 第 7 节完整调用流程图](uniform_call_flow.md#7-完整调用流程图cuda)

---

## 常见分布实现对比

| 分布 | 变换方法 | CPU 算法 | CUDA 函数 | 特点 |
|------|---------|----------|-----------|------|
| **uniform** | 线性映射 | U | `curand_uniform4` | 最简单，向量化 |
| **normal** | Box-Muller | Box-Muller | `curand_normal4` | cuRAND 内置 |
| **exponential** | 逆变换 | -log(1-U) | `curand_uniform4` + transform | 简单高效 |
| **cauchy** | 逆变换 | tan(π(U-0.5)) | `curand_uniform4` + transform | 重尾分布 |
| **log_normal** | 复合 | exp(normal) | `curand_normal4` + exp | 两步变换 |
| **bernoulli** | 比较 | U < p | `curand_uniform4` + 比较 | 向量化友好 |
| **geometric** | 逆变换 | ⌈log(U)/log(1-p)⌉ | `curand_uniform4` + transform | 离散分布 |
| **multinomial** | 累积概率 | 二分搜索 | GPU 并行采样 | 复杂算法 |
| **poisson** | 变换/拒绝 | 条件选择 | 条件选择 | λ 分段实现 |
| **gamma** | Marsaglia-Tsang | 拒绝采样 | 拒绝采样 | 高质量采样 |

**实现模式**：
1. **简单变换**: uniform, exponential, cauchy → 直接数学变换
2. **内置函数**: normal → cuRAND 提供优化实现
3. **复合分布**: log_normal → 组合已有分布
4. **复杂算法**: multinomial, gamma → 专门的采样算法

详细对比参见：
- [torch.rand 调用流程 - 第 9 节常见分布实现对比](uniform_call_flow.md#9-常见分布实现对比)
- [复杂采样函数详解 - 第 6 节实现对比总结](sampling.md#6-实现对比总结)

---

## 如何选择合适的 API

### 按使用场景选择

#### 1. 需要新张量 → 工厂函数
```python
# 初始化权重
weight = torch.randn(512, 256) * 0.01

# 生成噪声
noise = torch.rand(batch_size, channels, height, width)
```

#### 2. 复用现有张量 → 原地操作
```python
# 训练循环中复用缓冲区
noise_buffer = torch.empty(1000, 1000, device='cuda')
for epoch in range(100):
    noise_buffer.uniform_(-1, 1)
    # 使用 noise_buffer...
```

#### 3. 需要可复现性 → 使用 Generator
```python
# 实验可复现
torch.manual_seed(42)

# 或使用独立生成器
gen = torch.Generator().manual_seed(42)
x = torch.rand(10, generator=gen)
```

#### 4. 多个独立随机流 → 多个 Generator
```python
# 数据增强和模型初始化分离
data_gen = torch.Generator().manual_seed(111)
model_gen = torch.Generator().manual_seed(222)

augmented = random_crop(data, generator=data_gen)
init_weights(model, generator=model_gen)
```

### 按分布类型选择

| 需求 | API | 示例 |
|------|-----|------|
| 均匀分布 | `torch.rand` | 权重初始化、dropout 掩码 |
| 正态分布 | `torch.randn` | 高斯噪声、Xavier 初始化 |
| 整数索引 | `torch.randint` | 随机采样、类别标签生成 |
| 打乱顺序 | `torch.randperm` | 数据打乱、批次划分 |
| 分类采样 | `torch.multinomial` | 语言模型采样、强化学习 |
| 二分类 | `tensor.bernoulli_` | Dropout、二值掩码 |
| 等待时间 | `tensor.exponential_` | 事件间隔模拟 |

---

## 性能优化建议

### 1. CUDA 性能优化

```python
# ✅ 好: 一次生成大张量
x = torch.rand(10000, device='cuda')

# ❌ 差: 多次生成小张量
x = torch.cat([torch.rand(1, device='cuda') for _ in range(10000)])
```

**原因**: CUDA kernel 启动有固定开销，大张量摊销开销。

### 2. 复用张量避免分配

```python
# ✅ 好: 原地操作复用
buffer = torch.empty(1000, 1000, device='cuda')
for i in range(1000):
    buffer.uniform_(0, 1)

# ❌ 差: 每次分配新张量
for i in range(1000):
    x = torch.rand(1000, 1000, device='cuda')
```

### 3. 向量化生成

CUDA 实现内部已经向量化（`curand_uniform4` 一次生成 4 个），无需手动优化。

### 4. Generator 管理

```python
# ✅ 好: 复用 Generator
gen = torch.Generator(device='cuda').manual_seed(42)
for i in range(1000):
    x = torch.rand(100, 100, generator=gen, device='cuda')

# ❌ 差: 每次创建新 Generator
for i in range(1000):
    gen = torch.Generator(device='cuda').manual_seed(42)
    x = torch.rand(100, 100, generator=gen, device='cuda')
```

---

## 常见问题（FAQ）

### Q1: 为什么 CUDA 和 CPU 生成的随机数不一样？

**A**: 不同设备使用不同的 RNG 引擎：
- CPU 使用 MT19937
- CUDA 使用 Philox

即使设置相同种子，也会产生不同序列。如需一致性：
```python
# 在 CPU 生成再传输到 GPU
x = torch.rand(100, 100).cuda()
```

### Q2: `torch.rand` 和 `tensor.uniform_` 有什么区别？

**A**:
- `torch.rand`: 工厂函数，创建新张量
- `tensor.uniform_`: 原地操作，修改现有张量
- `torch.rand` 内部调用 `empty() + uniform_(0, 1)`

### Q3: 如何保证训练可复现？

**A**: 设置所有设备的种子：
```python
import torch
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
```

### Q4: Generator 的 offset 会溢出吗？

**A**: 理论上会，但实际不可能：
- uint64 最大值: 2^64 - 1 ≈ 1.8×10^19
- 假设每秒生成 10^12 个随机数，需要 584 年才会溢出

### Q5: 多 GPU 训练如何管理随机数？

**A**: 每个 GPU 有独立的默认 Generator：
```python
# 每个 GPU 使用不同种子
torch.cuda.manual_seed_all(42)  # 所有 GPU 使用相同种子

# 或为每个 GPU 设置不同种子
for i in range(torch.cuda.device_count()):
    torch.cuda.manual_seed(42 + i)
```

---

## 扩展阅读

### 官方文档
- [torch.random](https://pytorch.org/docs/stable/random.html)
- [torch.Generator](https://pytorch.org/docs/stable/generated/torch.Generator.html)
- [Reproducibility](https://pytorch.org/docs/stable/notes/randomness.html)

### 算法论文
- **Philox**: [Parallel Random Numbers: As Easy as 1, 2, 3](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf)
- **MT19937**: [Mersenne Twister](http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html)
- **Box-Muller**: [A Note on the Generation of Random Normal Deviates](https://projecteuclid.org/euclid.aoms/1177706645)

### 相关代码
- CPU Generator: `aten/src/ATen/CPUGeneratorImpl.h`
- CUDA Generator: `aten/src/ATen/cuda/CUDAGeneratorImpl.h`
- 分布实现: `aten/src/ATen/native/Distributions.cpp`
- CUDA kernel: `aten/src/ATen/native/cuda/DistributionTemplates.h`

---

## 贡献指南

欢迎贡献更多文档！如果你想：
- 补充缺失的分布文档
- 添加更多使用示例
- 修正错误或改进表述

请参考现有文档的结构风格，提交 PR 到 PyTorch 仓库。

---

**文档版本**: v1.0
**最后更新**: 2024
**维护者**: PyTorch Community
