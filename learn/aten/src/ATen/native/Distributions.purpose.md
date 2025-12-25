# ATen/native/Distributions.cpp & Distributions.h 主要功能

这两个文件实现了 PyTorch 中各种概率分布的采样和相关操作。

## 核心架构

**Distributions.h** 定义了设备无关的采样算法模板，**Distributions.cpp** 提供了 CPU 实现和分发逻辑。

采用 **DispatchStub** 模式实现跨设备分发：
- `DEFINE_DISPATCH(bernoulli_tensor_stub)` 等定义在 cpp 文件中
- 实际 CPU/CUDA 实现通过 stub 注册机制分发
- 每个分布都有对应的 Stub 结构体封装设备特定的调用

## 实现的分布类型

### 1. Bernoulli 分布（伯努利分布）
- `bernoulli_(Tensor& self, const Tensor& p, Generator)`：以张量 p 作为成功概率
- `bernoulli_(Tensor& self, double p, Generator)`：标量概率版本
- 输出 0/1 值

### 2. Normal 分布（正态分布）
多种重载支持：
- `normal_(Tensor& self, double mean, double std)`：就地填充
- `normal(const Tensor& mean, double std)`：mean 为张量
- `normal(double mean, const Tensor& std)`：std 为张量  
- `normal(const Tensor& mean, const Tensor& std)`：广播采样
- 通过 `normal_stub` 分发到设备实现

### 3. Uniform 分布（均匀分布）
- `uniform_(Tensor& self, double from, double to, Generator)`
- `[from, to)` 区间均匀采样
- Meta 版本 `uniform_meta_` 是 no-op（用于符号追踪）

### 4. Exponential 分布（指数分布）
- `exponential_(Tensor& self, double lambda, Generator)`
- lambda 为速率参数

### 5. Cauchy 分布
- `cauchy_(Tensor& self, double median, double sigma, Generator)`
- 重尾分布

### 6. Log-Normal 分布（对数正态分布）
- `log_normal_(Tensor& self, double mean, double std, Generator)`

### 7. Geometric 分布（几何分布）
- `geometric_(Tensor& self, double p, Generator)`
- 首次成功所需的试验次数

### 8. Random 系列
- `random_(Tensor& self, Generator)`：整个数据类型范围
- `random_(Tensor& self, int64_t to, Generator)`：`[0, to)` 整数
- `random_(Tensor& self, int64_t from, int64_t to, Generator)`：`[from, to)` 整数
- 通过 `RandomFromToStub` 处理范围逻辑

### 9. Poisson 分布（泊松分布）
**Distributions.cpp:81-126** 实现 `sample_poisson(double lambda, CPUGeneratorImpl*)`：
- **lambda >= 10**：使用 **Transformed Rejection Method**（Hoermann 1993）
  - 计算参数 `a, b, invalpha, vr`
  - 接受-拒绝循环，带快速路径（us >= 0.07 && V <= vr）
  - 对数域计算避免数值溢出
- **lambda == 0**：直接返回 0
- **0 < lambda < 10**：使用 **Knuth 算法**（inverse transform）
  - 累乘均匀随机数直到 `prod <= exp(-lambda)`
- 来自 NumPy 实现（MIT 许可）

`_s_poisson_cpu` (451-466) 通过 `TensorIterator` 对每个元素调用 `sample_poisson`。

### 10. Gamma 分布
**Distributions.h:90-118** `sample_gamma` 实现（设备通用）：
- **alpha < 1**：Boost 技巧，scale 调整后递归到 alpha+1
- **alpha >= 1**：**Marsaglia-Tsang 方法**（2000）
  - 构造 `d = alpha - 1/3`, `c = 1/sqrt(9d)`
  - 循环：采样 `x ~ N(0,1)`, 计算 `v = (1 + cx)³`
  - 快速接受：`u < 1 - 0.0331x⁴`
  - 对数检查：`log(u) < 0.5x² + d(1 - v + log(v))`

`_s_gamma_cpu` (468-496) 封装 uniform/normal 采样器调用 `sample_gamma`。

### 11. Dirichlet 分布
**Distributions.cpp:498-543** `_s_dirichlet_cpu`：
1. 对每个 alpha 分量采样 **Gamma(alpha, 1)**，结果为 double 精度防止下溢
2. 归一化：`gamma_i / sum(gamma)` 
3. 裁剪到 `[min, nexttoward(1.0, 0)]` 确保和为 1

### 12. Binomial 分布（二项分布）
**Distributions.h:150-250** 实现三种算法：

**binomial_inversion** (151-168)：小 np 时使用
- 累积几何分布直到超过 count

**btrs** (171-221)：大 np 时使用（**Transformed Rejection Sampling**）
- 计算 stddev, mode `m = floor((n+1)p)`
- 快速接受区域：`us >= 0.07 && V <= v_r` (86% × v_r 概率)
- Stirling 近似计算接受上界

**sample_binomial** (224-250) 路由逻辑：
- `count*prob < 10`：inversion
- `count*prob >= 10`：btrs
- 处理 `prob > 0.5` 通过对称性 `count - sample(count, 1-prob)`

`_s_binomial_cpu` (426-449) 通过 `TensorIterator` 应用。

### 13. Multinomial（多项式采样）
**Distributions.cpp:548-635** `multinomial_out`：

**输入校验**：
- 概率张量必须 1D 或 2D，浮点类型
- 类别数 `<= 2^24`（float32 精度限制）
- 非放回采样时 `n_sample <= n_categories`

**快速路径**（不放回或 n_sample=1）：**Gumbel-Max 技巧**
- 计算 `q = p / Exp(1)`（等价于 `p * exp(U)` 其中 U 均匀）
- `n_sample=1`：取 `argmax(q)`
- `n_sample>1`：取 `topk(q, n_sample)`
- 避免显式循环，向量化实现

**放回采样**：调用 `multinomial_with_replacement_stub`（设备特定实现）

## 梯度计算辅助函数

### Standard Gamma Gradient
**standard_gamma_grad_one** (Distributions.h:308-374)：
计算 `-∂cdf/∂alpha / pdf` 用于重参数化梯度：
- **x < 0.8**：Taylor 级数展开（5 阶）
- **alpha > 8 且 x ≈ alpha**：Rice 鞍点展开精确逼近
- **其他**：双变量有理函数近似（8 系数）

`_standard_gamma_grad_cpu` (391-404) 对张量应用。

### Dirichlet Gradient  
**dirichlet_grad_one** (Distributions.h:455-516)：
计算 `-∂cdf/∂alpha / pdf / (1-x)` 的 Beta 分布梯度：
- **x 接近 0**（`boundary < 2.5`）：`_beta_grad_alpha_small` Taylor 展开（10 阶）
- **x 接近 1**（`boundary < 0.75`）：对称性利用 `_beta_grad_beta_small`
- **alpha, beta 都大（> 6）**：`_beta_grad_alpha_mid` 鞍点逼近
- **其他**：3×3×4 系数有理校正逼近

`_dirichlet_grad_cpu` (406-420) 对张量应用。

## 设计模式要点

### 模板化 Stub 结构
每个分布定义形如：
```cpp
template<typename RNG>
struct BernoulliStub {
  void operator()(Tensor& self, ..., std::optional<Generator> gen) {
    bernoulli_tensor_stub(self.device().type(), ...);
  }
};
```
允许设备无关的模板实例化（`templates::bernoulli_impl_`）调用设备特定实现。

### TensorIterator 驱动
大多数采样通过 `TensorIteratorConfig` 构建迭代器，传递给 `cpu_serial_kernel` 或设备 stub，逐元素应用采样逻辑。

### Generator 锁管理
CPU 实现中显式获取 `CPUGeneratorImpl` 并加锁：
```cpp
std::lock_guard<std::mutex> lock(generator->mutex_);
```
确保线程安全的随机数生成。

### 数值稳定性技巧
- Gamma/Dirichlet 采样使用 double 中间精度
- 对数域计算（Poisson, Binomial 接受检查）
- Stirling 近似减少 lgamma 调用
- NaN 检查和边界裁剪

---

## 简要说明（忽略部分）

**ROCm 相关**：
- `__HIPCC__` 条件编译
- `c10::hip::compat` 数学函数兼容层

**Backward 相关**：
- `_standard_gamma_grad_cpu`：Gamma 分布的 alpha 梯度
- `_dirichlet_grad_cpu`：Dirichlet 分布的 alpha 梯度
- `digamma_one`：ψ(x) 函数用于梯度计算
- `_beta_grad_*`：Beta 分布的重参数化梯度近似
