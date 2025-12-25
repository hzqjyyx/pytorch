这个文件定义了 PyTorch ATen 库中用于随机数分布采样的核心辅助结构体。

**主要内容：**

- **uniform_int_from_to_distribution** - 离散均匀分布采样，范围 [base, base+range)
  - 根据范围大小选择使用 32 位或 64 位随机数生成器

- **uniform_int_full_range_distribution** - 完整范围整数均匀分布，覆盖 int64_t 的全部值

- **uniform_int_distribution** - 标准整数均匀分布，范围 [0, max_value(T)]

- **uniform_real_distribution** - 实数均匀分布，范围 [from, to)
  - 对 double 使用 64 位随机数，其他类型使用 32 位

- **normal_distribution** - 正态分布，使用 Box-Muller 算法
  - 缓存了一个样本（Box-Muller 一次生成两个）以提高效率
  - 支持自定义均值和标准差

- **bernoulli_distribution** - 伯努利分布，基于概率 p

- **geometric_distribution** - 几何分布，基于概率 p

- **exponential_distribution** - 指数分布，基于 lambda 参数

- **cauchy_distribution** - 柯西分布，基于中位数和 sigma

- **lognormal_distribution** - 对数正态分布，基于均值和标准差
  - 通过正态分布采样后取指数

- **SFINAE 宏机制** - 用于检测和调用生成器中的缓存方法（double/float normal samples）

- **设计特点**：所有分布都支持 HOST 和 DEVICE（GPU）执行，通过 `C10_HOST_DEVICE` 宏标记
