这个文件实现了 PyTorch 中一元特殊数学函数在 CUDA 上的核心计算逻辑。

## 核心架构模式

所有函数都遵循相同的实现模式：

1. **双路径编译策略**：通过 `AT_USE_JITERATOR()` 宏条件编译
   - JIT 路径：使用 `jitted_gpu_kernel` 动态生成 CUDA 代码
   - 预编译路径：使用 `gpu_kernel` 直接编译 lambda 表达式

2. **类型分发机制**：使用 `AT_DISPATCH_*` 宏根据张量数据类型分发到对应实现
   - `AT_DISPATCH_FLOATING_TYPES_AND2`：支持 float/double + Half/BFloat16
   - `AT_DISPATCH_COMPLEX_TYPES_AND`：支持复数类型
   - `AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2`：同时支持浮点和复数

3. **TensorIterator 驱动**：所有函数接收 `TensorIteratorBase&` 参数，处理多维张量的内存布局和并行迭代

## 具体实现的函数

### 指数/对数相关
- **exp2** (23-42 行)：计算 2^x
  - JIT: 使用 `exp2_string`
  - 预编译: 调用 `exp2_impl()`

### 修正贝塞尔函数系列
- **i0** (45-63 行)：第一类零阶修正贝塞尔函数
  - 使用 `opmath_t` 提升精度，调用 `calc_i0<opmath_t>(a)`
  
- **i0e** (67-83 行)：指数缩放的 i0，即 exp(-|x|) * i0(x)
  
- **i1** (88-103 行)：第一类一阶修正贝塞尔函数
  
- **i1e** (106-121 行)：指数缩放的 i1

### 激活函数
- **sigmoid** (124-160 行)：σ(x) = 1 / (1 + e^(-x))
  - 复数类型走 JIT 路径
  - 浮点类型直接计算，使用 `opmath_t` 避免精度损失

- **logit** (193-218 行)：sigmoid 的反函数，log(x / (1-x))
  - 支持 eps 参数做数值裁剪 [eps, 1-eps]，防止 log(0) 和除零错误

### 三角/特殊函数
- **sinc** (163-191 行)：sin(πx) / (πx)
  - 特殊处理 x=0 的情况返回 1
  - 使用 `c10::detail::pi<opmath_t>()` 获取高精度 π

### 误差函数家族
- **erf** (254-260 行)：误差函数，直接调用 `::erf(a)`
  
- **erfc** (263-279 行)：互补误差函数，1 - erf(x)
  
- **erfinv** (282-298 行)：误差函数的反函数
  
- **erfcx** (301-315 行)：缩放互补误差函数，exp(x²) * erfc(x)

### 统计分布相关
- **ndtri** (221-235 行)：标准正态分布的反函数（逆 CDF）
  
- **log_ndtr** (238-252 行)：标准正态分布 CDF 的对数

### 窗函数
- **kaiser_window** (318-349 行)：Kaiser 窗函数
  - 需要额外参数：window_length 和 beta
  - 预计算 `inv_alpha`, `inv_i0_beta` 作为常量传入 kernel
  - JIT 路径通过 `extra_args` 传递参数

### 信息论
- **entr** (352-379 行)：熵函数 -x * log(x)
  - 特殊处理：x=0 返回 0，x<0 返回 -∞，NaN 传播

## 关键技术细节

**opmath_t 精度提升**：对于 Half/BFloat16，内部计算使用 float 精度
```cpp
using opmath_t = at::opmath_type<scalar_t>;
return static_cast<scalar_t>(one/(one + std::exp(-opmath_t{a})));
```

**GPU_LAMBDA 宏**：标记 lambda 为设备代码，可在 CUDA kernel 中执行

**REGISTER_DISPATCH**：将 CUDA 实现注册到全局分发表，Python 调用时自动选择 CUDA 后端

---

**ROCm 相关**：无（本文件纯 CUDA 实现）

**Backward 相关**：无（本文件只实现前向计算，反向传播在其他文件）
