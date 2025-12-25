## `Math.h` 提供高精度数学特殊函数实现

这是 PyTorch 的 CPU 端核心数学库头文件,提供各类特殊函数的高精度数值计算实现。

### 核心功能

**1. 概率统计函数**
- `calc_erfinv` - 逆误差函数,用于 rational approximation + Newton-Raphson 修正
- `calc_ndtri` - 正态分布逆累积分布函数(Probit 函数)
- `calc_log_ndtr` - 对数正态累积分布
- `erfcx_y100` - 指数缩放补余误差函数,使用 100 段 Chebyshev 多项式分段逼近

**2. Gamma 函数族**
- `calc_digamma` - Digamma 函数(Psi 函数),使用渐近展开
- `trigamma` - Trigamma 函数(Psi 的一阶导数)
- `calc_polygamma` - Polygamma 函数(通过 zeta 函数实现)
- `zeta` - Riemann Zeta 函数
- `calc_igamma` / `calc_igammac` - 正则化不完全 Gamma 函数,包含多个辅助函数:
  - `_igam_helper_fac` - 计算 x^a * exp(-a) / gamma(a)
  - `_igam_helper_series` - 幂级数展开
  - `_igamc_helper_series` / `_igamc_helper_continued_fraction` - 补函数的系列和连分数展开
  - `_igam_helper_asymptotic_series` - 渐近级数(对 a ~ x 情况)
  - `lanczos_sum_expg_scaled` - Lanczos 近似

**3. Bessel 函数**
- Modified Bessel: `calc_i0`, `calc_i1`, `calc_i1e`(指数缩放版本)
- Standard Bessel: `bessel_j0_forward`, `bessel_j1_forward`, `bessel_y0_forward`, `bessel_y1_forward`
- Modified Bessel 第二类: `modified_bessel_k0_forward`, `modified_bessel_k1_forward`
- Spherical Bessel: `spherical_bessel_j0_forward`
- Airy 函数: `airy_ai_forward`

**4. 正交多项式**
- Chebyshev: `chebyshev_polynomial_t/u/v/w_forward` 及 shifted 版本
- Hermite: `hermite_polynomial_h_forward`, `hermite_polynomial_he_forward`
- Laguerre: `laguerre_polynomial_l_forward`
- Legendre: `legendre_polynomial_p_forward`

**5. 多项式求值工具**
- `polevl` - 标准多项式求值
- `ratevl` - 有理函数求值(分子/分母多项式比)
- `chbevl` - Chebyshev 多项式求值

### 数值方法

- 使用 **Chebyshev 多项式展开** 进行区间逼近(如 Bessel 函数)
- 采用 **分段逼近** 策略(如 erfcx 的 100 段分割)
- **连分数展开** 用于高精度计算
- **渐近级数** 处理参数较大情况
- **Newton-Raphson 迭代** 提高精度
- 针对不同参数范围使用不同算法分支

### 类型支持

- 主要支持 `float` 和 `double`
- 为 `c10::BFloat16` 和 `c10::Half` 提供向上转型包装
- 模板化设计,部分函数支持 `C10_HOST_DEVICE`(CPU/CUDA 通用)

### 代码来源

大部分实现源自:
- **Cephes Math Library**(3-Clause BSD 许可)
- **SciPy** 的数值算法
- **Boost Math Toolkit**
- **MIT Faddeeva** 项目

**忽略部分:**
- ROCm 相关:无明显 ROCm 专用代码
- Backward 相关:无梯度反向传播代码(纯前向数值计算)
