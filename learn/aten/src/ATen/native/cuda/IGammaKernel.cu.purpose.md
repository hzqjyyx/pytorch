# IGammaKernel.cu 功能分析

这个文件实现了**正则化不完全伽马函数**的CUDA内核，提供两个主要函数：
- `igamma`: 正则化下不完全伽马函数 (regularized lower incomplete gamma)
- `igammac`: 正则化上不完全伽马函数 (regularized upper incomplete gamma)

## 核心算法

### 1. 有理函数求值 (`ratevl`)
- 计算两个多项式的比值
- 根据 |x| 与 1 的大小关系，选择直接计算或使用 1/x 变换
- 用于后续的Lanczos近似

### 2. Lanczos求和 (`lanczos_sum_expg_scaled`)
- 使用预定义的13阶系数进行Lanczos近似
- 调用 `ratevl` 计算有理函数值

### 3. 辅助因子计算 (`_igam_helper_fac`)
- 计算 x^a * exp(-a) / gamma(a)
- 根据 |a-x| 的大小选择不同计算策略：
  - 差异较大时：直接对数计算
  - 差异较小时：使用Lanczos近似避免精度损失

### 4. 级数展开方法

**下不完全伽马级数** (`_igam_helper_series`):
- 使用 DLMF 8.11.4 公式
- 幂级数展开，最多迭代2000次
- 适用于 x ≤ a 的情况

**上不完全伽马级数** (`_igamc_helper_series`):
- 使用 DLMF 8.7.3 公式
- 特别处理以避免相消误差

**渐近级数** (`_igam_helper_asymptotic_series`):
- 使用 DLMF 8.12.3/8.12.4 公式
- 适用于 a 较大且 a ≈ x 的情况
- 使用预计算的25×25系数矩阵 d[][]
- 计算互补误差函数 erfc 加上修正项

### 5. 连分式方法 (`_igamc_helper_continued_fraction`)
- 使用 DLMF 8.9.2 公式
- 适用于 x > a 时计算 igammac

## 主计算函数

### `calc_igammac(a, x)` - 上不完全伽马函数

**边界条件处理**:
- x < 0 或 a < 0: 返回 NaN
- a = 0, x > 0: 返回 0
- x = 0: 返回 1
- a = ∞: 返回 1 (除非 x = ∞ 返回 NaN)
- x = ∞: 返回 0

**算法选择策略**:
```
if (a > SMALL) && (a < LARGE) && |x-a|/a < 0.3:
    使用渐近级数
else if (a > LARGE) && |x-a|/a < 4.5/√a:
    使用渐近级数
else if x > 1.1:
    if x < a: 返回 1 - igamma(a,x)
    else: 使用连分式
else if x ≤ 0.5:
    根据 -0.4/log(x) < a 选择级数或互补计算
else:
    根据 x*1.1 < a 选择级数或互补计算
```

### `calc_igamma(a, x)` - 下不完全伽马函数

**边界条件处理**:
- x < 0 或 a < 0: 返回 NaN
- a = 0, x > 0: 返回 1
- x = 0: 返回 0
- a = ∞: 返回 0 (除非 x = ∞ 返回 NaN)
- x = ∞: 返回 1

**算法选择策略**:
```
if (a > SMALL) && (a < LARGE) && |x-a|/a < 0.3:
    使用渐近级数
else if (a > LARGE) && |x-a|/a < 4.5/√a:
    使用渐近级数
else if (x > 1.0) && (x > a):
    返回 1 - igammac(a,x)
else:
    使用级数展开
```

## CUDA内核接口

- `igamma_kernel_cuda`: 调度 float/double 类型的 igamma 计算
- `igammac_kernel_cuda`: 调度 float/double 类型的 igammac 计算
- 使用 `gpu_kernel` 包装计算函数
- 通过 `CalcIgamma` 结构体统一两个函数的调用接口

## 数值精度控制

- **MACHEP**: 机器精度 (double: 1.11e-16, float: 5.96e-8)
- **MAXITER**: 最大迭代次数 2000
- **MAXLOG**: 最大对数值防止溢出
- 使用 `__noinline__` 标记主计算函数以减少编译时间

## 常量阈值

- SMALL = 20.0
- LARGE = 200.0  
- SMALLRATIO = 0.3
- LARGERATIO = 4.5

---

**忽略内容**:
- 文件顶部注释第9-10行提到Windows下 `__device__` lambda 的链接限制
- 文件底部第551-552行建议不要在此文件添加新内核以控制CUDA编译时间
