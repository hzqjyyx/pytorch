# TensorCompare.cpp/h 主要功能

这两个文件实现了 PyTorch 中张量的比较和条件操作，包含元素级比较、查找最值、条件选择等核心功能。

## 核心功能模块

### 1. 比较与相等性检查

**allclose / isclose** (lines 324-420)
- `allclose`: 检查两个张量是否在给定容差内全部接近
- `isclose`: 逐元素检查是否满足 `|a - b| <= atol + rtol * |b|`
- 容差检查：支持绝对容差(atol)和相对容差(rtol)
- 特殊处理：NaN 可选视为相等(`equal_nan`)，整型输入转换为浮点进行比较
- 类型提升：自动将 bool/整型提升到默认浮点类型避免精度损失

**特殊值检查** (lines 422-477)
- `isnan`: 利用 NaN 不等于自身的特性 (`self != self`)
- `isinf`: 检查绝对值是否等于无穷大，复数需检查实部和虚部
- `isfinite`: 整型恒为 true，浮点需检查既非 NaN 也非 inf
- `isreal`: 整型/浮点恒为 true，复数检查虚部是否为零
- `isposinf/isneginf`: 分别检查正无穷和负无穷

### 2. 值域限制 (Clamping)

**clamp 系列** (lines 92-154, 820-930)
- `clamp(min, max)`: 将值限制在 [min, max] 区间内
- `clamp_min/clamp_max`: 单侧限制
- `clip`: clamp 的别名接口
- 类型提升策略：
  - scalar 参数参与类型提升计算
  - 禁止 inplace 操作改变张量类型
  - 不支持复数类型
- 实现路径：
  - scalar 参数走 `clamp_scalar_stub`
  - tensor 参数走 `clamp_stub` / `maximum_stub` / `minimum_stub`
- NaN 处理：如果 min/max 为 NaN，结果填充 NaN

### 3. 最值与归约

**min/max (dim)** (lines 277-295, 760-776)
- 沿指定维度查找最小/最大值
- 返回值和索引两个张量
- 元函数检查：维度包装、零元素检查、不支持复数
- 特殊情况：
  - 零元素：返回对应形状的空张量
  - 单元素：直接填充值和索引 0
  - 一般情况：调用 `max_stub/min_stub` 设备分发

**量化版本 qmin/qmax** (lines 778-807)
- 仅支持 per-tensor 量化 (`kPerTensorAffine`)
- 对量化张量的 `int_repr()` 调用普通 min/max
- 结果重新打包为量化张量，保持原 scale 和 zero_point

**mode** (lines 668-739)
- 查找沿维度的众数(出现次数最多的值)
- 返回众数值和首次出现的索引
- 设备支持：仅 CPU/CUDA/XPU
- 空张量处理类似 min/max

**aminmax** (line 810-818)
- 同时返回最小值和最大值
- 已弃用 `_aminmax`，建议使用 `aminmax`

### 4. 条件选择 (where)

**where 操作** (lines 581-666)
- `where(condition, x, y)`: 根据 condition 逐元素选择 x 或 y
- 类型处理：
  - 自动推导 x, y 的公共类型
  - 支持 scalar 和 tensor 的各种组合重载
  - uint8 条件已弃用，自动转换为 bool
- 设备处理：允许 CPU scalar 在非 CPU 设备上使用，自动迁移
- `where(condition)`: 返回非零元素索引，等价于 `nonzero_numpy()`

### 5. 集合操作 (isin)

**isin** (lines 210-253, 974-1025)
- 检查 elements 中的值是否在 test_elements 中
- 支持三种重载：
  - Tensor-Tensor (line 210)
  - Tensor-Scalar (line 225)
  - Scalar-Tensor (line 240)
- 参数：
  - `assume_unique`: 是否假设输入已去重
  - `invert`: 反转结果
- 实现策略：
  - test_elements 较小：直接逐元素比较 (`isin_default_stub`)
  - test_elements 较大：排序算法 (`isin_sorting`)
- 启发式阈值：`test_elements.numel() < 10.0 * elements.numel()^0.145`

**isin_sorting 算法** (lines 519-569)
1. 去重：如果 `assume_unique=false`，对两个输入调用 `_unique`
2. 拼接：连接 elements 和 test_elements
3. 稳定排序：保持 elements 在 test_elements 之前
4. 查找重复：相邻相等的值表示存在于两个集合
5. 反向映射：通过排序索引恢复原始顺序
6. 输出形状：匹配原 elements 形状

### 6. 断言与调试

**断言函数** (lines 479-511)
- `_assert_async_cpu`: 检查张量是否非零
- `_assert_scalar`: 检查标量条件，支持符号布尔值
- `_functional_assert_*`: 函数式版本，克隆 dep_token 以跟踪依赖

**打印** (line 513-515)
- `_print`: 简单输出字符串到 stdout

## 头文件结构 (TensorCompare.h)

定义了分发存根(dispatch stubs)的函数签名：

- `max_stub/min_stub`: 归约最值 (line 22-23)
- `where_kernel`: 条件选择 (line 26)
- `isposinf_stub/isneginf_stub`: 无穷检查 (line 29-30)
- `mode_stub`: 众数计算 (line 33)
- `clamp_stub/clamp_scalar_stub/clamp_min_scalar_stub/clamp_max_scalar_stub`: 值域限制 (line 36-50)
- `isin_default_stub`: 集合成员检查 (line 54)

所有 stub 通过 `DECLARE_DISPATCH` 声明，支持 CPU/CUDA/XPU 等设备的具体实现。

## Named Tensor 支持

Lines 934-972 提供了 Dimname 重载，将命名维度转换为位置索引后调用原函数。

## 辅助功能

- `check_for_unsupported_isin_dtype` (line 82): 拒绝 Bool/ComplexFloat/ComplexDouble
- `check_unsupported_complex` (line 273): 通用复数类型检查
- `out_device` (line 572): 选择非 CPU 设备作为输出设备
- `get_zero_numel_tensor_size` / `_dimreduce_return_trivial_no_ident`: 处理空张量归约

---

**简要列出的其他内容：**
- ROCm: 无特定代码路径，通用 CUDA stub 可能支持
- Backward: 此文件仅包含前向操作，反向传播在 autograd 层实现
