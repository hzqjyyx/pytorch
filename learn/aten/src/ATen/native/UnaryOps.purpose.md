# UnaryOps.cpp/h 主要功能

## 核心职责

实现 PyTorch 中所有一元运算符（unary operations）的 CPU 和 CUDA 通用逻辑层，作为前端 API 和后端设备特定实现之间的桥梁。

## 架构设计

### 1. 三层实现模式

**Meta Function Layer (UnaryOps.cpp:173-293)**
- 定义输出张量的元信息（形状、dtype、布局）
- 不执行实际计算
- 两类主要宏：
  - `CREATE_UNARY_FLOAT_META_FUNC`: 整数/浮点输入→浮点输出（如 `acos`, `exp`, `log`）
  - `CREATE_UNARY_META_FUNC`: 保持 dtype（如 `bitwise_not`, `round`）

**Implementation Layer (UnaryOps.cpp:301-395)**
- `TORCH_IMPL_FUNC` 通过设备分发调用具体 kernel
- `CREATE_UNARY_TORCH_IMPL_FUNC(func_out, func_stub)` 宏生成标准实现
- 特殊处理：
  - `ceil/floor/round/trunc`: 整数输入直接复制，浮点才调用 kernel
  - `signbit`: bool 输入填充 false

**Dispatch Declaration (UnaryOps.h:26-92)**
- `DECLARE_DISPATCH` 声明设备分发函数指针
- 实际 kernel 在 CPU/CUDA 专用文件中实现

### 2. 运算符类型分组

**数学函数（浮点输出）**
```cpp
// 三角函数: acos, asin, atan, cos, sin, tan + 双曲版本
// 指数/对数: exp, exp2, expm1, log, log10, log2, log1p
// 特殊函数: erf, erfc, erfinv, digamma, lgamma, i0
// Bessel 函数: special_bessel_j0/j1/y0/y1
```

**位/逻辑运算**
```cpp
bitwise_not, logical_not, signbit
```

**取整运算**
```cpp
ceil, floor, round, trunc, frac
```

**其他**
```cpp
abs, neg, sqrt, rsqrt, reciprocal, sigmoid, sign, sgn
```

## 关键实现细节

### 1. 复数处理特殊逻辑

**abs/angle 复数→实数转换** (UnaryOps.cpp:543-576)
```cpp
// 复数输入返回对应的浮点类型
unary_op_impl_with_complex_to_float_out(result, self, abs_stub, ...)
// ComplexFloat → Float, ComplexDouble → Double
```

**real/imag 视图机制** (UnaryOps.cpp:578-613)
```cpp
// 通过 view_as_real + select 实现零拷贝
at::select(at::view_as_real(self), dim=-1, index=0/1)
// 处理 conjugate bit 的负号传播
```

### 2. Lazy Evaluation 优化

**conj/neg 延迟计算** (UnaryOps.cpp:592-658)
```cpp
Tensor _conj(const Tensor& self) {
  Tensor self_ = self.alias();
  self_._set_conj(!self.is_conj());  // 仅设置 bit
  return self_;
}

Tensor resolve_conj(const Tensor& self) {
  if (!self.is_conj()) return self;
  return self.clone();  // 真正执行共轭
}
```

### 3. 类型提升规则

**整数→浮点提升**
- `rad2deg/deg2rad`: 整数输入自动提升到默认浮点类型 (UnaryOps.cpp:501-502)
- `angle`: 对整数输入应用浮点运算

**严格类型检查**
- `trunc/floor/ceil`: 拒绝复数输入 (UnaryOps.cpp:261-290)
- `neg`: 拒绝 bool 输入 (UnaryOps.cpp:253)
- `positive`: 拒绝 bool 输入 (UnaryOps.cpp:850)

### 4. 别名（Alias）机制

为 NumPy/SciPy 兼容性提供多个函数名：
```cpp
arccos → acos
arcsin → asin  
arctan → atan
absolute → abs
special_erf → erf
special_gammaln → lgamma
```

## 工具函数模式

### unary_op_impl_* 函数族 (UnaryOps.cpp:404-485)

```cpp
// 基础输出版本
unary_op_impl_out(result, self, stub)

// 浮点输出版本（整数提升）
unary_op_impl_float_out(result, self, stub, args...)

// 复数→浮点特殊处理
unary_op_impl_with_complex_to_float_out(result, self, stub, promotes_int)

// In-place 版本
unary_op_impl_(self, out_impl)  // 调用 out_impl(self, self)
```

## 特殊函数示例

### mvlgamma (多元 log-gamma) (UnaryOps.cpp:887-940)
```cpp
// 计算 log(Γ_p(x)) = Σ lgamma(x + (1-i)/2) + p(p-1)/4 * log(π)
args = arange(-p/2 + 1/2, 1/2, 1/2).add(self.unsqueeze(-1))
return args.lgamma_().sum(-1).add_(p*(p-1)/4 * log(π))
```

### frexp (浮点分解) (UnaryOps.cpp:942-974)
```cpp
// 返回 (mantissa, exponent) 满足 self = mantissa * 2^exponent
frexp_out(self, mantissa, exponent)
// mantissa: 浮点型, exponent: int32
```

### nan_to_num (NaN 替换) (UnaryOps.cpp:804-842)
```cpp
// 整数输入直接复制
// 浮点输入: NaN→指定值, +inf→指定值, -inf→指定值
nan_to_num_stub(iter, nan, pos_inf, neg_inf)
```

## 性能优化要点

1. **TensorIterator 统一接口**: 所有运算通过 `TensorIterator::unary_op` 处理广播、步幅、并行化
2. **设备分发零开销**: `DECLARE_DISPATCH` 编译期解析设备类型
3. **整数快速路径**: `ceil/floor/round/trunc` 对整数输入跳过计算
4. **别名避免拷贝**: `conj/_neg_view` 通过 bit 操作延迟实际计算

## 与其他模块的关系

- **TensorIterator**: 提供统一的元素级并行迭代框架
- **DispatchStub**: 设备分发机制（CPU/CUDA/MPS 等）
- **native/cpu/UnaryOpsKernel.cpp**: CPU vectorized 实现
- **native/cuda/UnaryOpsKernel.cu**: CUDA kernel 实现

---

## ROCm 相关
- 与 CUDA 共用相同的分发机制，通过 HIP 层适配

## Backward 相关
- UnaryOps.cpp 只实现前向传播
- 反向传播在 `torch/csrc/autograd/FunctionsManual.cpp` 和相应 backward kernel 中实现
- 梯度公式示例：`d(sin(x))/dx = cos(x)`, `d(exp(x))/dx = exp(x)`
