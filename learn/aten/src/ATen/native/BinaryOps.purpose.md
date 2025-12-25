# BinaryOps.cpp 和 BinaryOps.h 功能分析

## 核心职责

这两个文件实现了 PyTorch ATen 库中所有**二元张量运算**（binary tensor operations）的CPU端调度和接口层。它们不包含实际的计算kernel，而是作为**分发层**（dispatch layer），将操作路由到特定设备的实现。

## 架构模式

采用了 PyTorch 的**结构化kernel模式**：

1. **Meta函数** (`TORCH_META_FUNC`): 负责输出形状推导和类型提升
2. **Impl函数** (`TORCH_IMPL_FUNC`): 调用设备相关的stub实现
3. **Stub声明** (在.h中): 定义函数指针类型，支持多设备分发

## 主要运算类别

### 1. 基础算术运算
- **add/sub/mul/div**: 加减乘除及其变体（trunc/floor除法）
- **remainder/fmod**: 取余运算
- **copysign**: 符号复制
- **ldexp**: 浮点数乘以2的幂次

### 2. 比较运算
- **lt/le/gt/ge/eq/ne**: 六种比较运算符
- 每个运算都有Tensor-Tensor和Tensor-Scalar两个重载版本
- 别名函数：`less`→`lt`, `greater`→`gt`, `not_equal`→`ne`等

### 3. 位运算
- **bitwise_and/or/xor**: 按位与或非
- **bitwise_left_shift/right_shift**: 位移运算
- 提供了Python风格的运算符别名（`__and__`, `__lshift__`等）

### 4. 逻辑运算
- **logical_and/or/xor**: 逻辑运算（输出布尔值）
- 与位运算的区别：会先将输入转换为布尔值

### 5. 数学特殊函数
- **三角函数**: `atan2`及其别名`arctan2`
- **多项式**: Chebyshev (t/u/v/w), Hermite (h/he), Laguerre, Legendre等
  - 包括shifted版本
  - 支持Tensor-Tensor和Tensor-Scalar组合
- **统计函数**: 
  - `igamma/igammac`: 不完全伽马函数
  - `zeta`: Riemann zeta函数
  - `xlogy/xlog1py`: 处理零值的对数乘法
- **其他**: 
  - `logaddexp/logaddexp2`: 数值稳定的log(exp(x)+exp(y))
  - `gcd/lcm`: 最大公约数/最小公倍数
  - `hypot`: 欧几里得距离
  - `nextafter`: 下一个浮点数
  - `heaviside`: 阶跃函数

### 6. 极值运算
- **maximum/minimum**: 逐元素最大最小值（传播NaN）
- **fmax/fmin**: 浮点最大最小值（忽略NaN）
- **max/min**: 别名函数

## 关键设计细节

### 类型提升策略
```cpp
build_borrowing_binary_op()      // 整数类型保持不变
build_borrowing_binary_float_op() // 强制提升到浮点类型
```

### 标量处理
所有Tensor-Scalar运算都通过`wrapped_scalar_tensor()`转换为Tensor-Tensor运算后重新分发。

### 输入验证
- **alpha_check**: 验证alpha参数类型与输出类型兼容
- **sub_check**: 禁止布尔张量做减法（引导用户使用XOR）

### 特殊情况处理

**零张量优化** (zerotensor):
- `mul_zerotensor`: 0×任意 = 0
- `div_zerotensor`: 处理0/0→NaN, x/0→INF, 0/x→0
- `add_zerotensor/sub_zerotensor`: 保留非零张量
- `linalg_cross_zerotensor`: 叉乘零张量

**设备选择逻辑** (`correct_out_device`):
当self在CPU上时使用other的设备，否则使用self的设备。

### 宏代码生成
```cpp
CREATE_BINARY_META_FUNC(func)              // 批量生成meta函数
CREATE_BINARY_TORCH_IMPL_FUNC(func, stub)  // 批量生成impl函数
CREATE_COMPARISON_SCALAR_TENSOR_META_FUNC  // 生成比较运算的两个重载
CREATE_COMPARISON_SCALAR_TENSOR_IMPL_FUNC  // 生成比较运算的实现
```

## 函数变体命名规范
- **基础**: `add(Tensor, Tensor)` - 返回新张量
- **inplace**: `add_(Tensor&, Tensor)` - 修改第一个输入
- **out变体**: `add_out(Tensor, Tensor, Tensor&)` - 写入预分配输出
- **别名**: `divide` → `div`, `multiply` → `mul`, `subtract` → `sub`

## ROCm 和 Backward 相关
- 所有backward相关函数（`sigmoid_backward`, `tanh_backward`, `logit_backward`）用于反向传播梯度计算
- ROCm相关内容在此CPU实现文件中未直接体现，通过stub机制在CUDA/ROCm后端实现
