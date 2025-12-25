# Activation.cpp/h 核心功能

这两个文件实现了 PyTorch 中各种神经网络激活函数的 CPU 端核心逻辑。

## 架构设计

采用 **meta function + dispatch stub** 模式：

1. **Meta functions** (`at::meta` namespace)：定义张量形状和内存布局逻辑，通过 `TensorIterator` 配置输入输出关系
2. **Impl functions** (`at::native` namespace)：调用设备特定的 dispatch stub（CPU/CUDA 分发）
3. **Dispatch stubs**：函数指针，在运行时根据设备类型选择具体实现

## 主要激活函数实现

### 1. ReLU 系列
- **ReLU**: 通过 `clamp_min(self, 0)` 实现，简单高效
- **ReLU6**: 通过 `hardtanh(self, 0, 6)` 实现
- **Leaky ReLU**: 支持负斜率，meta function 中检查 in-place backward 时负斜率的非法情况
- **PReLU**: 参数化 ReLU，weight 参数通过 reshape 广播到与输入相同维度

### 2. ELU 系列
- **ELU**: 支持 alpha/scale/input_scale 参数，backward 中检查 in-place 操作时 alpha 必须非负
- **SELU**: 固定参数的 ELU，`SELU_ALPHA=1.6732...`, `SELU_SCALE=1.0507...`
- **CELU**: 通过 `elu(self, alpha, 1.0, 1/alpha)` 实现

### 3. Sigmoid 系列
- **Sigmoid/Tanh**: 直接调用 ATen ops
- **Hardsigmoid**: 分段线性近似，通过专用 stub
- **Hardswish**: 移动端优化，优先使用 XNNPACK 加速（`C10_MOBILE && USE_XNNPACK`）
- **SiLU (Swish)**: `x * sigmoid(x)`，有专门的 fused kernel

### 4. GELU
- 支持两种近似模式：`None` (erf) 和 `Tanh`
- **MKLDNN 优化路径**：
  - 检查 `use_mkldnn()`: 判断是否启用、是否连续、数据类型支持（BF16/FP16/FP32）
  - 使用 `ideep::eltwise_forward/backward` 调用 oneDNN 库
  - ARM64 平台支持 Tanh 近似
- CPU fallback 使用 `GeluKernel` dispatch

### 5. 其他激活函数
- **Threshold**: 条件赋值 `self <= threshold ? value : self`
- **Softplus**: `log(1 + exp(beta * x)) / beta`，带 threshold 参数避免数值溢出
- **Hardshrink/Softshrink**: 稀疏化激活，lambda 参数控制死区范围
- **Mish**: `x * tanh(softplus(x))`
- **RReLU**: 随机 Leaky ReLU
  - 训练时：负值乘以 uniform(lower, upper) 随机数，存入 noise tensor
  - 推理时：负值乘以 `(lower + upper) / 2`
  - CPU 实现使用 `CPUGeneratorImpl` 生成随机数，需要 mutex 保护

### 6. Log Sigmoid
- CPU 版本使用 buffer 保存中间结果（避免重复计算）
- CUDA 版本忽略 buffer，直接从 input 重新计算（显存换计算）

## TensorIterator 使用模式

```cpp
// 一元操作
build_unary_op(maybe_get_output(), self);

// 二元操作（borrowing：输出复用某个输入的 metadata）
build_borrowing_binary_op(maybe_get_output(), grad_output, self);

// 手动配置
TensorIteratorConfig()
  .add_output(result)
  .add_const_input(self)
  .promote_inputs_to_common_dtype(true)
  .build();
```

## 性能优化点

1. **XNNPACK 移动端加速**：Hardswish 在移动设备上优先使用
2. **MKLDNN/oneDNN**：GELU 在 x86 CPU 上使用 BLAS 库优化
3. **Contiguous 内存**：RReLU 等函数要求连续内存，通过 `contiguous()` 确保
4. **Type promotion**：`promote_inputs_to_common_dtype` 自动类型提升
5. **In-place 检查**：`set_check_mem_overlap(false)` 对幂等操作（如 threshold）跳过检查

## Backward 相关（简述）
- 所有激活函数都有对应的 backward 实现
- 检查 in-place backward 的合法性（ELU/Leaky ReLU 负斜率限制）
- 使用 `TensorIterator::borrowing_binary_op` 复用输入内存布局

## ROCm 相关（简述）
- 文件中未包含 ROCm 特定代码
- 通过 dispatch 机制，ROCm 实现在其他文件中注册
