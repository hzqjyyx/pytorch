# Pool.h 文件功能解析

这是一个**池化操作的通用头文件**，为 PyTorch 的池化层提供类型定义、调度声明和形状计算工具。

## 核心组成

### 1. 调度函数类型定义

**Max Pooling 2D**
```cpp
using max_pool2d_fn = void(*)(const Tensor& output, const Tensor& indices, 
    const Tensor& input, int kW, int kH, int dW, int dH, int padW, int padH, 
    int dilationW, int dilationH);
DECLARE_DISPATCH(max_pool2d_fn, max_pool2d_kernel)
```
- 返回最大值和索引位置
- 支持膨胀卷积（dilation）

**Average Pooling 2D/3D**
```cpp
using avg_pool2d_fn = void(*)(const Tensor& output, const Tensor& input, 
    int64_t kW, int64_t kH, int64_t dW, int64_t dH, int64_t padW, int64_t padH, 
    bool count_include_pad, std::optional<int64_t> divisor_override);
```
- `count_include_pad`: 是否将填充区域计入平均值
- `divisor_override`: 可选的自定义除数，覆盖默认的窗口大小

**Max Pooling 3D**
```cpp
using max_pool3d_fn = void(*)(Tensor& output, Tensor& indices, 
    const Tensor& input, int kW, int kH, int kD, ...);
```
- 多了深度维度（kD, dD, pD, dilationD）

### 2. 输出形状计算

**基础公式实现**
```cpp
pooling_output_shape_pad_lr(inputSize, kernelSize, pad_l, pad_r, stride, dilation, ceil_mode)
```
输出尺寸计算：
```
output = ⌊(input + pad_l + pad_r - dilation*(kernel-1) - 1) / stride⌋ + 1
```

**ceil_mode 处理**（第67-72行）
- `ceil_mode=true`: 向上取整，允许池化窗口超出边界
- 修正：如果最后一个窗口起始位置 `≥ input + pad_l`，则减1确保窗口起始在图像内

**SAME 模式填充**（第91-117行）
```cpp
_pooling_same_mode_padding_lr(inputSize, kernelSize, stride, dilation)
```
- 计算总填充：`total_padding = dilation * (kernel - 1)`
- 优先对称填充 `(left, right)`
- 当 `stride > 2` 且总填充为奇数时，利用"wiggle room"优化为对称
- 支持符号整数（SymInt）用于动态形状

### 3. 形状验证

**pool2d_shape_check**（第121-166行）
验证内容：
- 核大小、步长、膨胀率 > 0
- 输入维度：
  - `ChannelsLast`: 必须 4D (NHWC)，允许 N=0
  - 其他格式：3D (CHW) 或 4D (NCHW)，允许 N=0
- 填充约束：`pad ≤ kernel/2`
- 输出尺寸 ≥ 1

**内存格式适配**（第144-154行）
- `ChannelsLast`（NHWC）：严格要求 4D 且 C/H/W 非零
- 默认（NCHW）：允许 3D 或 4D

**pool3d_shape_check**（第230-289行）
- 要求 4D 或 5D 输入
- 可选的输入尺寸检查：`input >= kernel`（用于 AveragePool3d）
- 所有非批次维度必须 > 0

### 4. 工具函数

**safe_downcast**（第50-58行）
```cpp
template <typename dest_t, typename src_t>
inline dest_t safe_downcast(src_t v)
```
安全的整数类型转换，防止溢出

## 设计特点

1. **模板化形状计算**：支持 `int64_t` 和 `c10::SymInt`，兼容静态/动态形状
2. **统一的验证逻辑**：前向和反向传播共享形状检查
3. **调度抽象**：通过 `DECLARE_DISPATCH` 分离接口和实现，支持 CPU/CUDA 等后端

---

**ROCm/Backward 相关**：
- `max_pool2d_backward_fn`, `avg_pool2d_backward_fn`, `max_pool3d_backward_fn`, `avg_pool3d_backward_fn` 及对应的 `DECLARE_DISPATCH`
- `max_pool2d_backward_shape_check`, `avg_pool2d_backward_shape_check`, `max_pool3d_backward_shape_check`, `avg_pool3d_backward_shape_check`
- 验证 `grad_output` 和 `indices` 的形状与前向输出一致
