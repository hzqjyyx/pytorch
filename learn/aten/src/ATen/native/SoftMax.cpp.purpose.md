# SoftMax.cpp 主要功能

这个文件实现了 PyTorch 中的 softmax 和 log_softmax 操作的 CPU 端代码。

## 核心实现

### 1. Meta 函数（元数据设置）

**`_softmax` 和 `_log_softmax` meta 函数** (40-78行)
- 验证 dim 参数有效性
- 设置输出张量的形状和数据类型
- 处理 `half_to_float` 转换选项（half 精度转 float）

### 2. CPU 实现函数

**`softmax_cpu_out`** (303-329行)
- 主要的 softmax CPU 实现入口
- 检查不支持 half_to_float（仅 CUDA 支持）
- 处理空张量和标量情况
- 根据 dim 是否为最后一维，分派到优化的 `softmax_lastdim_kernel` 或通用的 `softmax_kernel`

**`log_softmax_cpu_out`** (331-356行)
- log_softmax 的 CPU 实现
- 逻辑类似 softmax_cpu_out，但调用 log_softmax 专用 kernel

### 3. 公共 API

**`softmax` 和 `log_softmax`** (411-479行)
- 用户调用的公共接口
- 处理 dtype 转换
- CUDA half-to-float 特殊处理
- 传播命名维度信息

**`softmax_out` 和 `log_softmax_out`** (425-516行)
- 输出到预分配张量的版本
- 处理非连续输出张量（需要临时缓冲）

### 4. Masked Softmax

**`masked_softmax_cpu`** (540-596行)
- 支持掩码的 softmax 实现
- 三种掩码类型：
  - Type 0: 注意力掩码 (L×L)
  - Type 1: padding 掩码 (B×L) 
  - Type 2: 通用掩码（与输入同形）
- 调用 `host_softmax` 模板函数

**`host_softmax` 模板** (152-243行)
```cpp
template <typename scalar_t>
void host_softmax(Tensor& output, const Tensor& input, int64_t dim, 
                  bool* mask, const std::optional<int64_t> mask_type_)
```
- 并行化实现（使用 `parallel_for`）
- 计算流程：
  1. 找到非掩码位置的最大值（数值稳定性）
  2. 计算 `exp(x - max)` 的和
  3. 归一化：除以总和
- 掩码位置输出为 0
- 处理全掩码情况（输出 NaN）

### 5. 特殊别名

- `special_softmax` (463行): softmax 的别名
- `special_log_softmax` (518行): log_softmax 的别名
- 支持命名维度的重载 (532-538行)

### 6. Kernel 分派

定义了多个分派点 (522-530行)：
```cpp
DEFINE_DISPATCH(softmax_lastdim_kernel);
DEFINE_DISPATCH(log_softmax_lastdim_kernel);
DEFINE_DISPATCH(softmax_kernel);
DEFINE_DISPATCH(log_softmax_kernel);
```
实际的向量化实现在 `cpu/SoftmaxKernel.h` 中

## 设计特点

- **优化分支**: 最后一维的 softmax 使用专门优化的 kernel
- **数值稳定性**: 减去最大值避免 exp 溢出
- **并行化**: 使用 `parallel_for` 处理外层和内层维度
- **类型支持**: 通过 `AT_DISPATCH_FLOATING_TYPES_AND2` 支持多种浮点类型（Float、Double、Half、BFloat16）
- **连续性处理**: 自动转换为连续张量以优化性能

---

**其他功能（简要）：**
- Backward 函数：`_softmax_backward_data`、`_log_softmax_backward_data` meta 函数 + 实现
- `host_softmax_backward` 模板：masked softmax 的梯度计算
- `masked_softmax_backward_cpu`：masked softmax 反向传播
- ROCm 相关：无（仅 CUDA/CPU）
