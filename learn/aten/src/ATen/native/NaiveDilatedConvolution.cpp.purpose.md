# NaiveDilatedConvolution.cpp 主要功能分析

## 核心目的
实现 CPU 上的朴素（naive）2D/3D 带膨胀（dilated）卷积的前向和反向传播。"naive" 指使用 im2col/vol2col 展开 + GEMM 的经典实现方式。

## 关键组件

### 1. 维度抽象层 (hvol2col/col2hvol)
```cpp
template <typename Dtype, int64_t dim>
void hvol2col(...) // 28-82行
void col2hvol(...) // 86-140行
```
- **hvol** = hyper-volume，统一处理 2D/3D 输入
- `dim=2`: 调用 `im2col`/`col2im`（图像展开）
- `dim=3`: 调用 `vol2col`/`col2vol`（体积展开）
- 作用：将带膨胀的卷积窗口展开成矩阵列，便于用 GEMM 计算

### 2. 核心计算模板 `slow_conv_dilated_all_cpu_template<dim>`
统一处理前向和反向的完整卷积逻辑（171-520行）。

#### 前向传播路径 (output.defined())
**230-329行**：
1. **Bias 初始化**（232-258行）  
   若存在 bias，用 `fill_` 将 `bias[n]` 广播到输出的每个通道

2. **Im2col 展开**（260-270行）  
   ```cpp
   hvol2col(input_n, ..., columns)
   ```
   将输入窗口展开成 `columns` 矩阵：
   - Channels-first: `[C_in×K_h×K_w, H_out×W_out]`
   - Channels-last: `[H_out×W_out, C_in×K_h×K_w]`

3. **GEMM 计算输出**（299-329行）  
   ```
   output = weight × columns + bias
   ```
   - Channels-first: `columns^T × weight^T + output^T`
   - Channels-last: `weight^T × columns^T + output^T`

#### 反向传播路径 (grad_*.defined())
**梯度输入**（336-410行）：
```cpp
columns = weight^T × grad_output  // GEMM
col2hvol(columns, ..., grad_input) // 反向展开
```

**梯度权重**（413-486行）：
```cpp
hvol2col(input, ..., columns)       // 重新展开输入
grad_weight += grad_output × columns^T  // 累加 GEMM
```

**梯度 bias**（489-516行）：
```cpp
grad_bias += grad_output.sum(dims)  // 沿空间维度求和
```

### 3. 公开接口函数

#### `slow_conv_dilated2d_cpu` (524-577行)
- 处理 2D 卷积
- 支持 ChannelsLast 内存格式优化
- 自动处理 batch/non-batch 输入（通过 `unsqueeze(0)`）

#### `slow_conv_dilated3d_cpu` (579-628行)
- 处理 3D 卷积
- 仅支持 Contiguous 格式

## 实现特点

1. **内存格式感知**  
   2D 卷积检测 `thnn_conv_use_channels_last` 来选择布局，3D 仅用默认布局

2. **批处理循环**  
   对每个 batch 元素独立处理（225行 for-loop），避免一次性分配大缓冲区

3. **零拷贝优化**  
   通过 `select(0, elt)` 获取 batch 切片的视图，不产生数据拷贝

4. **GEMM 参数精细调优**  
   根据 channels-first/last 动态调整转置标志（Transpose/NoTranspose）和 leading dimension

5. **Undefined Tensor 处理**  
   通过 `if (xxx.defined())` 选择性计算输出/梯度，避免无用计算

## ROCm/Backward 相关
- **ROCm**: 无特定代码（纯 CPU 实现）
- **Backward**: 
  - `slow_conv_dilated2d_backward_cpu` (630-687行)
  - `slow_conv_dilated3d_backward_cpu` (689-742行)
  - 通过 `output_mask` 按需计算 `grad_input/grad_weight/grad_bias`
  - 注册到 dispatcher: `REGISTER_ALL_CPU_DISPATCH` (744-745行)
