这个文件实现了 3D 分数最大池化（Fractional Max Pooling 3D）的 CUDA 前向传播核心功能。

## 核心机制

**分数池化的本质**：与标准最大池化不同，分数池化允许非整数的池化比率。通过随机采样确定每个输出位置对应的输入区域起始位置。

## 关键函数

### `get_intervals` (38-52行)
计算每个输出位置对应的输入起始索引：
- 使用线性插值公式：`alpha = (inputSize - poolSize) / (outputSize - 1)`
- 对于第 `index` 个输出位置，起始位置为：`(index + sample) * alpha - sample * alpha`
- 最后一个输出位置特殊处理，直接返回 `inputSize - poolSize`

### `fractional_max_pool3d_out_frame` 核函数 (55-118行)
每个线程负责计算一个输出位置 (t, h, w)：

1. **定位输出坐标**：从线程索引计算 outputT, outputH, outputW
2. **确定池化窗口**：调用 `get_intervals` 获取三个维度的起始位置 poolT, poolH, poolW
3. **寻找最大值**：
   - 初始化 `maxVal` 为类型下界
   - 遍历 `[poolT, poolT+poolSizeT) × [poolH, poolH+poolSizeH) × [poolW, poolW+poolSizeW)` 区域
   - 对宽度维度做特殊优化：poolSizeW 在 [2,7] 时使用展开循环
   - NaN 处理：如果遇到 NaN 值会被选中（`val > maxVal || at::_isnan(val)`）
   - **保持第一个最大值**：与 THNN 行为一致
4. **写入结果**：保存最大值和对应的线性索引

### `fractional_max_pool3d_out_cuda` 入口函数 (248-307行)
- 处理 4D（无 batch）和 5D（有 batch）输入，统一转为 5D 处理
- 配置 CUDA 网格和线程块：
  - 每个线程块最多 128 线程
  - 网格三维：(输出平面大小/128, 通道数, batch大小)
- 使用 `AT_DISPATCH_FLOATING_TYPES_AND2` 支持 Float32/Float64/Half/BFloat16

## 设计细节

**线程组织**：
- `blockIdx.z` = batch 索引
- `blockIdx.y` = 通道（plane）索引  
- `blockIdx.x` 和 `threadIdx.x` 组合映射到输出空间位置

**性能优化**：
- 宽度维度循环根据 poolSizeW 大小选择不同路径（90-111行）
- 使用 `PackedTensorAccessor64` 实现高效多维数组访问
- 限制线程块为 4 个 warp（128 线程）

---

**Backward 相关**：
- `fractional_max_pool3d_backward_out_frame`：反向传播核函数，使用原子加法累积梯度
- `fractional_max_pool3d_backward_out_cuda_template`：反向传播模板函数
- 标记为非确定性操作（由于 atomicAdd）
