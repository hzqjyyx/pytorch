这个文件实现了 PyTorch functorch 库中池化操作的批处理规则（batch rules），用于支持 vmap（向量化映射）功能。

## 核心功能

**max_pool_with_indices_batch_rule_helper** (lines 14-40)
- 模板函数，处理带索引的最大池化操作的批处理逻辑
- 接收批处理维度 `self_bdim` 和池化参数（kernel_size, stride, padding, dilation, ceil_mode）
- 根据输入张量的逻辑秩（logical rank）分两种情况处理：

**情况1：logical_rank == n + 1** (lines 24-29)
- 输入形状为 `[B, C, H, W, ...]`（批次维度 + 通道维度 + 空间维度）
- 直接将批次维度移到最前面，然后调用标准池化函数
- 返回结果的批次维度索引为 0

**情况2：logical_rank == n + 2** (lines 30-39)
- 输入形状为 `[B, N, C, H, W, ...]`（多个批次维度）
- 将批次维度 reshape 合并到第一维：`[B, N, ...] -> [B*N, ...]`
- 调用标准池化函数
- 将结果 reshape 回原始批次结构：`[B*N, ...] -> [B, N, ...]`

## 具体实现

**max_pool2d_with_indices_batch_rule** (lines 50-56)
- 2D 最大池化的批处理规则
- 调用 helper 函数，传入 n=2 和 `at::max_pool2d_with_indices`

**max_pool3d_with_indices_batch_rule** (lines 42-48)
- 3D 最大池化的批处理规则
- 调用 helper 函数，传入 n=3 和 `at::max_pool3d_with_indices`

## 注册机制 (lines 58-75)

使用 `TORCH_LIBRARY_IMPL` 将批处理规则注册到 FuncTorchBatched 调度键：
- 自适应平均池化：`_adaptive_avg_pool2d`, `_adaptive_avg_pool3d`
- 平均池化：`avg_pool2d`, `avg_pool3d`
- 自适应最大池化：`adaptive_max_pool2d`, `adaptive_max_pool3d`
- 带索引的最大池化：`max_pool2d_with_indices`, `max_pool3d_with_indices`

---

**忽略的内容：**
- Backward 操作：`*_backward` 相关的注册（lines 60, 62, 65, 66, 69, 70, 73, 74）
- ROCm 相关：文件中未涉及
