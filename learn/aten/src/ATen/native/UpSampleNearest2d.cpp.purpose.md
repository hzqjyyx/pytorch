这个文件实现了 PyTorch 中 2D 最近邻上采样（Nearest Neighbor Upsampling）的核心功能。

**主要功能：**

- **元函数定义**（at::meta 命名空间）：
  - `TORCH_META_FUNC(upsample_nearest2d)`：验证输入张量维度和大小，计算输出形状
  - `TORCH_META_FUNC(_upsample_nearest_exact2d)`：精确版本的上采样元函数，功能相同但算法略有不同
  - 对应的 backward 函数验证梯度张量的形状

- **实现函数**（at::native 命名空间）：
  - `TORCH_IMPL_FUNC(upsample_nearest2d_out_cpu)`：CPU 上的正向传播，调用 `upsample_nearest2d_kernel`
  - `TORCH_IMPL_FUNC(_upsample_nearest_exact2d_out_cpu)`：精确版本的 CPU 实现
  - Backward 对应函数实现梯度计算

- **公共接口**：
  - `upsample_nearest2d()`：用户级 API，接受可选输出大小或缩放因子，转换参数后调用底层实现
  - `_upsample_nearest_exact2d()`：精确上采样的用户级 API

- **核心机制**：
  - 通过 `DEFINE_DISPATCH` 宏注册四个分发函数，允许不同后端（CPU/CUDA）提供优化实现
  - 支持通过 `scales_h`/`scales_w` 或 `output_size` 指定上采样参数
  - 保持输入张量的内存格式
