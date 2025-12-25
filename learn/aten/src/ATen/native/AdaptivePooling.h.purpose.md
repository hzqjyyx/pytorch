该文件定义了自适应池化（Adaptive Pooling）操作的接口和辅助函数：

**主要内容：**

- **函数指针类型定义**：为 2D 和 3D 的平均池化和最大池化操作定义了函数指针类型（forward 和 backward）

- **分发声明**：使用 `DECLARE_DISPATCH` 宏为各种池化核心操作声明分发器，用于跨不同硬件后端（CPU/CUDA）的实现

- **索引计算函数**：
  - `start_index()` - 计算自适应池化窗口的起始索引
  - `end_index()` - 计算自适应池化窗口的结束索引
  - 这两个函数基于输入大小、输出大小进行映射计算

- **验证函数**：
  - `adaptive_pool_empty_output_check()` - 检查梯度输出张量的维度是否为空，确保反向传播的输入有效

**核心功能概括：**

- 支持的操作：`adaptive_avg_pool2d`、`adaptive_max_pool2d`、`adaptive_avg_pool3d`、`adaptive_max_pool3d`
- 作用：为自适应池化提供跨平台的分发接口和通用计算工具
