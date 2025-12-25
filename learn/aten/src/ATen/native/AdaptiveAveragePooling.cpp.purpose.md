这个文件实现了自适应平均池化（Adaptive Average Pooling）的 CPU 版本，主要包含以下功能：

**核心功能：**

- **adaptive_avg_pool2d_out_cpu_template()** - 模板函数，处理自适应平均池化的核心逻辑，对输出张量进行 resize 并调用内核
- **adaptive_avg_pool2d_cpu()** - 创建空输出张量并调用模板函数
- **adaptive_avg_pool2d_out_cpu()** - 接收预分配的输出张量并执行池化操作
- **adaptive_avg_pool2d_symint()** - 处理符号整数版本，支持动态形状

**优化路径：**

- MKLDNN 支持：如果输入是 MKLDNN 格式，调用 mkldnn_adaptive_avg_pool2d()
- XNNPACK 优化：当输出尺寸为 1×1 时，使用全局平均池化替代自适应池化
- 内存格式保留：输出张量保持输入的内存格式（ChannelsLast 等）

**输入验证：**

- 检查 output_size 必须为 2 维
- 验证输入张量为 3D 或 4D
- 确保非批次维度大小 > 0
- 检查输入输出数据类型一致

**输出处理：**

- 3D 输入：resize 为 (channels, output_height, output_width)
- 4D 输入：resize 为 (batch, channels, output_height, output_width)
- 处理空张量边界情况
