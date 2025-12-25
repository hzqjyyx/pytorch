Col2Im.cpp 实现了 col2im 操作，这是图像处理中的一个关键操作，用于将 im2col 生成的列矩阵重新转换为图像张量。

**核心功能：**

- **Col2Im 转换**：将展开的列格式数据（通常来自 im2col）恢复为原始图像格式
- **支持参数化**：接收 kernel_size、dilation、padding、stride 等卷积参数来指导转换过程
- **批处理支持**：处理批量输入，可以自动添加或移除批次维度
- **类型支持**：通过 `AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3` 宏支持多种数据类型（浮点、复数、BFloat16、Half、Bool）
- **形状验证**：对所有输入参数（output_size、kernel_size、dilation 等）进行严格的尺寸检查
- **转置卷积反向传播支持**：注释中说明处理 output_padding >= stride 的复杂情况

**关键函数：**

- `col2im_out_cpu_template()`：核心模板函数，执行实际的 col2im 转换
- `col2im_out_cpu()`：输出到现有张量的版本
- `col2im_cpu()`：创建新张量并返回结果的版本

**计算逻辑：**

- 根据 output_size、kernel_size、dilation、padding、stride 计算 height_col 和 width_col
- 遍历批次中的每个样本，分别调用底层 col2im 核函数
- 使用卷积公式：`(output_size + 2*padding - dilation*(kernel-1) - 1) / stride + 1`
