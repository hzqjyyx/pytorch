这个文件实现了 PyTorch ATen 库中的池化（Pooling）操作的前向传播函数。主要包括：

**1. 自适应平均池化（Adaptive Average Pooling）**
- `adaptive_avg_pool1d()` - 1D自适应平均池化，通过unsqueeze转换为2D操作后再squeeze回1D

**2. 自适应最大池化（Adaptive Max Pooling）**
- `adaptive_max_pool1d()` - 1D自适应最大池化，返回output和indices元组

**3. 标准最大池化（Max Pooling）**
- `max_pool1d_with_indices()` - 1D最大池化，支持kernel_size、stride、padding、dilation、ceil_mode参数
- `max_pool2d()` - 2D最大池化，具有多种后端支持（quantized、mkldnn、xnnpack）
- `max_pool3d()` - 3D最大池化

**4. 平均池化（Average Pooling）**
- `avg_pool1d()` - 1D平均池化，支持ceil_mode和count_include_pad参数

**关键特点：**
- 1D操作通过unsqueeze/squeeze转换为2D操作实现
- 参数验证和检查（维度范围、参数尺寸）
- 多后端支持（CPU、CUDA、量化、MKL-DNN、XNNPACK等）
- 命名张量（Named Tensor）的名称传播处理
- 使用结构化绑定和tuple返回多个输出
