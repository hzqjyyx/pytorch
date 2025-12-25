这个文件实现了基于 cuDNN 的 grid_sampler 操作，用于空间变换采样。

**主要功能：**

- **条件编译**：当 `AT_CUDNN_ENABLED()` 为真时编译 cuDNN 实现，否则抛出编译错误提示
- **setSamplerDescriptor() 函数**：将张量的维度信息设置到 cuDNN 空间变换描述符中，用于配置采样器
- **checkGridSize() 函数**：验证网格张量的格式是否为 n×h×w×2（batch_size × height × width × 2个坐标）
- **cudnn_grid_sampler_forward() 函数**：
  - 验证输入张量和网格张量的合法性
  - 确保张量连续性和 GPU 对齐
  - 调用 cuDNN 的 `cudnnSpatialTfSamplerForward()` 执行前向采样
  - 返回采样后的输出张量
- **张量处理**：处理张量的连续性转换、内存分配和 GPU 数据指针管理

**核心操作流程**：输入验证 → 张量连续化 → 创建 cuDNN 描述符 → 调用 cuDNN 采样核函数 → 返回结果
