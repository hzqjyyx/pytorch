**文件功能分析：ChannelShuffle.cpp**

这个文件实现了PyTorch中的通道混洗操作，主要包含以下功能：

**核心函数：**

1. **`channel_shuffle_cpu`** (22-36行)
   - CPU上的通道混洗实现
   - 处理空张量的别名情况
   - 根据输入的内存格式创建输出张量
   - 调用kernel进行实际计算

2. **`channel_shuffle`** (38-62行)
   - 主入口函数，包含输入验证
   - 检查张量维度 > 2
   - 验证组数为正且能整除通道数
   - 优先使用XNNPACK（移动端优化）
   - 回退到通用native实现
   - 传播命名张量的维度名称

3. **`math_channel_shuffle`** (64-89行)
   - 数学层面的通道混洗实现
   - 将张量从 `[B, C, H, W]` reshape为 `[B, groups, C/groups, H*W]`
   - 通过permute重新排列维度：`[B, C/groups, groups, H*W]`
   - 最后reshape回原始大小

**关键特性：**

- 支持命名张量（Named Tensor）的名称传播
- 处理空张量的边界情况
- 支持多种内存格式（ChannelsLast等）
- 包含输入参数的详细验证

**输出：**

- 通道混洗后的张量
- 保持原始内存布局
- 保留命名维度信息
