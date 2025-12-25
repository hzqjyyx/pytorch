# c10/core/MemoryFormat.h 文件分析

## 核心功能

这个文件定义了 PyTorch 中张量内存布局的抽象机制，用于指导算子如何组织输出张量的内存结构。

## 主要内容

**内存格式枚举 (MemoryFormat)**
- `Contiguous`: 连续内存布局（标准 NCHW 格式）
- `Preserve`: 保留输入张量的内存格式
- `ChannelsLast`: 通道优先布局（NHWC 格式）
- `ChannelsLast3d`: 三维通道优先布局（NDHWC 格式）

**步长计算函数**
- `get_channels_last_strides_2d()`: 计算 2D 通道优先格式的步长
- `get_channels_last_strides_3d()`: 计算 3D 通道优先格式的步长

**步长验证函数**
- `is_channels_last_strides_2d()`: 判断张量是否采用 2D 通道优先步长
- `is_channels_last_strides_3d()`: 判断张量是否采用 3D 通道优先步长
- 包含细粒度的辅助函数用于不同的张量大小情况

## 关键特性

- 内存格式**不是**张量的固有属性，仅用于指导算子的输出格式
- 通过步长推断内存格式存在固有歧义（详见 Note [Ambiguous is_channels_last_strides_xd]）
- 对于边界情况（如 N111 张量）做了特殊处理以避免误判

## 快速总结

• 定义张量内存布局的四种格式选项  
• 提供步长计算函数支持通道优先优化  
• 实现步长模式匹配识别隐式的内存格式  
• 处理通道优先推断的歧义和边界案例  
• 支持 2D 和 3D 两种通道优先格式
