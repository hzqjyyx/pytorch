这个文件实现了旋转位置编码（Rotary Position Embedding, RoPE）的主机端生成和切片功能。

**主要功能：**

- **`rope_enum` 枚举**：定义三种RoPE应用方式（none、interleaved、half_rotated）

- **`generate_rotary_cos_sin()` 函数**：
  - 为给定序列长度和旋转维度生成cos和sin查找表
  - 基于随机初始化的角度值（0到2π）计算三角函数值
  - 返回两个HostTensor，用于后续的RoPE应用

- **`slice_rotary_cos_sin()` 函数**：
  - 从完整的cos/sin表中提取指定偏移量和长度的子集
  - 支持序列长度不同时的动态切片
  - 返回切片后的cos和sin张量，用于处理可变长度输入

**核心设计目的**：为HIP/ROCm平台的Flash Attention实现提供高效的RoPE支持，避免重复计算三角函数值。
