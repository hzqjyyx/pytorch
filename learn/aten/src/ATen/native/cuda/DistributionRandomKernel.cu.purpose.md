这个文件是 PyTorch 的 CUDA 随机数生成内核的包装层。

**主要功能：**

• 为三个随机数生成操作提供 CUDA 内核实现的入口点
• `random_from_to_kernel` - 生成指定范围内的随机整数（base 到 base+range）
• `random_full_64_bits_range_kernel` - 生成全 64 位范围的随机数
• `random_kernel` - 生成 [0, 1) 范围的随机浮点数

• 通过 `get_generator_or_default()` 获取或使用默认的 CUDA 随机数生成器
• 将具体实现委托给 `DistributionTemplates.h` 中的模板函数
• 使用 `REGISTER_DISPATCH` 宏注册这些内核到分发系统，使高层 API 可以调用它们
