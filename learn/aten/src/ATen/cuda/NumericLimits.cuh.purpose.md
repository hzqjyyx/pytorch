这个文件定义了 PyTorch ATen 库中用于 CUDA 计算的数值类型的极限值。

**主要功能：**

- 为常用类型（bool、int8_t、int16_t、int32_t、int64_t、Half、BFloat16、float、double）提供数值极限的特化模板
- 为每种类型定义四个静态方法：
  - `lowest()`：类型的最小值
  - `max()`：类型的最大值
  - `lower_bound()`：下界（积分类型同 lowest，浮点类型为 -∞）
  - `upper_bound()`：上界（积分类型同 max，浮点类型为 +∞）
- 所有方法都标记为 `__host__ __device__`，可在 CPU 和 GPU 上调用
- 浮点类型（float、double）的边界值使用 INFINITY 常数表示无穷大
- 半精度和 BFloat16 类型通过位表示直接构造极限值

**使用场景：** 在 CUDA 内核和主机代码中快速获取类型的数值范围，用于边界检查、初始化和类型转换操作。
