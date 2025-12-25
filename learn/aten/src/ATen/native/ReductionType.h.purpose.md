这个文件定义了PyTorch中的reduction操作类型和相关的枚举转换函数。

**主要功能：**

- **ReductionType枚举** - 定义了5种reduction操作类型：MAX、MEAN、MIN、SUM、PROD

- **get_reduction_enum()函数** - 将字符串参数转换为ReductionType枚举值
  - 支持的字符串：max/amax、mean、min/amin、sum、prod
  - 无效输入时抛出TORCH_CHECK错误

- **get_operator_enum()函数** - 用于scatter_reduce操作的向后兼容性支持
  - 新选项模式：调用get_reduction_enum()
  - 旧选项模式：只支持add（映射到SUM）和multiply（映射到PROD）
