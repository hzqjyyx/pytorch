这个文件定义了一系列编译时条件分支宏，用于在 Flash Attention CUDA 实现中根据不同条件进行静态代码分支。

**主要功能：**

- **BOOL_SWITCH** - 基础宏，根据运行时条件在编译时生成 true/false 两个分支，通过 constexpr 变量传递给 lambda 表达式

- **DROPOUT_SWITCH** - 控制 Dropout 功能的开启/关闭，若定义了 `FLASHATTENTION_DISABLE_DROPOUT` 则强制禁用

- **ALIBI_SWITCH** - 控制 ALiBi (Attention with Linear Biases) 功能，若定义了 `FLASHATTENTION_DISABLE_ALIBI` 则强制禁用

- **EVENK_SWITCH** - 控制是否处理非偶数长度的 K 维度，若定义了 `FLASHATTENTION_DISABLE_UNEVEN_K` 则强制假设 K 长度为偶数

- **LOCAL_SWITCH** - 控制本地/块级注意力特性，若定义了 `FLASHATTENTION_DISABLE_LOCAL` 则强制禁用

- **FP16_SWITCH** - 根据条件在 `cutlass::half_t` (FP16) 和 `cutlass::bfloat16_t` (BF16) 之间切换数据类型

- **HEADDIM_SWITCH** - 根据注意力头维度动态选择优化的编译版本 (支持 32/64/96/128/160/192/224/256)

**设计目的** - 通过编译时宏展开避免运行时分支开销，同时保持代码灵活性，允许通过编译标志禁用某些功能以减少二进制大小
