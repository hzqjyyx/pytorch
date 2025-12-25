## ts_native_functions.yaml 文件分析

这个文件定义了 PyTorch Lazy Tensor Core (LTC) 后端支持的原生函数操作集合。

**主要结构：**

- **backend**: 指定为 Lazy 后端
- **cpp_namespace**: torch::lazy 命名空间
- **four key sections**:
  1. **full_codegen** - 需要完整代码生成的操作（~140+ 个）
  2. **ir_gen** - 仅需生成中间表示的操作
  3. **supported** - 已支持但需特殊处理的操作
  4. **symint** - 支持符号整数的操作
  5. **autograd** - 自动微分相关操作
  6. **non_native** - 非标准原生模式的自定义操作

**full_codegen 包含的操作类别：**

- **激活函数** - relu, sigmoid, tanh, silu, gelu, elu, softplus 等
- **基础数学运算** - add, mul, div, sub, pow, exp, log, sqrt, abs 等
- **线性代数** - mm, bmm, mv, addmm, baddbmm 等
- **池化和卷积** - conv, avg_pool2d, max_pool2d, adaptive_avg_pool2d 等
- **上采样** - upsample_nearest2d, upsample_bilinear2d 等
- **张量操作** - cat, stack, gather, scatter, index_select, flip 等
- **范数和聚合** - sum, mean, max, min, std, norm, topk 等
- **视图操作**（*_copy 版本）- view_copy, reshape_copy, permute_copy, transpose_copy 等
- **比较操作** - eq, ne, lt, le, gt, ge 等
- **掩码和填充** - masked_fill, constant_pad_nd 等
- **归一化** - batch_norm, layer_norm, group_norm 等
- **嵌入** - embedding, embedding_dense 等
- **损失函数** - binary_cross_entropy, nll_loss, smooth_l1_loss 等
- **随机操作** - bernoulli, normal, uniform, random 等

**关键特点：**

- 使用 *_copy 版本替代传统视图操作以支持函数化
- non_native 部分包含如标量常数和类型转换这样的特殊操作
- symint 支持使得操作能处理符号形状维度
