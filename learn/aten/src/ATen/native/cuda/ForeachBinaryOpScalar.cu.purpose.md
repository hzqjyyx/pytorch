## 文件功能概述

这个文件实现了 PyTorch 中对多个张量执行二元标量操作的 CUDA 内核。

### 核心机制

**模板化操作框架**：使用模板参数 `Op` 定义不同的二元操作（加、乘、除、减、幂等），通过 `AT_DISPATCH_*` 宏根据张量数据类型（整数、浮点、复数、布尔等）分发到具体实现。

**两种操作模式**：
- `foreach_binary_op()` - 返回新张量，输入张量保持不变
- `foreach_binary_op_()` - 原地修改，直接改动输入张量

**多张量批处理**：通过 `multi_tensor_apply<>()` 将多个张量的操作融合到单个 CUDA 内核调用中，提高效率。

**快速路径优化**：`can_use_fast_route()` 检查是否能使用优化的 CUDA 内核，否则回退到较慢的通用实现。

### 具体实现细节

- **scalar_reciprocal()** - 将除法转换为乘以倒数（行 198-208），避免整数精度问题
- **foreach_tensor_sub_scalar_kernel_cuda()** - 减法操作有额外的布尔类型检查，遵循 torch.sub 的约束（行 238-276）
- **foreach_scalar_pow_list_kernel_cuda()** - 特殊幂运算，支持标量^张量的反向操作（行 180-189）
- **clamp 操作** - 使用 min/max 函子实现上限/下限钳制（行 280-281）

### 主要功能点

- 为 `_foreach_add`, `_foreach_mul`, `_foreach_div`, `_foreach_sub`, `_foreach_pow`, `_foreach_clamp_max/min` 等操作提供 CUDA 后端
- 支持数据类型：整数、浮点、复数、布尔、Half、BFloat16
- 通过宏 `FOREACH_BINARY_OP_SCALAR` 减少代码重复
- 版本跟踪（`increment_version()`）确保自动求导能正确追踪原地修改
- 调用前验证张量限制（`check_foreach_api_restrictions()`）
