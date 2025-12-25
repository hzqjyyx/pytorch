# ATen Scaled Dot Product Attention (SDP) C++ Backend Selection Utilities

这两个文件实现了PyTorch中Scaled Dot Product Attention的C++后端选择逻辑，用于在不同实现（Flash Attention和Math fallback）之间自动选择最优kernel。

## 核心架构

**sdp_params** 结构体封装了SDP操作的所有输入参数：
- query, key, value tensors
- optional attention mask
- dropout rate
- causal mask flag
- grouped query attention (GQA) enable flag

**SDPBackend** 枚举定义两种后端：
1. `flash_attention` - 优化的Flash Attention实现
2. `math` - 朴素数学实现作为fallback

## 主要功能流程

### 1. Backend Selection (`select_sdp_backend_cpp`)

后端选择的优先级顺序：
```
Flash Attention > Math fallback
```

选择逻辑（sdp_utils_cpp.cpp:61-103）：
1. 检查全局配置是否启用了Math或Flash SDP
2. 按优先级顺序遍历backends
3. 对Flash Attention执行一系列constraint检查
4. 如果所有约束都满足，返回该backend
5. 若Math被用户启用，直接返回（无约束检查）
6. 若无可用backend，以debug模式重新运行检查并打印失败原因

### 2. Flash Attention Constraint Checks

`use_flash_attention_cpp` (sdp_utils_cpp.cpp:36-58) 检查以下约束条件：

**数据类型约束**：
- 支持 Float, Double, BFloat16, Half
- query/key/value必须类型一致

**形状与维度约束**（按检查顺序）：
1. `check_runtime_disabled_flash` - Flash未被运行时禁用
2. `check_nested_tensor` - 不支持Nested Tensor (C++版本限制)
3. `check_for_dropout` - dropout必须为0
4. `check_tensor_shapes` - 所有输入必须是4维
5. `check_batch_size_and_num_heads_dense` - batch size和num_heads必须一致（不支持GQA）
6. `check_attn_mask_shape` - attention mask形状必须符合2D或4D规范
7. `check_head_dim_size_cpp` - query/key/value的最后一维（head_dim）必须相等
8. `check_nonzero_sequence_lengths_dense` - 序列长度不能为0
9. `check_last_dim_stride_equals_1_dense` - 最后一维stride必须为1（连续内存）

### 3. Head Dimension Check

`check_head_dim_size_cpp` (sdp_utils_cpp.cpp:14-34) 专门验证：
- `query.size(-1) == key.size(-1) == value.size(-1)`
- 这是Flash Attention的硬性要求

### 4. 辅助工具函数

**Scale计算** (sdp_utils_cpp.h:48-55)：
```cpp
softmax_scale = scale.has_value() ? scale : 1.0 / sqrt(query.size(-1))
```

**输入类型判断**：
- `has_only_dense_inputs` - 检查是否全部为dense tensor
- `input_requires_grad` - 检查是否需要梯度

**形状验证**：
- `check_tensor_shapes` - 验证4维输入
- `check_batch_size_and_num_heads_dense<supports_gqa>` - 模板化的批次/头数检查
- `check_attn_mask_shape` - 支持2D `[qSeq, kvSeq]` 或4D `[B, H, qSeq, kvSeq]` mask

**Stride检查**：
- `check_last_dim_stride_equals_1_dense<ignore_singleton_dim>` - 确保内存连续性

## 设计特点

1. **约束驱动的选择**：通过一系列constraint functions的组合定义backend可用性
2. **Debug友好**：所有检查函数接受`debug`参数，失败时可输出详细原因
3. **模板化设计**：使用模板参数控制GQA支持、singleton维度处理等
4. **符号形状支持**：使用`sym_size()`/`sym_stride()`支持符号整数（动态形状）
5. **渐进式回退**：Flash失败后自动降级到Math实现

## 与CUDA版本的区别

C++版本的约束更严格：
- 不支持Nested Tensor（check_nested_tensor强制返回false）
- 不支持Grouped Query Attention（check_batch_size_and_num_heads的template参数为false）
- Flash Attention要求head_dim完全相等

---

**ROCm相关**：文件中未涉及ROCm特定代码

**Backward相关**：
- `input_requires_grad` 检查是否需要梯度计算
- Nested Tensor训练时的广播限制（check_requires_grad_and_nested, check_batch_size_nested中的梯度检查）
- Attention mask不能requires_grad（check_attn_mask_shape:273-274）
