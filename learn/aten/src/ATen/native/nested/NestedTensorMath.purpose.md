# NestedTensorMath 文件功能分析

这两个文件实现了 PyTorch 中 NestedTensor（嵌套张量）的数学运算和变换操作。NestedTensor 是一种特殊的张量类型，用于高效处理不规则形状的张量集合（如变长序列）。

## 核心数据结构

NestedTensor 通过以下方式存储不规则张量：
- **buffer**: 连续存储所有子张量的数据
- **nested_sizes**: 记录每个子张量的形状
- **nested_strides**: 记录每个子张量的步幅
- **storage_offsets**: 记录每个子张量在 buffer 中的偏移

## 主要功能模块

### 1. 创建与转换操作

**`_nested_tensor_from_tensor_list`** (116-146行)
- 从张量列表创建 NestedTensor
- 验证所有张量维度一致
- 通过 TensorNode 包装创建嵌套结构

**`NestedTensor_nested_tensor_from_mask`** (64-93行)
- 从填充张量和掩码创建 NestedTensor
- 输入：`N * L * D` 的 3D 张量和 `N * L` 的布尔掩码
- 通过掩码确定每个序列的实际长度，移除填充部分

**`NestedTensor_to_padded_tensor_generic`** (246-327行)
- 将 NestedTensor 转换回填充的密集张量
- 计算所有子张量的最大尺寸
- 将每个子张量填充到最大尺寸后堆叠
- 支持自定义输出尺寸和填充值

### 2. 形状变换操作

**`view_nested` / `reshape_nested`** (820-948行)
- 实现 NestedTensor 的形状变换
- **view**: 仅支持可视化的变换（不改变内存布局）
- **reshape**: 必要时会克隆数据以实现变换
- 特殊规则：`-1` 表示"继承旧尺寸"而非推断（见 Note [Special size rule for nested tensor]）
- 批次维度（第0维）不可改变

**核心辅助函数 `NestedTensor_compute_size_stride`** (705-811行)
- 为每个子张量计算重塑后的尺寸和步幅
- 检查是否可以作为视图操作（viewable）
- 处理负数尺寸的特殊语义

**`transpose_nested`** (583-610行)
- 交换两个维度（不能包括批次维度0）
- 通过重排 size 和 stride 矩阵的列实现

**`squeeze_dim_nested` / `unsqueeze_nested`** (620-692行)
- 压缩尺寸为1的维度或添加新维度
- 不支持操作批次维度

### 3. 索引与选择操作

**`select_nested`** (428-482行)
- 在指定维度选择索引
- 维度0：返回单个子张量（普通张量）
- 其他维度：为每个子张量应用选择，返回新 NestedTensor

### 4. 数学运算

**`nested_layer_norm`** (148-197行)
- 对 NestedTensor 应用层归一化
- 要求输入连续且提供 weight 和 bias
- 调用底层 LayerNormKernel 处理展平的 buffer
- 返回：归一化输出、均值、标准差倒数

**`NestedTensor_sum_dim_CPU`** (357-426行)
- 仅支持对最后一维归约求和
- 要求 `keepdim=True`
- 手动遍历每个子张量的每个段进行求和

**`softmax_nested`** (505-541行)
- 对每个子张量独立应用 softmax
- 不能跨批次维度应用

**`NestedTensor_all`** (543-581行)
- 逻辑全真判断的归约操作
- 对每个子张量独立应用

### 5. 神经网络操作

**`NestedTensor_embedding`** (329-354行)
- 在 NestedTensor 索引上应用嵌入查找
- 输入：权重矩阵和 NestedTensor 索引
- 输出尺寸在最后添加嵌入维度

**`native_dropout_nested`** (484-503行)
- 对 NestedTensor 应用 dropout
- 在 buffer 上应用标准 dropout，保持嵌套结构

### 6. 拼接操作

**`cat_nested`** (1098-1101行)
- 沿指定维度拼接多个 NestedTensor

**`cat_nested_as_jagged`** (1021-1069行)
- 处理维度 > 0 的拼接
- 要求：仅支持单个不规则维度的结构 `(B, *, D_0, D_1, ...)`
- 将每个 NT 视为 jagged 布局后拼接 buffer

### 7. 视图转换边界

**`values_nested`** (858-862行)
- 从 NestedTensor 获取底层 buffer（非嵌套张量视图）

**`_nested_view_from_buffer`** (873-911行)
- 从普通张量 buffer 创建 NestedTensor 视图
- 验证尺寸、步幅、偏移的边界合法性

### 8. 工具函数

**`num_bytes`** (25-38行)
- 计算给定尺寸的连续张量所需的内存字节数

**`pad_tensor_to_shape`** (40-60行)
- 将张量填充到目标形状
- 使用 `constant_pad_nd` 实现

**`can_cat_nested_sizes`** (995-1018行)
- 检查两个 NestedTensor 是否可以在指定维度拼接
- 要求：拼接维度外的所有维度尺寸必须匹配

## 头文件 (NestedTensorMath.h)

### 模板工具

**`map_nt`** (14-19行)
- 对单个 NestedTensor 的 buffer 应用函数，保持嵌套结构

**`map_nt_binary`** (20-26行)
- 对两个 NestedTensor 的 buffer 应用二元函数

**`_check_nested_layer_norm_inputs`** (28-75行)
- 验证 layer_norm 的输入参数
- 检查 normalized_shape 不能扩展到不规则维度
- 计算归一化参数 M 和 N

## 设计特点

1. **内存效率**: 所有子张量数据存储在单个连续 buffer 中
2. **批次维度限制**: 大多数操作不允许改变或推断批次维度（第0维）
3. **连续性要求**: 许多操作要求输入为连续张量
4. **Jagged 布局支持**: 部分操作支持特定的 jagged 布局（单个不规则维度）
5. **视图语义**: 尽可能使用视图操作避免数据拷贝

---

**ROCm/Backward 相关内容**:
- 文件中未直接包含 ROCm 特定代码
- 未实现反向传播函数（backward 由 autograd 系统在其他位置处理）
- 部分操作（如 layer_norm、dropout）返回前向结果，反向传播由调用方处理
