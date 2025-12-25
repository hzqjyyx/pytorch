# TensorShape.cpp/h 核心功能理解

## 主要职责

这两个文件实现 PyTorch 张量的**形状操作（shape manipulation）**核心功能，包括视图变换、拼接、切片、重塑等不改变数据内容但改变张量结构的操作。

## 核心功能模块

### 1. 张量拼接 (Concatenation)

**`cat` 系列函数** (行235-757)
- **元函数** `TORCH_PRECOMPUTE_META_FUNC(cat)`: 计算输出形状、检查兼容性
  - 跳过空张量 `cat_should_skip_tensor()`: 仅跳过形状为 [0] 的 1 维张量（向后兼容）
  - 内存格式推断 `cat_compute_output_memory_format()`: 优先保持非 Contiguous 格式
  - 类型提升 `result_type()`: 确定输出数据类型
  - 输出形状计算: 除拼接维度外其他维度必须相同

- **CPU 实现** `cat_out_cpu` (行668-757)
  - **快速路径1** (dim=0, 连续内存): 直接 `memcpy` 拼接
  - **快速路径2** (单线程, 浮点类型): 调用 `cat_serial_stub`
  - **通用路径**: 使用 `TensorIterator` + `copy_stub` 逐个复制切片

- **稀疏张量拼接** `cat_sparse_impl` (行857-977)
  - **稀疏维度拼接**: 拼接 indices，累加偏移量
  - **密集维度拼接**: 需要在 values 两侧填充零张量

**别名函数**: `concat`, `concatenate` 都映射到 `cat`

### 2. 堆叠 (Stacking)

虽然文件被截断，但从 `DEFINE_DISPATCH(stack_serial_stub)` 可见包含 `stack` 操作（在新维度上拼接）。

### 3. 切片与窄化 (Slicing & Narrowing)

**`narrow` 系列** (行1663-1737)
- **视图版本** `narrow()`: 返回 `slice(self, dim, start, start+length, 1)` 的视图
- **符号整数版本** `narrow_symint()`: 支持动态形状
- **复制版本** `narrow_copy_dense_cpu()` (行1511-1661): 
  - 按块复制数据 `memcpy(dst + i*dst_block, src + i*src_block, block_size)`
  - 用于需要物理复制而非视图的场景

**稀疏窄化** `narrow_copy_sparse()` (行1522-1567)
- 稀疏维度: 通过掩码过滤索引
- 密集维度: 直接在 values 上窄化

### 4. 分块 (Chunking & Splitting)

**`chunk`** (行1069-1089): 将张量分成 N 块
- 计算分块大小: `split_size = (dim_size + chunks - 1) / chunks`
- 特殊处理零尺寸: 调用 `split_with_sizes` 保持块数

**`tensor_split`** (行1091-1199)
- **按数量分割** `tensor_split_sections_symint()`: 尽可能均分
- **按索引分割** `_tensor_split_indices()`: 在指定位置切分
- **张量索引版本**: 支持传入包含切分点的张量

**`unsafe_chunk/split`**: 跳过部分检查的快速版本

### 5. 对角线操作 (Diagonal)

**`diagonal`** (行1226-1309)
- 通过 `as_strided` 实现视图提取:
  - 删除 dim1, dim2 维度
  - 添加新维度，步长为 `stride[dim1] + stride[dim2]`
  - 支持正负偏移量

**`diag_embed`** (行1311-1334): 逆操作，将向量嵌入对角线
- 创建零张量，用 `diagonal().copy_(self)` 填充

**`diagflat`** (行1222-1224): 展平后提取对角线

### 6. 维度变换 (Permutation & Transposition)

**`permute`** (行1776-1780)
- 重新排列维度顺序
- 通过 `as_strided` 调整 sizes 和 strides

**`permute_sparse_coo`** (行1782-1854)
- **限制**: 稀疏/密集维度间不能转置
- 分别对稀疏索引和密集值进行 permute

### 7. 重复与平铺 (Repeat & Tile)

**`repeat`** (行1856-1902)
- 先 `expand` 添加前导维度
- 用 `unfold` + `copy_` 实现重复
- 支持量化张量

**`tile`** (行1904-1919): NumPy 风格接口
- 自动在前面填充 1 对齐维度
- 内部调用 `repeat`

### 8. 重塑 (Reshape)

**`reshape_symint`** (行1991-2000, 截断)
- 连续张量: 返回 `view`（零拷贝）
- 非连续张量: 需要推断形状并可能复制数据

**辅助函数**:
- `_reshape_from_tensor()`: 从张量读取目标形状
- `_shape_as_tensor()`: 将形状转为张量

### 9. 扩展 (Expansion)

**`expand`** (行1336-1364)
- 广播规则: 尺寸为 1 的维度可扩展
- 通过 `inferExpandGeometry_dimvector` 计算新 strides
- 返回 `as_strided` 视图（步长为 0 实现广播）

**`expand_as`**: 扩展到另一张量的形状

**`sparse_broadcast_to`** (行514-642): 稀疏张量广播
- 计算索引重复因子 `nnz_expand_factor`
- 广播 values 和 indices

### 10. `as_strided` 底层实现

**核心函数** `as_strided_tensorimpl` (行1401-1417)
- 创建共享 Storage 的新 TensorImpl
- 设置自定义 sizes/strides/storage_offset
- 量化版本 `as_strided_qtensorimpl` (行1452-1497)

**符号整数版本** `as_strided_tensorimpl_meta_symint` (行1429-1450)
- 支持动态形状，使用 `setStridedUnchecked` 避免生成 guard

### 11. 存储设置 (Storage Setting)

**`set_` 系列** (行367-512)
- `set_storage_cpu_`: 绑定到指定 Storage + offset/size/stride
- `set_storage_meta__symint`: 符号形状版本，计算并调整 storage nbytes
- `set_tensor_`: 使另一张量共享源张量的 Storage
- `set_cpu_/set_meta_`: 创建空 Storage 重置张量

### 12. 工具函数

**`alias_with_sizes_and_strides`** (行1925-1989)
- 创建共享 Storage 的别名，但使用不同 sizes/strides
- 模板化支持 IntArrayRef 和 SymIntArrayRef

**`block_diag`** (行988-1067)
- 构造块对角矩阵
- 扩展输入为 2D，在零矩阵上复制到对角块

## 头文件 TensorShape.h

**辅助检查函数**:
- `cat_should_skip_tensor()`: 判断是否跳过空张量
- `check_cat_shape_except_dim()`: 验证拼接兼容性
- `check_cat_no_zero_dim()`: 禁止 0 维张量拼接
- `get_num_splits()`: 计算分块数量
- `preprocess_chunk_cat_inputs()`: `_chunk_cat` 输入验证

**导出函数**:
- `clone_preserve_strides()`: 保留步长的克隆

## 设计特点

1. **视图优先**: 大部分操作通过 `as_strided` 避免数据复制
2. **性能优化**: 多条快速路径（连续内存、单线程、特定数据类型）
3. **符号形状**: 广泛支持 `SymInt` 用于动态形状追踪
4. **稀疏支持**: 针对 COO 格式实现专用路径
5. **量化支持**: `QTensorImpl` 处理量化张量的特殊逻辑

---

**其他涉及但未详述的内容**:
- ROCm/CUDA 相关优化路径
- Backward 操作（`diagonal_backward`, `select_backward`, `slice_backward`）
- MKL-DNN 特化路径（`_mkldnn_reshape`, `_mkldnn_transpose`）
- 命名张量（Dimname）接口
- `unfold`/`unflatten` 等高级视图操作
- 稀疏压缩格式（CSR/CSC）张量构造
