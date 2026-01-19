这个文件实现了 PyTorch functorch 库中 scatter/gather/index 等操作的批处理规则（batch rules），用于支持 vmap（向量化映射）功能。

## 核心功能

### 1. 高级索引（Advanced Indexing）批处理

**index_batch_rule** (176-318行)
- 处理 `tensor[indices]` 形式的高级索引操作
- 关键挑战：处理"相邻高级索引"（adjacent advanced indices）问题
  - 当索引列表中的张量索引相邻时，结果维度位置符合直觉
  - 当索引不相邻时，NumPy 规范要求新维度出现在最前面
  - 批处理后可能改变索引的相邻性，需要通过 `swap_regions` 重排维度

**batchIndices** (59-123行)
- 将索引列表转换为批处理版本
- 三种情况：
  1. 仅 self 批处理：在索引前插入 None 来广播
  2. 仅索引批处理：无需修改
  3. self 和索引都批处理：插入 arange 索引批次维度

### 2. index_put 操作批处理

**index_put_batch_rule** (607-648行)
- 处理 `tensor[indices] = values` 赋值操作
- 使用 `index_put_batch_rule_helper` 准备批处理的 self、indices、values
- 调用 `maybe_permute_values` 确保 values 的形状与索引结果匹配
- 支持 accumulate 模式（累加而非覆盖）

**index_put_batch_rule_helper** (404-454行)
- 统一处理 self、indices、values 的批次维度对齐
- 计算索引后的形状 `get_indexed_shape`
- 处理 values 的广播：当 indexed_shape 维度更多时，插入单位维度

### 3. scatter/gather 操作批处理

**scatter_batch_rule** (676-743行，模板函数)
- 统一处理 scatter 系列操作：scatter、scatter_add、scatter_reduce
- 两个重载版本：
  - 标量值版本：`scatter(self, dim, index, scalar_value)`
  - 张量值版本：`scatter(self, dim, index, src_tensor)`
- 处理逻辑：
  - 移动批次维度到最前面
  - 处理 0 维张量（unsqueeze）
  - 确保所有张量都有批次维度（通过 `ensure_has_bdim`）
  - 调整物理维度索引（+1 因为批次维度在前）

**gather_batch_rule** (816-844行)
- 从 self 中按 index 收集元素
- 结果形状与 index 相同
- 处理 0 维张量的特殊情况

### 4. index_add 操作批处理

**index_add_batch_rule_impl** (949-1019行)
- 处理 `self.index_add(dim, index, other, alpha)`
- 两种策略：
  - index 未批处理：直接调用批处理版本的 index_add
  - index 批处理：使用 for 循环逐个处理每个批次（因为缺乏通用的批处理 kernel）

### 5. index_fill 操作批处理

**index_fill_batch_rule_helper** (1076-1110行)
- 处理 `self.index_fill(dim, index, value)`
- 策略：
  - 当 self 非标量时：为每个批次的索引添加偏移量，然后展平批次维度
  - 当 self 是标量时：展平索引的批次维度

### 6. 分解实现（Decompositions）

**index_select_decomp** (870-890行)
- 将 `index_select` 分解为 `gather` 操作
- 扩展 index 以匹配 self 的维度

**index_copy_decomp** (892-902行)
- 将 `index_copy` 分解为 `scatter` 操作

**slice_scatter_decomp** (909-916行)
- 将 `slice_scatter` 分解为 `scatter` 操作
- 生成 arange 索引表示切片范围

**select_scatter_decomp** (918-927行)
- 将 `select_scatter` 分解为 `scatter` 操作
- 将标量索引转换为张量并扩展

### 7. 辅助工具函数

- **any_has_value**: 检查是否有任何批次维度存在
- **get_num_leading_nones**: 计算索引列表前导 None 的数量
- **get_max_index_logical_dim**: 获取索引张量的最大逻辑维度
- **is_advanced_index**: 判断是否为高级索引
- **are_advanced_indices_adjacent**: 判断高级索引是否相邻
- **swap_regions**: 交换张量中两个区域的维度顺序
- **binary_pointwise_align**: 对齐两个张量用于逐点操作

## 其他相关内容

- **masked_fill_scalar_batch_rule**: 处理掩码填充操作
- **diagonal_scatter_batch_rule**: 处理对角线散布操作
- **compute_indexed_shape/get_indexed_shape**: 计算索引后的张量形状
- **unpackSelfAndIndicesAndValuesAtCurrentLevel**: 解包当前 vmap 层级的张量和批次维度
