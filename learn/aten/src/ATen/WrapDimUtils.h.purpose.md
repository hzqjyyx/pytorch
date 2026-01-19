这个文件提供了一组用于处理张量维度索引的工具函数，主要解决负数索引转换和维度范围检查的问题。

## 核心功能

### 1. 基础维度包装 - `maybe_wrap_dim`

将负数维度索引转换为正数索引，例如对于3维张量：
- `dim=-1` → `dim=2`（最后一维）
- `dim=-2` → `dim=1`（倒数第二维）

提供了多个重载版本：
- `maybe_wrap_dim(dim, TensorImpl*)` - 基于 TensorImpl 对象
- `maybe_wrap_dim(dim, TensorList)` - 基于张量列表（使用第一个张量的维度）
- `maybe_wrap_dim(dim, vector<vector<int64_t>>)` - 基于形状向量

### 2. 批量维度包装 - `maybe_wrap_dims_n` / `maybe_wrap_dims`

对维度数组进行原地包装转换，同时进行范围检查：
- 检查每个维度是否在 `[-dim_post_expr, dim_post_expr-1]` 范围内
- 将负数索引转换为正数索引
- `wrap_scalars` 参数：标量张量（0维）是否允许指定维度0或-1

### 3. 特殊处理 - `legacy_cat_wrap_dim`

为 `cat` 操作提供向后兼容的维度包装逻辑：
- 跳过形状为 `[0]` 的空张量（历史遗留行为）
- 使用第一个非空张量的维度数进行包装
- 提供了三个版本：
  - 基于 `vector<vector<int64_t>>`
  - 基于 `vector<vector<SymInt>>`（符号整数，用于动态形状）
  - 基于 `MaterializedITensorListRef`

### 4. 批量包装 - `wrap_all_dims`

对 `vector<int64_t>` 中的所有维度进行包装转换。

## 设计要点

- **原地修改**：`maybe_wrap_dims_n` 和 `wrap_all_dims` 直接修改传入的数组
- **错误处理**：使用 `TORCH_CHECK_INDEX` 进行维度越界检查，提供清晰的错误信息
- **标量张量特殊处理**：0维张量在 `wrap_scalars=true` 时允许使用维度0或-1
- **空列表处理**：空张量列表直接返回原始 dim，由底层实现处理错误

---

**其他内容：**
- 无 ROCm 相关内容
- 无 Backward 相关内容
