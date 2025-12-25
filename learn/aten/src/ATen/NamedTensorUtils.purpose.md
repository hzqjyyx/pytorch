# NamedTensorUtils 核心功能

这两个文件实现了 PyTorch 的 **命名张量（Named Tensor）** 功能的工具函数，允许用户通过名称而非位置索引来引用张量的维度。

## 主要功能模块

### 1. 维度名称到位置的转换

```cpp
int64_t dimname_to_position(const Tensor& tensor, Dimname dim)
```

将维度名称（如 'N', 'C', 'H', 'W'）转换为对应的位置索引。例如对于 `Tensor['N', 'C', 'H', 'W']`，查询 `'H'` 会返回索引 2。

### 2. 名称统一（Unification）

```cpp
std::vector<Dimname> unify_from_right(DimnameList names, DimnameList other_names, const char* action)
```

**核心算法**：从右向左对齐两个维度名称列表，用于广播操作。规则：
- 从右边开始逐个匹配维度名称
- 相同位置的名称必须兼容（相同或至少一个是通配符）
- 检测名称错位（同一名称出现在不同相对位置）

示例：
- `['H', 'W']` + `['C', 'H', 'W']` → `['C', 'H', 'W']`
- `['N', 'H', 'W']` + `['C', 'H', 'W']` → 错误（'H' 位置不对齐）

### 3. 名称传播（Propagation）

**核心函数族**：
```cpp
propagate_names(result, names)              // 强制传播
propagate_names_if_nonempty(result, names)  // 仅当 names 非空时传播
propagate_names(result, src)                // 从源张量复制所有名称
propagate_names_except(result, src, excluded_idxs)  // 排除特定维度
```

**设计模式**：
1. **compute_outnames** 阶段：检查输入名称并计算输出名称
2. **propagate_names** 阶段：将计算的名称应用到结果张量

空名称列表是性能优化：未使用命名张量的用户零开销。

### 4. 特定操作的名称推断

#### 矩阵乘法（matmul）
```cpp
compute_matmul_outnames(self_names, other_names)
```

处理复杂的批量矩阵乘法规则：
- 统一批次维度（前 N-2 个维度）
- 向量（1D）会被完全收缩，不贡献名称
- 矩阵的特征维度：取第一个操作数的倒数第二维和第二个操作数的最后一维
- 检查输出特征维度名称不重复

示例：
- `Tensor[B, N, D] @ Tensor[B, D, M]` → `Tensor[B, N, M]`
- `Tensor[N, D] @ Tensor[D]` → `Tensor[N]`

#### 降维操作（squeeze）
```cpp
compute_squeeze_outnames(tensor)
```

移除大小为 1 的维度，同时移除对应的名称。

#### 拼接操作（cat）
```cpp
compute_cat_outnames(tensors)
```

统一所有张量的名称，确保维度数量一致。

#### 对角线操作（diagonal）
```cpp
compute_diagonal_outnames(tensor, dim1, dim2)
```

移除 `dim1` 和 `dim2` 的名称，在末尾添加通配符。

#### 扩展操作（expand）
```cpp
propagate_names_for_expand(result, self)
```

新维度使用通配符，原有维度名称右对齐。
- `Tensor[H, W].expand(3, 3, 3, 3)` → `Tensor[None, None, H, W]`

### 5. 错误检查和验证

- **位置错误检测**：`report_positional_error` - 同一位置名称不匹配
- **错位检测**：`check_for_misalignment` - 名称在不同张量中位置不一致
- **特征维度检查**：`check_feature_names_are_distinct` - matmul 输出不能有重复名称
- **名称相等性**：`are_names_equal` - 比较两个张量的名称

### 6. 复合操作

- **addmm/addmv**：`propagate_names_for_addmm/addmv` - 先计算矩阵乘法名称，再与 bias 统一
- **baddbmm**：批量矩阵乘法加法
- **bmm**：批量矩阵乘法
- **cdist**：成对距离计算，特殊的批次处理

## 关键数据结构

- `Dimname`：维度名称，可以是具体名称或通配符（`None`）
- `DimnameList`：维度名称列表（`ArrayRef<Dimname>`）
- `NameVector`：小向量优化的名称容器
- `TensorNames`：辅助类，用于操作名称切片和统一

## 性能优化策略

1. **快速路径**：如果所有输入都没有名称，直接返回空向量
2. **单元素优化**：特殊处理单个排除维度的情况
3. **小向量优化**：使用 `SmallVector` 避免小尺寸分配
4. **条件编译**：`STRIP_ERROR_MESSAGES` 可移除调试字符串

---

**ROCm/Backward 相关**：
- 本文件不涉及 ROCm 特定逻辑
- 不涉及反向传播逻辑，这是前向操作的名称管理层
