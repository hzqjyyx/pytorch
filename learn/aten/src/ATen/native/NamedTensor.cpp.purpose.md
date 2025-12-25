# NamedTensor.cpp 核心功能分析

## 主要功能

这个文件实现了 PyTorch 的**命名张量（Named Tensor）**功能，允许用户为张量的维度指定名称而不是仅使用数字索引，提高代码可读性和安全性。

## 核心操作

### 1. 张量重命名（Rename）
- `rename_()`（原地）和 `rename()`（非原地）：为张量维度设置或修改名称
- `refine_names()`：在保持兼容性的前提下细化维度名称（允许将通配符 `*` 细化为具体名称）

### 2. 张量对齐（Alignment）
**核心概念**：将张量的维度重新排列以匹配指定的维度名称顺序

**对齐规则**（lines 151-156）：
1. 张量的维度名称序列必须是目标名称序列的子序列
2. 对齐操作不能改变未命名维度从右侧数的绝对位置

**主要函数**：
- `align_to(tensor, names)`：将张量对齐到指定的维度名称列表
- `align_to(tensor, order, ellipsis_idx)`：支持省略号（`...`）语法的对齐，自动扩展未提及的维度
- `align_as(tensor, other)`：将一个张量对齐到另一个张量的维度名称
- `align_tensors(tensors)`：将多个张量对齐到其中维度最多的张量

**实现细节**：
- `aligned_size()`：计算对齐后的张量大小（lines 81-117）
- 使用 `as_strided()` 实现维度重排，通过调整 strides 避免数据拷贝

### 3. Dimname 重载操作
为常见张量操作提供基于维度名称的重载版本：

**已实现**（调用 `dimname_to_position` 转换为位置索引）：
- `index_fill()` / `index_fill_()`：根据名称填充指定维度
- `squeeze()`：根据维度名称压缩

**未实现**（调用 `reportNYIDimnameOverload` 报错）：
- `gather()`, `index_add()`, `index_copy()`, `index_select()`
- `scatter()`, `scatter_add()`, `sort()`

## 关键辅助功能

### 错误报告
- `report_moving_unnamed_dim_error()`：检测非法的未命名维度移动
- `report_not_a_subsequence_error()`：检测维度名称不是子序列的情况

### 元数据管理
- 使用 `internal_set_names_inplace()` 设置维度名称
- 使用 `NoNamesGuard` 临时禁用名称检查以执行底层操作

## 典型使用场景示例

```cpp
// 示例：对齐操作
Tensor t = ...; // shape: [N=3, C=4, H=5, W=6]
t.align_to({'W', 'H', 'C', 'N'}); // 重排为 [6, 5, 4, 3]

// 省略号语法
t.align_to({'W', ..., 'N'}); // 扩展为 [W, C, H, N]
```

## Backward/ROCm 相关
本文件无反向传播或 ROCm 特定实现。
