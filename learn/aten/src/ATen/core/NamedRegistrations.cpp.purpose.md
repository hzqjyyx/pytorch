这个文件定义了 PyTorch 中 Named Tensor 相关的算子注册逻辑。

## 核心功能

**默认行为设置（第 7-9 行）**
```cpp
TORCH_LIBRARY_IMPL(_, Named, m) {
  m.fallback(CppFunction::makeNamedNotSupported());
}
```
为所有算子的 Named dispatch key 设置默认回退行为：不支持 Named Tensor。这意味着除非明确注册，否则所有算子默认都不支持命名维度。

**白名单机制（第 11-534 行）**
```cpp
TORCH_LIBRARY_IMPL(aten, Named, m) {
  m.impl("abs", CppFunction::makeFallthrough());
  m.impl("add.Tensor", CppFunction::makeFallthrough());
  // ... 数百个算子
}
```
显式注册了支持 Named Tensor 的算子。`makeFallthrough()` 表示这些算子会"穿透"到下一层 dispatch key，即它们可以正常处理带命名维度的张量。

## 支持的算子类型

文件注册了这些类别的算子：
- **逐元素运算**：abs, sin, cos, exp, log 等数学函数
- **归约操作**：sum, mean, std, var 及其各种变体
- **线性代数**：mm, bmm, matmul, addmm, dot
- **形状操作**：reshape, transpose, flatten, squeeze, select
- **比较操作**：eq, ne, lt, gt, le, ge
- **逻辑运算**：logical_and, logical_or, logical_not
- **索引操作**：index_fill（包含 Dimname 变体）
- **类型转换**：to.device, to.dtype
- **特殊操作**：cat（包含 .names 变体）, chunk, split

## 命名维度专用变体

许多算子注册了专门的命名维度版本：
- `cat.names` / `cat.names_out` (91-92)
- `index_fill.Dimname_Scalar` / `Dimname_Tensor` (226-233)
- `cummax.dimname` / `cummin.dimname` (129-134)
- `transpose.Dimname` (485)
- `select.Dimname` (412)

## 特殊注册（521-534 行）

文件最后注册了一些自动微分和元数据相关的算子，这些并非官方标记为支持 Named Tensor，但为了保持向后兼容性而注册：

**主要涉及的算子：**
- ROCm 相关：无明显标记
- Backward 相关：`_backward`, `requires_grad_`, `retain_grad`, `is_leaf`, `_version`, `set_data`, `data`, `_fw_primal`, `_make_dual`
