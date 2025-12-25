# ReduceOpsUtils.h 核心功能

这个头文件为 PyTorch 的张量降维（reduction）操作提供了一套完整的工具函数和辅助设施。

## 1. 边界值定义

```cpp
template <typename scalar_t>
constexpr scalar_t upper_bound() / lower_bound()
```

为各种标量类型提供上下界，支持浮点数的无穷大和整数的极值。用于初始化 min/max 操作。

## 2. 降维结果处理

**核心流程三部曲：**

- `resize_reduction_result()`: 根据 DimMask 调整输出张量大小
- `review_reduce_result()`: 在不改变 keepdim 的情况下，重新以 strided 视图查看结果
- `shape_from_dim_mask()`: 从维度掩码计算输出形状

**示例：** 对形状 `[3, 4, 5]` 的张量在 dim=1 上降维：
- 如果 `keepdim=true`：输出形状 `[3, 1, 5]`
- 如果 `keepdim=false`：输出形状 `[3, 5]`

## 3. TensorIterator 构建

`make_reduction()` 函数族是降维操作的核心工厂方法：

**单输出版本：**
```cpp
make_reduction(name, result, self, dim, keepdim, out_dtype)
```
- 处理类型提升（如 Half→Float）
- GPU 混合精度优化路径
- 传播命名张量信息

**双输出版本：**
```cpp
make_reduction(name, result1, result2, self, dim, keepdim, dtype1, dtype2)
```
用于同时返回值和索引的操作（如 `max()`、`min()`）

## 4. 维度掩码系统

```cpp
DimMask make_dim_mask(OptionalIntArrayRef opt_dims, int64_t ndim)
```

使用 bitset 表示要降维的维度：
- `None` 或空数组 → 全部维度（全降维）
- `[0, 2]` → 只降维第 0 和第 2 个维度

## 5. 边界情况处理

**零元素张量：**
```cpp
zero_numel_check_dims()           // 检查维度合法性
get_zero_numel_tensor_size()      // 计算输出形状
zero_numel_tensor_resize()        // 调整输出大小
```

**平凡降维：**
```cpp
_dimreduce_return_trivial()       // 有单位元的降维（如 sum 返回 0）
_allreduce_return_trivial()       // 全降维的特殊情况
```

## 6. 类型推断辅助

```cpp
get_dtype_from_self()      // 从输入推断输出类型，可选整数提升
integer_upcast()           // 整数类型提升为 Long
```

## 7. Meta 函数支持

`at::meta` 命名空间提供了符号形状推断功能：

```cpp
resize_reduction()                    // 设置输出元信息
resize_reduction_with_indices()       // 双输出元信息
get_reduction_shape()                 // 纯形状计算
```

用于 PyTorch 的编译器和 meta tensor 系统。

## 8. 工具函数

- `restride_dim()`: 将某个维度的步长设为 0（广播语义）
- `check_scalar_type_device_layout_equal()`: 验证张量兼容性
- `make_dim_vector()`: 将可选维度参数转为具体维度列表

---

**简要提及（已忽略详细分析）：**
- ROCm 相关优化路径
- Backward 梯度传播支持
