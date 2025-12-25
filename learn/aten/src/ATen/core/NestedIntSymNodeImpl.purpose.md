## NestedIntSymNodeImpl 概述

这个文件实现了PyTorch中的**嵌套整数符号节点**，用于表示和处理Jagged张量（不规则张量）的形状信息。

### 核心概念

**NestedIntSymNodeImpl** 将不规则张量的形状结构编码为单个整数。例如，一个形状为 `[B, [s_0, s_1, s_2], D]` 的Jagged张量可以用 `j0` 代表中间的可变长度维度。

### 主要设计

1. **两个成员变量**：
   - `val_`：整数ID，代表特定的形状模式
   - `coeff_`：系数，用于计算stride（步幅）。例如 `[B, j0, D]` 可以有不同的stride方案：`[D*j0, D, 1]` 或 `[j0, 1, sum(j0)]`

2. **值范围假设**：
   - NestedInt 被视为在范围 `[2, int64_t::max]` 内的已知整数
   - 这允许回答 `j0 >= 1`（True）和 `j0 == 0`（False）这样的查询

### 比较操作实现

**_eq()** 函数：
- 检查两个nested int的ID和系数是否完全相等

**_ge()** 函数：
- 当两个nested int的ID相同时，比较系数
- 当ID不同时，只允许与常数值为≤2的int进行比较（利用值范围假设）
- 不确定的关系会抛出错误

### 支持的操作

- **比较**：`eq`, `ne`, `ge`, `gt`, `lt`, `le`
- **乘法**：`mul`（只能与常数相乘，不能与另一个nested int相乘）
- **克隆**：`clone`

### 不支持的操作

通过宏定义编译时拒绝以下操作：
- 二元操作：`add`, `sub`, `truediv`, `pow`, `floordiv`, `mod`, `sym_min`, `sym_max`, `sym_and`, `sym_or`
- 一元操作：`sym_not`, `ceil`, `floor`, `neg`, `sym_float`

### 主要特性

- **字符串表示**：`j0` 或 `coeff*j0` 格式
- **类型信息**：是整数但不是浮点数或布尔值
- **符号性**：标记为非符号化的值（`is_symbolic() = false`）

---

**关键点总结：**

- Jagged张量形状编码为单个整数ID加系数
- 比较基于值范围假设 `[2, int64_t::max]`
- 不确定关系会报错而不是返回False
- 仅支持与常数的乘法
- 设计目的是在追踪时表达输出形状和stride作为输入的函数
