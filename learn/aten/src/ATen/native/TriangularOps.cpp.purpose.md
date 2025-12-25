## 主要功能

这个文件实现了 PyTorch 中的 **三角矩阵操作** (triu/tril)，即提取矩阵的上三角和下三角部分。

### 核心实现

**元数据定义** (`at::meta` 命名空间)
- `tril` 和 `triu` 的元函数，定义输出张量的形状和选项
- 验证输入张量至少有 2 维

**CPU 计算实现**

1. **`apply_triu_tril_single<scalar_t>`** (42-83 行)
   - 单个矩阵的核心处理函数
   - 参数 `k` 控制对角线偏移：k > 0 上偏移，k < 0 下偏移
   - `upper=true` 时提取上三角，`upper=false` 时提取下三角
   - 支持 in-place 操作和非 in-place 操作
   - 使用 `parallel_for` 并行处理行

2. **`apply_triu_tril<scalar_t>`** (85-125 行)
   - 处理批量矩阵和多维张量
   - 计算 stride，支持 3D+ 批量张量
   - 调用 `apply_triu_tril_single` 处理每个批次

3. **`compute_triu_tril<Triangle>`** (137-170 行)
   - 统一接口，通过模板参数区分上/下三角
   - 处理内存连续性问题
   - 使用 `AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4` 支持多种数据类型

**CPU 入口点**
- `tril_cpu` 和 `triu_cpu` (174-180 行)

### 关键特性

- **支持的数据类型**：所有标量类型 + 复数 + Half + BFloat16 + Bool
- **对角线偏移**：参数 `k` 控制对角线位置
- **批量处理**：支持任意维度的张量
- **内存效率**：检测 in-place 操作，避免不必要复制
- **并行计算**：使用 `parallel_for` 加速

### 主要函数

- **`tril(Tensor, k)`**：提取下三角矩阵
- **`triu(Tensor, k)`**：提取上三角矩阵
- **`trace_backward_symint`**：trace 操作的反向传播（梯度计算）
