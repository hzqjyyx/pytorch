## NestedTensorBinaryOps 核心功能

这两个文件实现了 **NestedTensor 的二元运算操作**，包括加减乘除、比较、填充等。

### 主要架构

**1. 核心模板函数：`NestedTensor_elementwise_Tensor`**

处理 NestedTensor 之间或与普通 Tensor 的元素级运算，支持三种场景：

- **标量运算**（lines 84-102）：当 self 或 other 是标量时，直接对底层 buffer 操作，复用原有的嵌套结构
- **NestedTensor + Dense Tensor（CUDA 专用）**（lines 104-165）：
  - `[B, *, D] + [B, 1, D]` 的 3D 广播：调用优化的 CUDA kernel（`nested_dense_elementwise_stub`）
  - `[B, *] + [B, 1]` 的 2D 广播：转换为 3D 处理
  - `[B, C, *, *] + [C, 1, 1]` 的 4D 广播：逐元素 unbind 后运算
- **NestedTensor + NestedTensor**（lines 170-179）：
  - 通过 `get_elementwise_nested_tensor_impl` 严格验证维度、嵌套大小、步幅、偏移量完全匹配
  - 直接对底层 buffer 进行运算，复用嵌套结构元数据

**2. 验证函数：`get_elementwise_nested_tensor_impl`**（lines 22-72）

确保两个 NestedTensor 可以进行元素级运算：
- 检查双方都是 NestedTensor
- 验证维度相同（不支持广播）
- 验证嵌套大小（`nested_sizes`）完全相等
- 验证步幅（`nested_strides`）完全相等
- 验证存储偏移量（`storage_offsets`）逐一匹配

**3. 原地运算模板：`NestedTensor_elementwise__Tensor`**（lines 236-262）

处理原地修改的二元运算（`add_`, `mul_` 等）：
- 直接修改 self 的底层 buffer
- 同样处理标量和 NestedTensor 两种情况
- 将 buffer 展平为 1D 视图后运算

### 具体运算实现

**只读运算**（返回新 Tensor）：
- `NestedTensor_add_Tensor`：加法，支持 alpha 缩放（lines 182-190）
- `NestedTensor_sub_Tensor`：减法，支持 alpha 缩放（lines 192-200）
- `NestedTensor_mul_Tensor/Scalar`：乘法（lines 202-212）
- `NestedTensor_div_Tensor/Scalar`：除法（lines 214-224）
- `NestedTensor_masked_fill`：根据 mask 填充值（lines 225-233）
- 比较运算：`ge_scalar_nested`, `gt_scalar_nested`, `eq_scalar_nested`, `eq_tensor_nested`（lines 298-329）

**原地运算**（修改原 Tensor）：
- `NestedTensor_add__Tensor`（lines 264-272）
- `NestedTensor_mul__Tensor/Scalar`（lines 274-284）
- `fill_nested_`：用标量或 Tensor 填充整个 NestedTensor（lines 286-296）

### 头文件定义（NestedTensorBinaryOps.h）

- **枚举 `NESTED_DENSE_OP`**：定义 ADD 和 MUL 两种优化操作类型
- **函数指针类型 `nested_dense_elementwise_fn`**：CUDA kernel 的调度接口
- **调度声明 `DECLARE_DISPATCH`**：声明 `nested_dense_elementwise_stub` 用于设备特定的实现

### 设计特点

1. **零拷贝优化**：所有运算都直接操作底层连续 buffer，避免解包/重新打包嵌套结构
2. **严格的结构匹配**：NestedTensor 间运算不支持广播，要求所有元数据完全一致
3. **CUDA 加速路径**：针对特定广播模式提供专门的 CUDA kernel
4. **分层抽象**：通过模板函数和 lambda 表达式复用验证逻辑，减少代码重复

---

**简要提及的其他内容：**
- ROCm 相关：`REGISTER_NO_CPU_DISPATCH` 宏用于注册 CPU 空实现
- Backward 相关：文件中无反向传播相关代码（二元运算的梯度由 autograd 自动处理）
