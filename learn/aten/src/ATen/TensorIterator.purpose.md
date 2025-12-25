# TensorIterator 核心功能解析

## 1. 总体职责

TensorIterator 是 PyTorch 中处理**元素级操作**（element-wise operations）的核心基础设施，负责：
- 统一处理广播（broadcasting）
- 类型转换和类型提升
- 内存布局优化
- 多操作数的迭代协调

## 2. 核心机制

### 2.1 配置与构建流程

通过 `TensorIteratorConfig` 构建，必须按顺序添加：
1. 先添加所有输出张量（`add_output`）
2. 再添加所有输入张量（`add_input`）

构建过程 (`build` 方法) 执行以下步骤：
1. **populate_operands**: 填充操作数信息
2. **mark_outputs**: 标记输出张量，检测读写冲突
3. **compute_mem_overlaps**: 检查内存重叠
4. **compute_names**: 计算命名维度
5. **compute_shape**: 通过广播规则计算最终形状
6. **compute_types**: 类型推断和提升
7. **fast_set_up** 或完整路径：
   - **compute_strides**: 计算步长
   - **reorder_dimensions**: 重排维度以优化访问
   - **allocate_or_resize_outputs**: 分配输出内存
   - **coalesce_dimensions**: 合并相邻维度

### 2.2 维度重排优化

`reorder_dimensions()` 实现关键的性能优化：
- 按步长升序排列维度（反转 C 连续顺序，使 `strides[0]` 成为最快移动维度）
- reduction 操作中将 reduced 维度移到前面（步长为 0）
- 使用插入排序，支持模糊比较（处理广播的 0 步长）
- 相同步长时通过维度大小打破平局

### 2.3 维度合并

`coalesce_dimensions()` 合并相邻维度的条件：
- 某个维度大小为 1，或
- `shape[n] * stride[n] == stride[n+1]`（连续性）

合并后可能将多维迭代降为 1D，极大提升性能。

### 2.4 类型处理系统

**通用 dtype 计算**（`compute_common_dtype`）：
- 使用 `at::native::ResultTypeState` 进行类型提升
- 仅考虑输入张量，输出不参与

**配置标志控制行为**：
- `promote_inputs_to_common_dtype_`: 将输入提升到通用类型
- `cast_common_dtype_to_outputs_`: CPU 上创建临时输出，计算完成后转换回原类型
- `promote_integer_inputs_to_float_`: 整数类型提升为默认浮点类型
- `enforce_safe_casting_to_output_`: 检查通用类型能否安全转换到输出类型

### 2.5 快速路径优化

`fast_set_up()` 检测特殊情况避免完整构建：
- **CONTIGUOUS**: 所有张量都是连续的
- **CHANNELS_LAST**: 所有张量都是 channels-last 布局
- **NON_OVERLAPPING_DENSE**: 所有张量形状相同且步长一致

快速路径直接将迭代降为 1D，跳过维度重排和逐步合并。

## 3. 迭代执行

### 3.1 循环接口

```cpp
void for_each(loop2d_t loop, int64_t grain_size)
```
- 支持并行化（grain_size 控制粒度）
- 内部循环处理 2D 切片（最快的两个维度）
- 1D lambda 会自动转换为 2D

### 3.2 32 位索引分割

`SplitUntil32Bit` 递归分割超过 32 位索引范围的迭代器：
- `can_use_32bit_indexing()` 检查 `numel` 和最大偏移
- `split(dim)` 在指定维度分割为两个子迭代器
- `get_dim_to_split()` 选择最大 extent 的维度

## 4. 数据结构

### 4.1 OperandInfo

存储每个操作数的元数据：
- `data`: 实际数据指针
- `stride_bytes`: 广播后的字节步长
- `target_dtype` / `current_dtype`: 类型信息
- `device`: 设备信息
- `is_output` / `is_read_write` / `is_const`: 语义标记
- `will_resize`: 是否需要调整大小
- `tensor_base_` / `original_tensor_base_`: 张量引用（支持类型转换临时张量）

### 4.2 TensorIteratorBase 核心字段

- `shape_`: 计算形状（经过重排和合并）
- `perm_`: 维度重排的排列映射
- `operands_`: 所有操作数（输出在前）
- `common_dtype_` / `common_device_`: 计算类型和设备
- `has_coalesced_dimensions_`: 是否已合并维度

## 5. 典型使用场景

### 5.1 二元操作
```cpp
TensorIterator::binary_op(out, a, b)
  // 启用: 内存重叠检查、CPU 标量、类型提升、安全转换
```

### 5.2 比较操作
```cpp
TensorIterator::comparison_op(out, a, b)
  // 输出未定义时强制为 bool 类型
  // 输出为 bool 时跳过类型转换（性能优化）
```

### 5.3 Reduction 操作
```cpp
TensorIterator::reduce_op(out, a)
  // is_reduction=true, resize_outputs=false
  // 输出形状与输入不匹配（已归约）
```

## 6. 内存优化细节

- **步长计算**: 广播维度步长设为 0
- **数据指针调整**: `narrow()` 和 `select_all_keeping_dim()` 通过指针偏移实现切片
- **临时张量管理**: `exchange_tensor()` / `restore_original_tensor()` 支持类型转换临时输出
- **原地操作检测**: `is_read_write` 标记输出同时是输入的情况

## 7. 设备支持

- CPU: 完整支持，包括类型转换临时张量
- CUDA/XLA/Lazy/IPU/MTIA/MAIA/HPU: 跳过数据指针初始化（无存储或特殊处理）
- Meta 张量: `is_meta_` 标记，仅计算元数据不执行

---

**ROCm 相关**:
- `GemmHipblaslt.h` / `GemmRocblas.h`: HIP/ROCm BLAS 后端
- `TunableGemm.h`: 可调优矩阵乘法

**Backward 相关**:
- TensorIterator 本身不直接处理反向传播
- 用于实现前向和反向元素级核函数
- `VariableFallbackKernel.cpp` 处理 autograd 回退
