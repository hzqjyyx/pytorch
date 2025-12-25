# Sorting.cpp/h 主要功能

这两个文件实现了 PyTorch 的排序和选择相关操作。

## 核心算法

**quick_select_template** (132-186行)
- 基于 Sedgewick 1978年论文的快速选择算法实现
- 用于在未排序数组中查找第k个元素
- 使用三数取中法选择pivot
- 特殊处理NaN值（作为最大值）
- 通过模板化的比较函数和swap函数支持不同数据类型

## 主要操作

### 1. topk (54-71, 828-847行)
- 返回张量在指定维度的前k个最大/最小值及其索引
- 元函数验证k的范围，设置输出形状
- 调用 `topk_stub` 分发到具体设备实现

### 2. sort (73-92, 943-998行)
- 对张量在指定维度排序，返回排序后的值和索引
- 支持stable参数控制稳定排序
- 不支持复数类型（78-81行检查）
- 优化stride以避免过度内存分配（86-88行）
- 调用 `sort_stub` 分发实现
- **argsort**: 只返回排序索引
- **msort**: 沿第0维排序

### 3. kthvalue (417-501, 782-826行)
- 返回第k小的值及其索引
- CPU实现：克隆输入数据，使用quick_select找到第k个元素
- 使用TensorIterator并行处理多个slice
- NaN被视为最大值以兼容numpy

### 4. median/nanmedian (504-652, 849-941行)
- **median_with_indices_impl**: 计算中位数及索引
  - 使用间接索引避免修改原数据
  - 用std::nth_element进行部分排序
  - median遇到NaN直接返回NaN
  - nanmedian忽略NaN只计算非NaN值的中位数
  
- **median_impl**: 计算标量中位数
  - 克隆数据后直接使用std::nth_element

### 5. quantile/nanquantile (190-780行)

**辅助函数**:
- `get_quantile_interpolation_mode`: 解析插值模式字符串（linear/lower/higher/midpoint/nearest）
- `quantile_checks`: 验证输入合法性（非空、q为标量或1D、浮点类型、设备匹配）
- `quantile_output_shape`: 计算输出形状

**核心计算** (249-370行):
1. 对输入排序（flatten或沿指定维度）
2. 将q值转换为索引ranks
   - quantile: 基于完整大小计算，遇NaN设为最后索引
   - nanquantile: 基于非NaN数量计算
3. 根据插值模式调整ranks
   - LOWER/HIGHER/NEAREST: 取floor/ceil/round
   - LINEAR/MIDPOINT: 计算上下界并插值
4. 用gather取对应位置的值，LINEAR/MIDPOINT模式下使用lerp插值
5. 处理Composite Compliance Tensor (CCT)的特殊情况（305-308, 342-358行）

## 辅助工具

**_fill_indices** (101-112行)
- 用0到dim_size-1填充索引张量
- 使用arange + as_strided实现广播填充

**Dispatch机制** (98-99行)
```cpp
DEFINE_DISPATCH(sort_stub);
DEFINE_DISPATCH(topk_stub);
```
分发到CPU/CUDA等不同设备的具体实现。

## 设计特点

- **并行化**: 使用TensorIterator + grain_size控制并行粒度
- **内存优化**: stride推断、原地操作、避免不必要分配
- **Named tensor支持**: 通过NoNamesGuard和propagate_names_for_reduction处理
- **类型分发**: AT_DISPATCH_ALL_TYPES_AND2宏支持多种数据类型包括BFloat16/Half
- **NaN处理**: median系列操作对NaN有明确语义区分

---

**忽略内容**:
- ROCm相关实现
- Backward/梯度计算相关代码
