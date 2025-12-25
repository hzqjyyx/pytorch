# ExpandUtils 核心功能

这组文件提供 PyTorch 张量的**广播（broadcasting）**和**扩展（expansion）**工具函数。

## 主要功能模块

### 1. 形状推断 (infer_size)

实现 NumPy 风格的广播规则，用于推断两个形状如何广播到兼容的形状。

**实现逻辑** (aten/src/ATen/ExpandUtils.cpp:17-42):
- 从最右侧（最后一维）开始对齐两个形状
- 对每一维：
  - 如果两个尺寸相等，使用该尺寸
  - 如果其中一个是 1，使用另一个尺寸（包括 0）
  - 否则抛出错误
- 返回广播后的形状

**变体函数**:
- `infer_size()` - 返回 `std::vector<int64_t>`
- `infer_size_symint()` - 支持符号整数的版本
- `infer_size_dimvector()` - 返回 `DimVector`（栈优化的小型向量）
- `infer_size_symdimvector()` - 符号整数版的 DimVector

### 2. 扩展几何计算 (inferExpandGeometry)

给定张量的当前形状/步长和目标形状，计算扩展后的新形状和步长。

**实现逻辑** (aten/src/ATen/ExpandUtils.cpp:62-114):
- 从右向左遍历目标形状的每一维
- 对每一维：
  - 如果目标尺寸是 `-1`，保持原尺寸
  - 如果原尺寸是 1，设置步长为 0（实现广播的关键）
  - 如果原尺寸不是 1 且与目标不匹配，抛出错误
- 返回 `InferExpandGeometryResult<Container>` 包含新的 sizes 和 strides

**步长为 0 的意义**：这是实现广播的内存技巧，使得同一个元素在该维度上被重复访问。

### 3. 密集步长推断 (infer_dense_strides)

将非密集或重叠的步长转换为密集且非重叠的步长，同时保持内存布局顺序。

**实现逻辑** (aten/src/ATen/ExpandUtils.cpp:148-225):
1. 使用插入排序对维度排序，排序依据：
   - 首先按步长从小到大
   - 步长相同时，尺寸大的维度排后面
   - 步长为 0 的维度不移动
2. 按排序后的顺序重新计算密集步长：
   - 从步长最小的维度开始，设为 1
   - 后续维度步长 = 前一个步长 × 前一个维度尺寸

**注意事项**：该函数假设输入确实是非密集的，不会检查；如果输入已经是密集的，仍会执行完整计算造成性能浪费。

### 4. 就地扩展辅助 (expand_inplace)

将 `to_expand` 张量扩展到与 `tensor` 相同的形状。

**优化设计** (aten/src/ATen/ExpandUtils.h:99-107):
- 如果形状已经相同，返回 `MaybeOwned::borrowed(to_expand)`（零拷贝）
- 否则返回 `MaybeOwned::owned(to_expand.expand(...))`（新张量）

**重载版本**:
- 单张量版本
- 双张量版本（同时扩展两个张量）
- 带 `api_name` 的版本（用于错误消息）

**安全机制**：所有接受右值引用的版本都被 `= delete` 删除，防止悬空引用（见 NOTE [ExpandUtils Borrowing]）。

### 5. 外部扩展辅助 (expand_outplace)

推断多个张量的广播形状，并将它们都扩展到该形状。

**实现策略** (aten/src/ATen/ExpandUtils.h:187-201):
- 2 张量版本：使用 `infer_size_symdimvector()` 推断形状
- 3 张量版本：先推断前两个，再与第三个推断
- TensorList 版本：遍历所有张量，逐步推断最终形状
- 同样使用 `MaybeOwned` 优化

### 6. 尺寸扩展 (expand_size)

将张量扩展到指定的 `IntArrayRef` 尺寸。

### 7. 反向求和 (sum_to)

将张量通过求和缩减到目标形状，是扩展的"反向"操作。

**实现逻辑** (aten/src/ATen/ExpandUtils.h:449-483):
- 找出所有需要缩减的维度：
  - 前导维度（张量维度多于目标形状）
  - 目标形状为 1 但张量该维度不为 1 的维度
- 使用 `tensor.sum(reduce_dims, keepdim=true)` 求和
- 如果有前导维度，使用 `view()` 或 `view_copy()` 调整形状
- `always_return_non_view` 参数用于函数化 pass，确保返回非视图

### 8. 可扩展性检查

**`are_expandable()`** (aten/src/ATen/ExpandUtils.h:60-73):
- 检查两个形状是否可以广播
- 逻辑与 `infer_size_impl` 保持同步

**`is_expandable_to()`** (aten/src/ATen/ExpandUtils.h:501-525):
- 检查 `shape` 是否可以扩展到 `desired`
- 单向检查：`shape` 的每一维必须等于 `desired` 或为 1

### 9. 辅助工具

**`check_defined()`** (aten/src/ATen/ExpandUtils.h:76-84):
- 检查初始化列表中的所有张量是否已定义
- 使用 `reference_wrapper` 避免拷贝构造

## 设计要点

### MaybeOwned 借用优化

所有返回 `c10::MaybeOwned<Tensor>` 的函数遵循 NOTE [ExpandUtils Borrowing]:
- 如果不需要扩展，返回借用引用（零开销）
- 如果需要扩展，返回新拥有的张量
- 删除所有右值引用重载，防止临时对象导致的悬空引用

### 符号整数支持

大多数函数都有 `SymInt` 版本，支持 PyTorch 的符号形状系统（用于动态形状和 tracing）。

### 容器类型灵活性

使用模板支持多种容器类型：
- `std::vector<int64_t>` - 标准版本
- `DimVector` - 小型向量优化（通常避免堆分配）
- 对应的 `SymInt` 版本

---

## 简要说明

**ROCm 相关**：无直接 ROCm 代码，但 CUDA 相关的扩展逻辑在运行时会应用到 ROCm 后端。

**Backward 相关**：`sum_to()` 函数是许多前向扩展操作的梯度实现基础，在反向传播时用于将梯度缩减回原始形状。
