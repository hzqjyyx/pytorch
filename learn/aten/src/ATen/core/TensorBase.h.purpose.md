## TensorBase 核心功能

TensorBase 是 PyTorch 的核心数据结构，作为 Tensor 类的基类存在。它的主要设计目标是**减少编译依赖**，提升增量编译速度。

### 设计理念

- **打破头文件依赖**：与 Tensor 不同，TensorBase 不包含代码生成的方法（来自 native_functions.yaml），因此修改算子签名时不需要重新编译整个 ATen
- **引用计数句柄**：TensorBase 本质上是 TensorImpl 的引用计数智能指针封装 (`intrusive_ptr<TensorImpl, UndefinedTensorImpl>`)
- **继承关系**：Tensor 继承自 TensorBase，所以接受 `const TensorBase&` 的函数可以直接使用 Tensor

### 主要功能分类

**1. 基础属性访问**
- 维度信息：`dim()`, `size()`, `sizes()`, `stride()`, `strides()`
- 符号形状支持：`sym_size()`, `sym_stride()`, `sym_sizes()` (用于动态形状)
- 元素信息：`numel()`, `itemsize()`, `element_size()`, `nbytes()`
- 存储信息：`storage_offset()`, `has_storage()`, `storage()`

**2. 数据类型与设备**
- 类型查询：`dtype()`, `scalar_type()`, `is_floating_point()`, `is_complex()`, `is_signed()`
- 设备判断：`device()`, `is_cpu()`, `is_cuda()`, `is_xpu()`, `is_mps()` 等各种后端检查
- 布局与格式：`layout()`, `is_sparse()`, `is_quantized()`, `is_meta()`

**3. 内存布局与连续性**
- 连续性检查：`is_contiguous()`, `is_non_overlapping_and_dense()`
- 格式建议：`suggest_memory_format()` (返回 Contiguous/ChannelsLast/ChannelsLast3d)
- 连续化操作：`contiguous()`, `expect_contiguous()` (后者优化了引用计数)

**4. 数据访问**
- 原始指针：`data_ptr()`, `const_data_ptr()`, `mutable_data_ptr()`
- 类型化指针：模板方法 `data_ptr<T>()`, `const_data_ptr<T>()`, `mutable_data_ptr<T>()`
- 结构化访问：`accessor()`, `packed_accessor32()`, `packed_accessor64()` (用于 CPU/CUDA kernel)

**5. TensorImpl 管理**
- 获取实现：`unsafeGetTensorImpl()`, `getIntrusivePtr()`
- 释放所有权：`unsafeReleaseTensorImpl()`, `unsafeReleaseIntrusivePtr()`
- 生命周期：`defined()`, `reset()`, `use_count()`, `weak_use_count()`
- 创建包装：静态方法 `wrap_tensor_impl()`

**6. 基础操作**
- 填充操作：`fill_()`, `zero_()`
- 设备转换：`to()` (支持选项、非阻塞、拷贝、内存格式)
- 类型转换：通过 TensorOptions

**7. 特殊标记位**
- Zero tensor：`_is_zerotensor()`, `_set_zero()`
- 共轭标记：`is_conj()`, `_set_conj()` (lazy conjugation)
- 负数标记：`is_neg()`, `_set_neg()` (lazy negation)
- 嵌套 tensor：`is_nested()`

**8. 命名维度支持**
- 名称查询：`names()`, `opt_names()`, `has_names()`
- 元数据访问：`get_named_tensor_meta()`

**9. Autograd 接口**
- 梯度追踪：`requires_grad()`, `set_requires_grad()`, `requires_grad_()`
- 叶子节点：`is_leaf()` (用于判断是否在 backward 时填充梯度)
- 视图关系：`is_view()`, `_base()` (获取视图的基础 tensor)
- 梯度函数：`grad_fn()` (获取创建该 tensor 的操作节点)
- 数据分离：`tensor_data()`, `variable_data()`, `data()`
- 版本追踪：`_version()` (用于检测 in-place 修改)
- Hook 机制：`register_hook()`, `remove_hook()` (注册梯度回调)
- 梯度保留：`retain_grad()`, `retains_grad()` (非叶子节点梯度保存)

**10. Forward AD 支持**
- 前向梯度：`_fw_grad()`, `_set_fw_grad()` (支持多层级前向模式自动微分)

**11. 辅助功能**
- 别名检查：`is_alias_of()` (检查两个 tensor 是否共享存储)
- 共享内存：`share_memory_()` (转换为跨进程共享的 shm 存储)
- 调试输出：`print()`, `toString()`
- 配置获取：`options()` (返回 TensorOptions)

### 特殊设计细节

**unsafe_borrow_t 构造函数**
- 创建 +0 引用计数的 Tensor，用于 `MaybeOwned<Tensor>` 优化
- 避免不必要的引用计数增减，提升性能

**expect_contiguous()**
- 如果已连续，返回 borrowed reference（无引用计数开销）
- 如果不连续，返回 owned contiguous copy
- 禁止右值调用（避免悬空引用）

**编译时检查**
- 可在文件顶部 `#define TORCH_ASSERT_NO_OPERATORS` 确保无 native_functions.yaml 依赖

### ROCm 相关
- `is_hip()`: 检查是否为 AMD HIP 后端

### Backward 相关
详见第 677-900 行的 Autograd API 部分，包括：
- `is_leaf()`, `backward()`, `detach()`, `detach_()`
- `retain_grad()`, `retains_grad()`
- `grad_fn()`, `output_nr()`, `set_data()`
- Hook 注册机制
- Forward AD 梯度管理
