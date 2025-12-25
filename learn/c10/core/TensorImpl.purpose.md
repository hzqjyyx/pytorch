# TensorImpl 核心功能解析

## 1. 核心职责

TensorImpl 是 PyTorch 中**张量的底层实现**，负责管理张量的所有元数据和实际数据。它是一个可引用计数的对象（继承自 `intrusive_ptr_target`），将张量的"视图信息"与"存储数据"分离。

## 2. 关键数据成员

### 存储管理
- `storage_`: 指向实际数据的 Storage 对象，多个张量可共享同一 Storage
- `storage_offset_`: 该张量在 Storage 中的偏移量（元素数，非字节数）
- `data_type_`: 数据类型元信息（TypeMeta）

### 形状与步长
- `sizes_and_strides_`: 存储张量的尺寸和步长信息
- `numel_`: 元素总数缓存
- `has_symbolic_sizes_strides_`: 是否使用符号形状（用于动态形状场景）

### 调度系统
- `key_set_`: DispatchKeySet，决定该张量如何分派操作（如 CPU/CUDA/Autograd 等）
- `device_opt_`: 设备信息（CPU/CUDA/etc）

### 内存布局标志
- `is_contiguous_`: 是否连续存储（C 连续）
- `is_channels_last_contiguous_`/`is_channels_last_3d_contiguous_`: 通道优先布局
- `is_non_overlapping_and_dense_`: 是否无重叠且紧密

### 扩展元数据
- `extra_meta_`: 可选的额外元数据，包含：
  - `symbolic_shape_meta_`: 符号形状信息
  - `named_tensor_meta_`: 命名张量元数据
  - `backend_meta_`: 后端特定元数据
  - 自定义错误消息

### 自动微分
- `autograd_meta_`: 自动微分元数据（梯度、requires_grad 等）
- `version_counter_`: 版本计数器，追踪原地修改

## 3. 主要功能模块

### 构造与初始化
```cpp
// c10/core/TensorImpl.cpp:79-178
```
- 多种构造函数，支持传入 Storage 或仅元数据
- **Python key removal**: 新建 TensorImpl 时移除 Python 调度键（lines 86-101），因为 PyObject 尚未创建
- 根据 InferenceMode 状态决定是否添加 Autograd 相关键（lines 148-170）
- 初始化版本计数器（非推理张量）

### 形状与步长访问
```cpp
// c10/core/TensorImpl.h:614-813
```
提供多层访问模式：
1. **快速路径**: `sizes()`/`strides()` - 直接返回内部数组
2. **策略检查**: 通过 `matches_policy(SizesStridesPolicy)` 判断是否需要自定义行为
3. **自定义路径**: `sizes_custom()`/`strides_custom()` - 允许子类（如 NestedTensor）重载
4. **符号形状**: `sym_sizes()`/`sym_strides()` - 支持符号整数

### 内存布局检查
```cpp
// c10/core/TensorImpl.cpp:221-274
// c10/core/TensorImpl.h:822-883
```
- `compute_contiguous()`: 计算张量是否按 C 顺序连续
- `compute_channels_last_contiguous_2d/3d()`: NHWC/NDHWC 布局检查
- `compute_non_overlapping_and_dense()`: 检查元素是否无重叠
- 对稀疏张量直接返回 false

### 尺寸与步长修改
```cpp
// c10/core/TensorImpl.cpp:846-982
// c10/core/TensorImpl.h:1747-1897
```

**整数版本**:
- `set_sizes_and_strides(IntArrayRef, IntArrayRef)`: 直接设置
- `set_sizes_contiguous(IntArrayRef)`: 设置连续布局
- `empty_tensor_restride(MemoryFormat)`: 根据内存格式重新计算步长

**符号版本**:
- `set_sizes_and_strides(SymIntArrayRef, SymIntArrayRef)`: 设置符号形状
- `generic_set_sizes_contiguous(SymIntArrayRef)`: 符号形状连续布局
- `empty_tensor_restride_symint()`: 符号形状步长计算（lines 915-982）

关键约束检查：
- `allow_tensor_metadata_change()`: 是否允许修改元数据（detach/data 视图限制）
- 符号形状需要 `extra_meta_->symbolic_shape_meta_`

### 自动微分集成
```cpp
// c10/core/TensorImpl.cpp:40-75, 449-486
```
- `mutable_grad()`/`grad()`: 获取梯度张量
- `set_requires_grad()`: 设置是否需要梯度（推理张量限制检查）
- `_fw_grad()`/`_set_fw_grad()`: 前向模式自动微分支持
- 延迟初始化 `autograd_meta_`（通过工厂模式）

### 浅拷贝与分离
```cpp
// c10/core/TensorImpl.cpp:487-637
```
- `shallow_copy_and_detach()`: 创建共享存储但独立 autograd 历史的副本
- 优先调用 Python 解释器的 `detach()` 如果存在 `__torch_dispatch__`
- `copy_tensor_metadata()`: 复制所有元数据（除了 key_set_/storage_/version_counter_ 等）
- `copy_generic_tensor_metadata()`: 复制通用元数据（用于包装张量，如 functionalization）

### 存储管理
```cpp
// c10/core/TensorImpl.cpp:759-832
```
- `FreeMemory()`: 释放存储，保留元数据（用于延迟分配）
- `ShareData()`: 与另一张量共享存储
- `ShareExternalPointer()`: 使用外部数据指针（不拥有内存）
- `HandleResize()`: 根据 `caffe2_keep_on_shrink` 标志决定是否释放内存

### Legacy Caffe2 操作
```cpp
// c10/core/TensorImpl.cpp:641-757
```
- `Extend()`: 扩展第一维，支持增长百分比
- `ReserveSpace()`: 预留空间但不改变尺寸
- `Reshape()`: 必须保持元素数不变的形状变换

### 自定义行为支持
```cpp
// c10/core/TensorImpl.cpp:313-422
```
通过虚函数允许子类重载：
- `is_contiguous_custom()`/`is_strides_like_custom()`
- `sizes_custom()`/`strides_custom()`/`dim_custom()`
- `device_custom()`/`layout_custom()`
- Python 自定义通过 `pyobj_slot_.load_pyobj_interpreter()` 调用

### 调度键管理
```cpp
// c10/core/TensorImpl.cpp:180-197
```
- `_change_backend_component_keys()`: 更换设备时更新后端相关键
- 处理 Autocast 键的迁移（目前 Autocast 还是全局的，未来会变成 per-backend）

### 错误处理
```cpp
// c10/core/TensorImpl.cpp:290-311
```
- `throw_cannot_call_with_symbolic()`: 符号形状张量上调用不支持的方法
- `throw_storage_access_error()`/`throw_data_ptr_access_error()`: 自定义存储访问错误消息

## 4. 策略机制

### SizesStridesPolicy (TensorImpl.h:930-944)
分三级定制：
- **Default**: 标准密集张量
- **CustomStrides**: 可自定义步长（稀疏/mkldnn）
- **CustomSizes**: 可自定义尺寸和维度（嵌套张量）

实现通过位域字段：
- `sizes_strides_policy_`: 实际策略
- `custom_sizes_strides_`: 标记是否为自定义实现
- `python_custom_sizes_strides_`: 标记是否为 Python 自定义

## 5. 版本追踪
```cpp
// c10/core/TensorImpl.h:327-417
```
`VariableVersion` 结构：
- 共享版本计数器用于视图、detach、保存变量
- `bump()`: 原地操作时递增版本（推理张量跳过）
- `unique()`: 检查版本计数器是否唯一持有
- 推理张量的 `version_counter_` 为禁用状态

## 6. 数据访问
```cpp
// c10/core/TensorImpl.h:1515-1659
```
- `data<T>()`/`mutable_data<T>()`: 类型检查的数据访问
- `data_ptr_impl<T>()`: 高效的非类型检查访问（供 Tensor 使用）
- 处理零尺寸张量时避免 UB（不对 nullptr 做偏移计算）
- 检查存储是否初始化

## 7. 位域优化
TensorImpl.h 中大量使用位域来节省内存：
- 布局标志（连续性、通道优先等）
- 策略标志
- 元数据变更权限
- Python/符号形状等状态

---

## ROCm/Backward 相关（简要）
- 代码中无明显 ROCm 特定逻辑（设备类型通过 DispatchKey 抽象）
- Backward 相关主要在 `autograd_meta_` 中封装，TensorImpl 仅提供访问接口和版本追踪
